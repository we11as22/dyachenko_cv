import os
import glob
import random
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Config ----
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT      = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR  = os.path.join(ROOT, "..", "data")
CKPT_DIR  = os.path.join(ROOT, "task_4_kpconv", "checkpoints")
METR_DIR  = os.path.join(ROOT, "task_4_kpconv", "metrics")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METR_DIR, exist_ok=True)

NUM_POINTS   = 4096
NUM_CLASSES  = 10  # fallback; auto-detected at runtime
BATCH_SIZE   = 12
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 15
K            = 24
NUM_KP       = 15
SIGMA_RATIO  = 1.0
DIMS         = [64, 128, 256, 512]
NPOINTS      = [4096, 1024, 256, 64]
DROPOUT      = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---- PLY reader ----
def read_ply(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_end = num_vertices = 0; prop_names = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"): num_vertices = int(s.split()[-1])
        if s.startswith("property"): prop_names.append(s.split()[-1])
        if s == "end_header": header_end = i+1; break
    xi,yi,zi=prop_names.index("x"),prop_names.index("y"),prop_names.index("z")
    li=prop_names.index("scalar_Label")
    xyz=np.zeros((num_vertices,3),np.float32); labels=np.zeros(num_vertices,np.int64)
    for j,dl in enumerate(lines[header_end:header_end+num_vertices]):
        v=dl.strip().split(); xyz[j]=[float(v[xi]),float(v[yi]),float(v[zi])]; labels[j]=int(float(v[li]))
    return xyz, labels


# ---- Dataset ----
class PointCloudDataset(Dataset):
    def __init__(self, file_list, num_points=NUM_POINTS, augment=False):
        self.file_list=file_list; self.num_points=num_points; self.augment=augment

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        xyz,labels=read_ply(self.file_list[idx])
        n=xyz.shape[0]
        choice=np.random.choice(n,self.num_points,replace=(n<self.num_points))
        xyz,labels=xyz[choice],labels[choice]
        xyz-=xyz.mean(0); d=np.max(np.linalg.norm(xyz,axis=1))
        if d>1e-8: xyz/=d
        if self.augment:
            # yaw only (rotation around Z) — correct for ground-based scenes
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ Rz.T
            # random scale
            xyz = xyz * np.random.uniform(0.9, 1.1)
            # jitter
            xyz += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
        return torch.from_numpy(xyz), torch.from_numpy(labels)


# ---- KPConv ----
def init_kernel_points(num_kp, ndims=3):
    if num_kp == 1: return torch.zeros(1, ndims)
    pts = F.normalize(torch.randn(num_kp, ndims), dim=-1)
    for _ in range(100):
        dists = torch.cdist(pts, pts) + torch.eye(num_kp)*1e6
        forces = torch.zeros_like(pts)
        for i in range(num_kp):
            diff = pts[i]-pts; d = dists[i].unsqueeze(-1)+1e-6
            forces[i] = (diff/(d**2)).sum(0)
        pts = F.normalize(pts + 0.01*forces, dim=-1)
    return pts

def knn_query(xyz, k):
    N = xyz.shape[1]
    k = min(k, N - 1)
    inner=-2*torch.bmm(xyz,xyz.transpose(1,2))
    sq=torch.sum(xyz**2,dim=-1,keepdim=True)
    dist=sq+inner+sq.transpose(1,2)
    return dist.topk(k+1,dim=-1,largest=False)[1][:,:,1:]

def fps(xyz, npoint):
    B,N,_=xyz.shape
    centroids=torch.zeros(B,npoint,dtype=torch.long,device=xyz.device)
    dist=torch.full((B,N),1e10,device=xyz.device)
    farthest=torch.randint(0,N,(B,),device=xyz.device)
    bi=torch.arange(B,device=xyz.device)
    for i in range(npoint):
        centroids[:,i]=farthest
        c=xyz[bi,farthest].unsqueeze(1)
        d=torch.sum((xyz-c)**2,dim=-1)
        dist=torch.min(dist,d); farthest=dist.argmax(-1)
    return centroids

class KPConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, num_kp, k, sigma=0.5):
        super().__init__(); self.k=k; self.num_kp=num_kp
        self.sigma = sigma  # fixed constant, not computed from batch
        self.register_buffer("kernel_pts", init_kernel_points(num_kp))
        self.kernel_weights=nn.Parameter(torch.randn(num_kp,in_ch,out_ch)*math.sqrt(2.0/in_ch))
        self.bn=nn.BatchNorm1d(out_ch)

    def forward(self, xyz, feat):
        B,N,C=feat.shape
        idx=knn_query(xyz,self.k)                             # (B,N,k)
        flat=idx.reshape(B,-1).unsqueeze(-1)                  # (B,N*k,1)
        kxyz=xyz.gather(1, flat.expand(-1,-1,3)).reshape(B,N,self.k,3)
        kfeat=feat.gather(1, flat.expand(-1,-1,C)).reshape(B,N,self.k,C)
        rel=kxyz-xyz.unsqueeze(2)
        kp_dist=torch.norm(rel.unsqueeze(3)-self.kernel_pts.reshape(1,1,1,-1,3),dim=-1)
        corr=torch.clamp(1.0-kp_dist/self.sigma,min=0.0)
        out=torch.einsum('bnkK,bnkC,KCo->bno',corr,kfeat,self.kernel_weights)
        return F.relu(self.bn(out.reshape(B*N,-1)).reshape(B,N,-1),True)

class KPConvStage(nn.Module):
    def __init__(self, in_dim, out_dim, npoint, num_kp, k, sigma):
        super().__init__(); self.npoint=npoint
        self.kpc=KPConvLayer(in_dim,out_dim,num_kp,k,sigma)

    def forward(self, xyz, feat):
        # 1. KPConv on the full cloud at current resolution
        feat=self.kpc(xyz,feat)
        # 2. FPS downsampling (vectorized)
        idx=fps(xyz,self.npoint)                              # (B, npoint)
        idx3=idx.unsqueeze(-1)
        sxyz=xyz.gather(1, idx3.expand(-1,-1,3))
        sf=feat.gather(1, idx3.expand(-1,-1,feat.shape[-1]))
        return sxyz, sf

class KPConvUp(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_dim+skip_dim,out_dim,bias=False),
                               nn.BatchNorm1d(out_dim),nn.ReLU(True))

    def forward(self, xyz1, xyz2, feat1, feat2):
        # interpolate feat2 (sparse, xyz2) to xyz1 (dense) using 3-NN
        dists = torch.cdist(xyz1, xyz2)                    # (B, N_dense, N_sparse)
        knn_dists, idx = dists.topk(3, dim=-1, largest=False)  # (B, N, 3)
        w = 1.0 / (knn_dists + 1e-6)
        w = w / w.sum(-1, keepdim=True)                    # (B, N, 3)
        B, N, _ = idx.shape
        # gather feat2 neighbors: (B, N, 3, C)
        feat2_exp = feat2.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N_sparse, C)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, feat2.shape[-1])
        neighbors = feat2_exp.gather(2, idx_exp)           # (B, N, 3, C)
        interp = (neighbors * w.unsqueeze(-1)).sum(2)      # (B, N, C)
        x = torch.cat([feat1, interp], -1)
        B_, N_, C_ = x.shape
        return self.mlp(x.reshape(B_ * N_, C_)).reshape(B_, N_, -1)

class KPConvSemSeg(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dims=DIMS, npoints=NPOINTS,
                 num_kp=NUM_KP, k=K, sigma=SIGMA_RATIO, dp=DROPOUT):
        super().__init__()
        self.stem=nn.Linear(3,dims[0],bias=False)
        self.stages=nn.ModuleList([KPConvStage(dims[i],dims[i+1],npoints[i+1],num_kp,k,sigma) for i in range(len(dims)-1)])
        # ups: from dims[-(i+1)] with skip dims[-(i+2)] -> dims[-(i+2)]
        self.ups=nn.ModuleList([
            KPConvUp(dims[-(i+1)], dims[-(i+2)], dims[-(i+2)])
            for i in range(len(dims)-1)
        ])
        self.head=nn.Sequential(nn.Linear(dims[0],128),nn.BatchNorm1d(128),nn.ReLU(True),nn.Dropout(dp),nn.Linear(128,num_classes))

    def forward(self, xyz):
        feat=F.relu(self.stem(xyz),True); skips=[(xyz,feat)]
        for stage in self.stages:
            xyz,feat=stage(xyz,feat); skips.append((xyz,feat))
        # decoder: start from bottom, chain through up blocks
        _, feat = skips[-1]
        for i, up in enumerate(self.ups):
            xyz_dense, feat_dense = skips[-(i+2)]   # skip from encoder
            xyz_sparse, _ = skips[-(i+1)]            # coarser coordinates
            feat = up(xyz_dense, xyz_sparse, feat_dense, feat)
            xyz = xyz_dense
        B,N,C=feat.shape
        return self.head(feat.reshape(B*N,C)).reshape(B,N,-1)


# ---- Helpers ----
def compute_metrics(preds, labels, num_classes):
    oa=(preds==labels).sum()/len(labels); ious=[]
    for c in range(num_classes):
        i=((preds==c)&(labels==c)).sum(); u=((preds==c)|(labels==c)).sum()
        ious.append(i/u if u>0 else float('nan'))
    return oa, np.nanmean(ious), np.array(ious)

@torch.no_grad()
def evaluate(model, loader, cw, num_classes):
    model.eval(); all_p,all_l=[],[]; tl,n=0.0,0
    for xyz,labels in loader:
        xyz,labels=xyz.to(DEVICE),labels.to(DEVICE)
        logits=model(xyz)
        loss=F.cross_entropy(logits.reshape(-1,num_classes),labels.reshape(-1),weight=cw)
        tl+=loss.item()*xyz.size(0); n+=xyz.size(0)
        all_p.append(logits.argmax(-1).cpu().numpy().flatten())
        all_l.append(labels.cpu().numpy().flatten())
    p,l=np.concatenate(all_p),np.concatenate(all_l); oa,miou,ious=compute_metrics(p,l,num_classes)
    return tl/n,oa,miou,ious,p,l


def main():
    all_files=sorted(glob.glob(os.path.join(DATA_DIR,"*.ply")))
    print(f"Found {len(all_files)} PLY files"); assert len(all_files)>0

    # Auto-detect number of classes
    print("Scanning labels...")
    all_labels_set = set()
    for f in tqdm(all_files, desc="Scanning", leave=False):
        _, lbl = read_ply(f)
        all_labels_set.update(lbl.tolist())
    num_classes = max(all_labels_set) + 1
    print(f"Detected {num_classes} classes: {sorted(all_labels_set)}")

    random.shuffle(all_files)
    n_tr=int(0.70*len(all_files)); n_val=int(0.15*len(all_files))
    tr,val,te=all_files[:n_tr],all_files[n_tr:n_tr+n_val],all_files[n_tr+n_val:]
    print(f"Train: {len(tr)}, Val: {len(val)}, Test: {len(te)}")
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id); random.seed(SEED + worker_id)
    mk=dict(num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    g=torch.Generator(); g.manual_seed(SEED)
    train_loader=DataLoader(PointCloudDataset(tr,augment=True),BATCH_SIZE,shuffle=True,drop_last=True,generator=g,**mk)
    val_loader  =DataLoader(PointCloudDataset(val),BATCH_SIZE,**mk)
    test_loader =DataLoader(PointCloudDataset(te),BATCH_SIZE,**mk)

    cc=np.zeros(num_classes,dtype=np.float64)
    for f in tr:
        _,lbl=read_ply(f)
        for c in range(num_classes): cc[c]+=(lbl==c).sum()
    freq=cc/cc.sum(); w=1.0/(freq+1e-6); w=w/w.sum()*num_classes
    cw=torch.tensor(w,dtype=torch.float32).to(DEVICE)

    model=KPConvSemSeg(num_classes).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=5,min_lr=1e-6)

    best_miou,patience_cnt=0.0,0
    hist={"train_loss":[],"val_loss":[],"val_oa":[],"val_miou":[]}
    ckpt=os.path.join(CKPT_DIR,"best_kpconv.pth")

    print("\n=== Training ===")
    for epoch in range(1,EPOCHS+1):
        model.train(); rl,ns=0.0,0
        for xyz,labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            xyz,labels=xyz.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(xyz).reshape(-1,num_classes),labels.reshape(-1),weight=cw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
            optimizer.step(); rl+=loss.item()*xyz.size(0); ns+=xyz.size(0)
        tl=rl/ns
        vl,voa,vmiu,_,_,_=evaluate(model,val_loader,cw,num_classes)
        scheduler.step(vmiu)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["val_oa"].append(voa); hist["val_miou"].append(vmiu)
        print(f"Epoch {epoch:3d}/{EPOCHS} | LR {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train {tl:.4f} | Val {vl:.4f} | OA {voa:.4f} | mIoU {vmiu:.4f}")
        if vmiu>best_miou:
            best_miou=vmiu; patience_cnt=0
            torch.save(model.state_dict(),ckpt); print(f"  -> Best mIoU: {best_miou:.4f}")
        else: patience_cnt+=1
        if patience_cnt>=PATIENCE: print(f"\nEarly stopping at epoch {epoch}"); break

    print("\n=== Test ===")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
    else:
        print("Warning: no checkpoint found, evaluating with current weights")
    _,toa,tmiou,tious,tp,tl_arr=evaluate(model,test_loader,cw,num_classes)
    tf1=f1_score(tl_arr,tp,average="macro",zero_division=0)
    cm=confusion_matrix(tl_arr,tp,labels=list(range(num_classes)))
    print(f"Test OA: {toa:.4f} | mIoU: {tmiou:.4f} | F1: {tf1:.4f}")
    for c in range(num_classes):
        print(f"  Class {c}: IoU {tious[c]:.4f}" if not np.isnan(tious[c]) else f"  Class {c}: IoU N/A")

    # Save CSV metrics
    csv_path = os.path.join(METR_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model","test_oa","test_miou","test_f1"] + [f"class_{c}_iou" for c in range(num_classes)])
        row = ["kpconv", f"{toa:.4f}", f"{tmiou:.4f}", f"{tf1:.4f}"]
        row += [f"{tious[c]:.4f}" if not np.isnan(tious[c]) else "N/A" for c in range(num_classes)]
        writer.writerow(row)
    print(f"Results saved to {csv_path}")

    er=range(1,len(hist["train_loss"])+1)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(er,hist["train_loss"],label="Train"); ax.plot(er,hist["val_loss"],label="Val")
    ax.set_title("Loss"); ax.legend(); ax.grid(True,alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR,"loss_curves.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(er,hist["val_miou"],label="mIoU"); ax.plot(er,hist["val_oa"],label="OA")
    ax.set_title("Val mIoU & OA"); ax.legend(); ax.grid(True,alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR,"miou_acc_curves.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(9,8))
    cm_n=cm.astype("float")/(cm.sum(axis=1,keepdims=True)+1e-8)
    sns.heatmap(cm_n,annot=True,fmt=".2f",cmap="Blues",ax=ax,vmin=0,vmax=1)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR,"confusion_matrix.png"),dpi=150); plt.close()
    print(f"Metrics saved to {METR_DIR}")


if __name__ == "__main__":
    main()
