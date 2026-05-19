import os
import glob
import random
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
CKPT_DIR  = os.path.join(ROOT, "task_2_dgcnn", "checkpoints")
METR_DIR  = os.path.join(ROOT, "task_2_dgcnn", "metrics")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METR_DIR, exist_ok=True)

NUM_POINTS  = 4096
NUM_CLASSES = 10  # fallback; auto-detected at runtime
BATCH_SIZE  = 8
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 50
PATIENCE    = 10
K           = 20
EMB_DIMS    = 1024
DROPOUT     = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---- PLY reader ----
def read_ply(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_end = num_vertices = 0
    prop_names = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"): num_vertices = int(s.split()[-1])
        if s.startswith("property"): prop_names.append(s.split()[-1])
        if s == "end_header": header_end = i+1; break
    xi,yi,zi = prop_names.index("x"),prop_names.index("y"),prop_names.index("z")
    li = prop_names.index("scalar_Label")
    xyz = np.zeros((num_vertices,3),np.float32); labels = np.zeros(num_vertices,np.int64)
    for j,dl in enumerate(lines[header_end:header_end+num_vertices]):
        v = dl.strip().split()
        xyz[j]=[float(v[xi]),float(v[yi]),float(v[zi])]; labels[j]=int(float(v[li]))
    return xyz, labels


# ---- Dataset ----
class PointCloudDataset(Dataset):
    def __init__(self, file_list, num_points=NUM_POINTS, augment=False):
        self.file_list=file_list; self.num_points=num_points; self.augment=augment

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        xyz, labels = read_ply(self.file_list[idx])
        n = xyz.shape[0]
        choice = np.random.choice(n, self.num_points, replace=(n<self.num_points))
        xyz, labels = xyz[choice], labels[choice]
        xyz -= xyz.mean(0); d = np.max(np.linalg.norm(xyz,axis=1))
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


# ---- DGCNN ----
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2,1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    dist = -xx - inner - xx.transpose(2,1)
    return dist.topk(k=k, dim=-1)[1]

def get_graph_feature(x, k=20):
    B,C,N = x.shape
    idx = knn(x, k)
    xt = x.transpose(2,1).contiguous()
    feat = xt.view(B*N,-1)[idx.view(B*N*k)].view(B,N,k,C)
    xt = xt.unsqueeze(2).expand(-1,-1,k,-1)
    return torch.cat([xt, feat-xt], dim=3).permute(0,3,1,2)

class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__(); self.k=k
        self.net = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.net(get_graph_feature(x, self.k)).max(-1)[0]

class DGCNNSemSeg(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, k=K, emb=EMB_DIMS, dp=DROPOUT):
        super().__init__()
        self.ec1=EdgeConv(3,64,k); self.ec2=EdgeConv(64,64,k)
        self.ec3=EdgeConv(64,64,k); self.ec4=EdgeConv(64,128,k)
        self.conv5=nn.Sequential(nn.Conv1d(320,emb,1,bias=False),nn.BatchNorm1d(emb),nn.LeakyReLU(0.2,True))
        self.head=nn.Sequential(
            nn.Conv1d(emb+320,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(0.2,True),nn.Dropout(dp),
            nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(0.2,True),nn.Dropout(dp),
            nn.Conv1d(256,128,1,bias=False),nn.BatchNorm1d(128),nn.LeakyReLU(0.2,True),
            nn.Conv1d(128,num_classes,1))

    def forward(self, xyz):
        x = xyz.permute(0,2,1)
        x1=self.ec1(x); x2=self.ec2(x1); x3=self.ec3(x2); x4=self.ec4(x3)
        xc=torch.cat([x1,x2,x3,x4],1)
        x5=self.conv5(xc)
        xg=x5.max(-1,keepdim=True)[0].expand_as(x5)
        return self.head(torch.cat([xg,xc],1)).permute(0,2,1)


# ---- Helpers ----
def compute_metrics(preds, labels, num_classes):
    oa = (preds==labels).sum()/len(labels)
    ious = []
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
    p,l=np.concatenate(all_p),np.concatenate(all_l)
    oa,miou,ious=compute_metrics(p,l,num_classes)
    return tl/n,oa,miou,ious,p,l


def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR,"*.ply")))
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

    model=DGCNNSemSeg(num_classes).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=5,min_lr=1e-6)

    best_miou,patience_cnt=0.0,0
    hist={"train_loss":[],"val_loss":[],"val_oa":[],"val_miou":[]}
    ckpt=os.path.join(CKPT_DIR,"best_dgcnn.pth")

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
        row = ["dgcnn", f"{toa:.4f}", f"{tmiou:.4f}", f"{tf1:.4f}"]
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
