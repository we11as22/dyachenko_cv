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
CKPT_DIR  = os.path.join(ROOT, "task_3_point_transformer", "checkpoints")
METR_DIR  = os.path.join(ROOT, "task_3_point_transformer", "metrics")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METR_DIR, exist_ok=True)

NUM_POINTS   = 4096
NUM_CLASSES  = 10   # fallback; auto-detected at runtime
BATCH_SIZE   = 8
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 20
WARMUP_EPOCHS = 5
K            = 32
# 4-stage encoder: stem outputs 32, then 32->64->128->256->512
PT_DIMS   = [32, 64, 128, 256, 512]
NPOINTS   = [4096, 2048, 512, 128, 32]
DROPOUT   = 0.5
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
        if s == "end_header": header_end = i + 1; break
    xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    li = prop_names.index("scalar_Label")
    xyz = np.zeros((num_vertices, 3), np.float32); labels = np.zeros(num_vertices, np.int64)
    for j, dl in enumerate(lines[header_end:header_end + num_vertices]):
        v = dl.strip().split()
        xyz[j] = [float(v[xi]), float(v[yi]), float(v[zi])]
        labels[j] = int(float(v[li]))
    return xyz, labels


# ---- Dataset ----
class PointCloudDataset(Dataset):
    def __init__(self, file_list, num_points=NUM_POINTS, augment=False):
        self.file_list = file_list; self.num_points = num_points; self.augment = augment

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        xyz, labels = read_ply(self.file_list[idx])
        n = xyz.shape[0]
        choice = np.random.choice(n, self.num_points, replace=(n < self.num_points))
        xyz, labels = xyz[choice], labels[choice]
        xyz -= xyz.mean(0); d = np.max(np.linalg.norm(xyz, axis=1))
        if d > 1e-8: xyz /= d
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


# ---- Point Transformer ----
def knn_idx(xyz, k):
    """Compute k-NN indices. xyz: (B, N, 3) -> (B, N, k). Clamps k to N-1."""
    N = xyz.shape[1]
    k = min(k, N - 1)
    inner = -2 * torch.bmm(xyz, xyz.transpose(1, 2))
    sq = torch.sum(xyz ** 2, dim=-1, keepdim=True)
    dist = sq + inner + sq.transpose(1, 2)
    return dist.topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]


def gather(pts, idx):
    """Vectorized gather. pts: (B,N,C), idx: (B,M,k) -> (B,M,k,C)"""
    B, N, C = pts.shape
    _, M, k = idx.shape
    idx_exp = idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, C)
    return pts.gather(1, idx_exp).reshape(B, M, k, C)


def fps(xyz, npoint):
    """Farthest Point Sampling. xyz: (B,N,3) -> (B,npoint) indices"""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    bi = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[bi, farthest].unsqueeze(1)
        d = torch.sum((xyz - c) ** 2, dim=-1)
        dist = torch.min(dist, d); farthest = dist.argmax(-1)
    return centroids


class PointTransformerLayer(nn.Module):
    """Point Transformer V1 attention layer with BatchNorm."""

    def __init__(self, dim, k):
        super().__init__(); self.k = k
        self.q = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.pos_enc = nn.Sequential(nn.Linear(3, dim), nn.ReLU(True), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))
        # BatchNorm applied after attn_mlp and pos_enc outputs (over feature dim)
        self.bn_attn = nn.BatchNorm1d(dim)
        self.bn_pos  = nn.BatchNorm1d(dim)

    def forward(self, xyz, feat):
        B, N, C = feat.shape
        idx = knn_idx(xyz, self.k)           # (B, N, k_actual) where k_actual <= self.k
        k_actual = idx.shape[-1]
        kxyz = gather(xyz, idx)              # (B, N, k_actual, 3)
        kfeat = gather(feat, idx)            # (B, N, k_actual, C)

        q = self.q(feat).unsqueeze(2)        # (B, N, 1, C)
        k = self.k_proj(kfeat)               # (B, N, k_actual, C)
        v = self.v(kfeat)                    # (B, N, k_actual, C)

        # pos encoding with BN
        pos_raw = self.pos_enc(xyz.unsqueeze(2) - kxyz)  # (B, N, k_actual, C)
        pos_flat = pos_raw.reshape(B * N * k_actual, C)
        pos = self.bn_pos(pos_flat).reshape(B, N, k_actual, C)

        # attention weights with BN
        attn_raw = self.attn_mlp(q - k + pos)            # (B, N, k_actual, C)
        attn_flat = attn_raw.reshape(B * N * k_actual, C)
        attn = self.bn_attn(attn_flat).reshape(B, N, k_actual, C)
        attn = F.softmax(attn, dim=2)

        return (attn * (v + pos)).sum(2)     # (B, N, C)


class PTStage(nn.Module):
    """Encoder stage: FPS first, then project, then PTLayer on subsampled cloud."""

    def __init__(self, in_dim, out_dim, npoint, k):
        super().__init__(); self.npoint = npoint
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.ptl  = PointTransformerLayer(out_dim, k)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz, feat):
        # 1. FPS downsampling
        idx = fps(xyz, self.npoint)                           # (B, npoint)
        idx3 = idx.unsqueeze(-1)                              # (B, npoint, 1)
        sxyz = xyz.gather(1, idx3.expand(-1, -1, 3))
        # 2. project features at subsampled points
        sf = self.proj(feat.gather(1, idx3.expand(-1, -1, feat.shape[-1])))
        # 3. PTLayer on the subsampled cloud only — O(M²) not O(N²)
        sf = self.norm(sf + self.ptl(sxyz, sf))
        return sxyz, sf


class PTUp(nn.Module):
    """Decoder upsampling block with 3-NN interpolation."""

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + skip_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.ReLU(True),
        )

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        xyz1  — dense coordinates  (B, N_dense,  3)
        xyz2  — sparse coordinates (B, N_sparse, 3)
        feat1 — skip features      (B, N_dense,  skip_dim)
        feat2 — coarse features    (B, N_sparse, in_dim)
        """
        dists = torch.cdist(xyz1, xyz2)                        # (B, N_dense, N_sparse)
        knn_dists, idx = dists.topk(3, dim=-1, largest=False)  # (B, N, 3)
        w = 1.0 / (knn_dists + 1e-6)
        w = w / w.sum(-1, keepdim=True)                        # (B, N, 3)
        interp = (gather(feat2, idx) * w.unsqueeze(-1)).sum(2) # (B, N, in_dim)
        return self.mlp(torch.cat([feat1, interp], -1))


class PointTransformerSemSeg(nn.Module):
    """4-stage Point Transformer for semantic segmentation."""

    def __init__(self, num_classes=NUM_CLASSES, dims=None, npoints=None, k=K, dp=DROPOUT):
        super().__init__()
        if dims    is None: dims    = PT_DIMS
        if npoints is None: npoints = NPOINTS

        # stem: raw XYZ -> dims[0]
        self.stem = nn.Linear(3, dims[0], bias=False)

        # encoder: 4 stages, dims[0]->dims[1]->dims[2]->dims[3]->dims[4]
        self.stages = nn.ModuleList([
            PTStage(dims[i], dims[i + 1], npoints[i + 1], k)
            for i in range(len(dims) - 1)
        ])

        # decoder: PTUp(in_dim=dims[-(i+1)], skip_dim=dims[-(i+2)], out_dim=dims[-(i+2)])
        # for 4 stages: PTUp(512,256,256), PTUp(256,128,128), PTUp(128,64,64), PTUp(64,32,32)
        self.ups = nn.ModuleList([
            PTUp(dims[-(i + 1)], dims[-(i + 2)], dims[-(i + 2)])
            for i in range(len(dims) - 1)
        ])

        # head: dims[0]=32 -> 128 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(dims[0], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dp),
            nn.Linear(128, num_classes),
        )

    def forward(self, xyz):
        feat = self.stem(xyz); skips = [(xyz, feat)]
        for stage in self.stages:
            xyz, feat = stage(xyz, feat); skips.append((xyz, feat))

        # decoder: start from the coarsest level, chain through up blocks
        _, feat = skips[-1]
        for i, up in enumerate(self.ups):
            xyz_dense, feat_dense = skips[-(i + 2)]  # skip connection (denser level)
            xyz_sparse, _         = skips[-(i + 1)]  # coordinates of current coarse level
            feat = up(xyz_dense, xyz_sparse, feat_dense, feat)
            xyz  = xyz_dense

        # head expects (B*N, C) for BatchNorm1d
        B, N, C = feat.shape
        out = self.head(feat.reshape(B * N, C)).reshape(B, N, -1)
        return out


# ---- Helpers ----
def compute_metrics(preds, labels, num_classes):
    oa = (preds == labels).sum() / len(labels); ious = []
    for c in range(num_classes):
        i = ((preds == c) & (labels == c)).sum()
        u = ((preds == c) | (labels == c)).sum()
        ious.append(i / u if u > 0 else float('nan'))
    return oa, np.nanmean(ious), np.array(ious)


@torch.no_grad()
def evaluate(model, loader, cw, num_classes):
    model.eval(); all_p, all_l = [], []; tl, n = 0.0, 0
    for xyz, labels in loader:
        xyz, labels = xyz.to(DEVICE), labels.to(DEVICE)
        logits = model(xyz)
        loss = F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), weight=cw)
        tl += loss.item() * xyz.size(0); n += xyz.size(0)
        all_p.append(logits.argmax(-1).cpu().numpy().flatten())
        all_l.append(labels.cpu().numpy().flatten())
    p, l = np.concatenate(all_p), np.concatenate(all_l)
    oa, miou, ious = compute_metrics(p, l, num_classes)
    return tl / n, oa, miou, ious, p, l


def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ply")))
    print(f"Found {len(all_files)} PLY files"); assert len(all_files) > 0

    # Auto-detect number of classes
    print("Scanning labels...")
    all_labels_set = set()
    for f in tqdm(all_files, desc="Scanning", leave=False):
        _, lbl = read_ply(f)
        all_labels_set.update(lbl.tolist())
    num_classes = max(all_labels_set) + 1
    print(f"Detected {num_classes} classes: {sorted(all_labels_set)}")

    random.shuffle(all_files)
    n_tr  = int(0.70 * len(all_files)); n_val = int(0.15 * len(all_files))
    tr, val, te = all_files[:n_tr], all_files[n_tr:n_tr + n_val], all_files[n_tr + n_val:]
    print(f"Train: {len(tr)}, Val: {len(val)}, Test: {len(te)}")

    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id); random.seed(SEED + worker_id)

    mk = dict(num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(PointCloudDataset(tr, augment=True), BATCH_SIZE,
                              shuffle=True, drop_last=True, generator=g, **mk)
    val_loader   = DataLoader(PointCloudDataset(val), BATCH_SIZE, **mk)
    test_loader  = DataLoader(PointCloudDataset(te),  BATCH_SIZE, **mk)

    # Class-weighted loss
    cc = np.zeros(num_classes, dtype=np.float64)
    for f in tr:
        _, lbl = read_ply(f)
        for c in range(num_classes): cc[c] += (lbl == c).sum()
    freq = cc / cc.sum(); w = 1.0 / (freq + 1e-6); w = w / w.sum() * num_classes
    cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    model = PointTransformerSemSeg(num_classes).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing scheduler (after warmup)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    best_miou, patience_cnt = 0.0, 0
    hist = {"train_loss": [], "val_loss": [], "val_oa": [], "val_miou": []}
    ckpt = os.path.join(CKPT_DIR, "best_point_transformer.pth")

    print("\n=== Training ===")
    for epoch in range(1, EPOCHS + 1):
        # LR schedule: linear warmup for first WARMUP_EPOCHS epochs
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = 1e-5 + (LR - 1e-5) * (epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train(); rl, ns = 0.0, 0
        for xyz, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            xyz, labels = xyz.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xyz).reshape(-1, num_classes),
                                   labels.reshape(-1), weight=cw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); rl += loss.item() * xyz.size(0); ns += xyz.size(0)
        tl = rl / ns

        # Step cosine scheduler only after warmup
        if epoch > WARMUP_EPOCHS:
            cosine_scheduler.step()

        vl, voa, vmiu, _, _, _ = evaluate(model, val_loader, cw, num_classes)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["val_oa"].append(voa); hist["val_miou"].append(vmiu)
        print(f"Epoch {epoch:3d}/{EPOCHS} | LR {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train {tl:.4f} | Val {vl:.4f} | OA {voa:.4f} | mIoU {vmiu:.4f}")
        if vmiu > best_miou:
            best_miou = vmiu; patience_cnt = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Best mIoU: {best_miou:.4f}")
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}"); break

    print("\n=== Test ===")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        print("Warning: no checkpoint found, evaluating with current weights")
    _, toa, tmiou, tious, tp, tl_arr = evaluate(model, test_loader, cw, num_classes)
    tf1 = f1_score(tl_arr, tp, average="macro", zero_division=0)
    cm  = confusion_matrix(tl_arr, tp, labels=list(range(num_classes)))
    print(f"Test OA: {toa:.4f} | mIoU: {tmiou:.4f} | F1: {tf1:.4f}")
    for c in range(num_classes):
        if not np.isnan(tious[c]):
            print(f"  Class {c}: IoU {tious[c]:.4f}")
        else:
            print(f"  Class {c}: IoU N/A")

    # Save CSV metrics
    csv_path = os.path.join(METR_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_oa", "test_miou", "test_f1"]
                        + [f"class_{c}_iou" for c in range(num_classes)])
        row = ["point_transformer", f"{toa:.4f}", f"{tmiou:.4f}", f"{tf1:.4f}"]
        row += [f"{tious[c]:.4f}" if not np.isnan(tious[c]) else "N/A"
                for c in range(num_classes)]
        writer.writerow(row)
    print(f"Results saved to {csv_path}")

    er = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(er, hist["train_loss"], label="Train"); ax.plot(er, hist["val_loss"], label="Val")
    ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR, "loss_curves.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(er, hist["val_miou"], label="mIoU"); ax.plot(er, hist["val_oa"], label="OA")
    ax.set_title("Val mIoU & OA"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR, "miou_acc_curves.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 8))
    cm_n = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(os.path.join(METR_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Metrics saved to {METR_DIR}")


if __name__ == "__main__":
    main()
