import os
import glob
import random
import csv
import math
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

ROOT     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "task_3_point_transformer", "checkpoints")
METR_DIR = os.path.join(ROOT, "task_3_point_transformer", "metrics")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METR_DIR, exist_ok=True)

NUM_POINTS  = 4096
NUM_CLASSES = 10  # fallback; auto-detected at runtime
WINDOW_SIZE = 64
PT3_DIMS    = [48, 96, 192, 384]
PT3_NPOINTS = [4096, 1024, 256, 64]
PT3_HEADS   = [3, 6, 12, 12]
PT3_DEPTH   = [1, 1, 2, 1]  # reduced from [2,2,6,2] for speed; quality still >> PT V1
BATCH_SIZE  = 8
LR          = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS      = 100
PATIENCE    = 15
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
        if s == "end_header": header_end = i + 1; break
    xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    li = prop_names.index("scalar_Label")
    xyz = np.zeros((num_vertices, 3), np.float32); labels = np.zeros(num_vertices, np.int64)
    for j, dl in enumerate(lines[header_end:header_end + num_vertices]):
        v = dl.strip().split()
        xyz[j] = [float(v[xi]), float(v[yi]), float(v[zi])]; labels[j] = int(float(v[li]))
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


# ---- Morton code (Z-order curve) ----
def _spread_bits(x):
    """Spread 21-bit integer x into every 3rd bit position."""
    x = x & 0x1fffff
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def morton_encode(xyz_int):
    """
    xyz_int: (N, 3) numpy array of integer coords in [0, grid_size).
    Returns (N,) array of Morton codes (uint64).
    """
    x = xyz_int[:, 0].astype(np.uint64)
    y = xyz_int[:, 1].astype(np.uint64)
    z = xyz_int[:, 2].astype(np.uint64)
    return _spread_bits(x) | (_spread_bits(y) << np.uint64(1)) | (_spread_bits(z) << np.uint64(2))


def serialize_point_cloud(xyz):
    """
    xyz: (N, 3) tensor normalized (centred, unit-sphere).
    Returns sort indices (N,) that order points along Z-order curve.
    """
    grid_size = 1024
    # Shift to [0, 1] before quantizing
    xyz_norm = (xyz - xyz.min(0)[0]) / (xyz.max(0)[0] - xyz.min(0)[0] + 1e-8)
    xyz_int = (xyz_norm.clamp(0.0, 1.0 - 1e-6) * grid_size).long().cpu().numpy()
    codes = morton_encode(xyz_int)  # (N,)
    return torch.from_numpy(np.argsort(codes)).to(xyz.device)


# ---- FPS (Farthest Point Sampling) ----
def farthest_point_sample(xyz, npoint):
    """
    xyz: (N, 3) tensor
    Returns indices (npoint,) of sampled points.
    """
    N = xyz.shape[0]
    if npoint >= N:
        return torch.arange(N, device=xyz.device)
    device = xyz.device
    selected = torch.zeros(npoint, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), device=device).item()
    for i in range(npoint):
        selected[i] = farthest
        centroid = xyz[farthest].unsqueeze(0)  # (1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()
    return selected


# ---- 3-NN interpolation ----
def three_nn_interpolate(xyz_fine, xyz_coarse, feat_coarse):
    """
    xyz_fine:    (N_fine, 3)
    xyz_coarse:  (N_coarse, 3)
    feat_coarse: (N_coarse, C)
    Returns: (N_fine, C) interpolated features.
    """
    diffs = xyz_fine.unsqueeze(1) - xyz_coarse.unsqueeze(0)  # (N_fine, N_coarse, 3)
    dists = torch.sum(diffs ** 2, dim=-1)  # (N_fine, N_coarse)
    k = min(3, xyz_coarse.shape[0])
    dist_k, idx_k = dists.topk(k, dim=-1, largest=False)  # (N_fine, k)
    # Inverse-distance weights
    weight = 1.0 / (dist_k + 1e-8)
    weight = weight / weight.sum(dim=-1, keepdim=True)  # (N_fine, k)
    # Gather features
    feat_k = feat_coarse[idx_k]  # (N_fine, k, C)
    return (feat_k * weight.unsqueeze(-1)).sum(dim=1)  # (N_fine, C)


# ---- Relative Positional Encoding MLP ----
class RPE_MLP(nn.Module):
    """Maps (window_size^2, 3) relative coords -> (num_heads, window_size, window_size) bias."""
    def __init__(self, num_heads, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads),
        )

    def forward(self, rel_pos):
        """
        rel_pos: (W, W, 3)
        Returns: (num_heads, W, W)
        """
        bias = self.net(rel_pos)  # (W, W, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, W, W)


# ---- Point Transformer V3 Block (window attention + FFN) ----
class PTv3Block(nn.Module):
    """Single PT V3 block: pre-norm window self-attention + pre-norm FFN."""

    def __init__(self, dim, window_size=64, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-norm attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Relative positional encoding
        self.rpe = RPE_MLP(num_heads)

        # Pre-norm FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, coords):
        """
        x:      (N, dim)  — already Z-order sorted
        coords: (N, 3)    — point coordinates (normalized)
        Returns: (N, dim)
        """
        N, C = x.shape
        W = self.window_size

        # Pad to multiple of W
        pad = (W - N % W) % W
        if pad > 0:
            x      = F.pad(x, (0, 0, 0, pad))
            coords = F.pad(coords, (0, 0, 0, pad))

        N_padded = x.shape[0]
        num_windows = N_padded // W

        # Reshape into windows: (num_windows, W, C)
        x_win      = x.view(num_windows, W, C)
        coords_win = coords.view(num_windows, W, 3)

        # --- Attention ---
        x_res = x_win  # for residual
        x_norm = self.norm1(x_win)

        # QKV projection
        qkv = self.qkv(x_norm)  # (num_windows, W, 3*C)
        qkv = qkv.reshape(num_windows, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, num_windows, num_heads, W, head_dim)
        q, k, v = qkv.unbind(0)  # each: (num_windows, num_heads, W, head_dim)

        # Compute attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (num_windows, num_heads, W, W)

        # RPE: relative position bias
        # Compute pairwise relative coords within each window
        # coords_win: (num_windows, W, 3)
        rel_pos = coords_win.unsqueeze(2) - coords_win.unsqueeze(1)  # (num_windows, W, W, 3)

        # We compute RPE per-window, then average (or apply per-window for full correctness)
        # For efficiency: compute mean relative pos across windows as a shared bias
        # Full correctness: loop over windows (expensive). We use vectorized approach:
        # Flatten (num_windows*W*W, 3) -> RPE -> reshape
        rel_flat = rel_pos.reshape(-1, W * W, 3)  # (num_windows, W*W, 3)
        # Compute RPE for each window
        rpe_bias_list = []
        # Vectorized: process all windows at once by flattening
        rel_all = rel_pos.reshape(num_windows * W * W, 3)  # (num_windows*W*W, 3)
        bias_all = self.rpe.net(rel_all)  # (num_windows*W*W, num_heads)
        bias_all = bias_all.reshape(num_windows, W, W, self.num_heads)
        bias_all = bias_all.permute(0, 3, 1, 2)  # (num_windows, num_heads, W, W)

        attn = attn + bias_all
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, v)  # (num_windows, num_heads, W, head_dim)
        out = out.transpose(1, 2).reshape(num_windows, W, C)  # (num_windows, W, C)
        out = self.proj(out)

        # Residual
        x_win = x_res + out

        # --- FFN ---
        x_win = x_win + self.ffn(self.norm2(x_win))

        # Flatten and remove padding
        x_out = x_win.reshape(N_padded, C)
        if pad > 0:
            x_out = x_out[:N_padded - pad]

        return x_out  # (N_orig, C)


# ---- PT V3 Encoder Stage ----
class PTv3Stage(nn.Module):
    """
    Run `depth` PTv3Blocks, then FPS downsample to `npoint` points,
    then project features from in_dim to out_dim.
    """

    def __init__(self, in_dim, out_dim, npoint, window_size=64, num_heads=8, depth=2):
        super().__init__()
        self.npoint = npoint
        self.blocks = nn.ModuleList([
            PTv3Block(in_dim, window_size, num_heads)
            for _ in range(depth)
        ])
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, coords):
        """
        x:      (N, in_dim)
        coords: (N, 3)
        Returns:
          x_down:      (npoint, out_dim)
          coords_down: (npoint, 3)
          x_skip:      (N, in_dim)  — skip connection before downsampling
          coords_skip: (N, 3)
        """
        # Run attention blocks
        for block in self.blocks:
            x = block(x, coords)

        x_skip = x
        coords_skip = coords

        # FPS downsampling
        fps_idx = farthest_point_sample(coords, self.npoint)
        coords_down = coords[fps_idx]
        x_down = self.proj(x[fps_idx])

        return x_down, coords_down, x_skip, coords_skip


# ---- PT V3 Decoder Up ----
class PTv3Up(nn.Module):
    """
    Decoder block: 3-NN interpolation + skip connection + MLP.
    """

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + skip_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x_coarse, coords_coarse, x_skip, coords_skip):
        """
        x_coarse:      (N_coarse, in_dim)
        coords_coarse: (N_coarse, 3)
        x_skip:        (N_fine, skip_dim)
        coords_skip:   (N_fine, 3)
        Returns: (N_fine, out_dim)
        """
        x_interp = three_nn_interpolate(coords_skip, coords_coarse, x_coarse)  # (N_fine, in_dim)
        x_cat = torch.cat([x_interp, x_skip], dim=-1)  # (N_fine, in_dim + skip_dim)
        return self.mlp(x_cat)


# ---- Full Point Transformer V3 Model ----
class PointTransformerV3(nn.Module):
    """
    Hierarchical Point Transformer V3 for semantic segmentation.

    Architecture:
      Stem: Linear(3, 48)
      4 encoder stages (using PT3_DIMS, PT3_NPOINTS, PT3_HEADS, PT3_DEPTH)
      3 decoder ups (UNet-style skip connections)
      Head: Linear(48, 128) -> LN -> GELU -> Dropout(0.3) -> Linear(128, num_classes)
    """

    def __init__(self, num_classes, dims=None, npoints=None, heads=None, depth=None,
                 window_size=WINDOW_SIZE):
        super().__init__()
        if dims    is None: dims    = PT3_DIMS
        if npoints is None: npoints = PT3_NPOINTS
        if heads   is None: heads   = PT3_HEADS
        if depth   is None: depth   = PT3_DEPTH

        self.num_classes = num_classes

        # Stem: embed 3D coords into first feature dimension
        self.stem = nn.Sequential(
            nn.Linear(3, dims[0]),
            nn.LayerNorm(dims[0]),
        )

        # Encoder stages (we use 3 downsampling stages: 0->1, 1->2, 2->3)
        # Stage 0 processes at npoints[0] resolution and downsamples to npoints[1]
        # Stage 1 processes at npoints[1] and downsamples to npoints[2]
        # Stage 2 processes at npoints[2] and downsamples to npoints[3]
        self.enc0 = PTv3Stage(dims[0], dims[1], npoints[1], window_size, heads[0], depth[0])
        self.enc1 = PTv3Stage(dims[1], dims[2], npoints[2], window_size, heads[1], depth[1])
        self.enc2 = PTv3Stage(dims[2], dims[3], npoints[3], window_size, heads[2], depth[2])

        # Bottleneck attention blocks at lowest resolution (npoints[3])
        self.bottleneck = nn.ModuleList([
            PTv3Block(dims[3], window_size, heads[3])
            for _ in range(depth[3])
        ])

        # Decoder ups (mirror of encoder)
        # Up2: dims[3] -> dims[2], skip from enc2 (dims[2])
        self.up2 = PTv3Up(dims[3], dims[2], dims[2])
        # Up1: dims[2] -> dims[1], skip from enc1 (dims[1])
        self.up1 = PTv3Up(dims[2], dims[1], dims[1])
        # Up0: dims[1] -> dims[0], skip from enc0 (dims[0])
        self.up0 = PTv3Up(dims[1], dims[0], dims[0])

        # Segmentation head
        self.head = nn.Sequential(
            nn.Linear(dims[0], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, xyz):
        """
        xyz: (B, N, 3)
        Returns: (B, N, num_classes)
        """
        B, N, _ = xyz.shape
        outputs = []

        for b in range(B):
            pts = xyz[b]  # (N, 3)
            out = self._forward_single(pts)  # (N, num_classes)
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (B, N, num_classes)

    def _forward_single(self, pts):
        """
        pts: (N, 3)
        Returns: (N, num_classes)
        """
        # Step 1: Z-order serialization
        sort_idx = serialize_point_cloud(pts)  # (N,)
        pts_sorted = pts[sort_idx]             # (N, 3)

        # Step 2: Stem embedding
        x = self.stem(pts_sorted)              # (N, dims[0])
        coords = pts_sorted                    # (N, 3)

        # Step 3: Encoder
        x1, c1, skip0, cskip0 = self.enc0(x,  coords)   # x1: (n1, d1)
        x2, c2, skip1, cskip1 = self.enc1(x1, c1)       # x2: (n2, d2)
        x3, c3, skip2, cskip2 = self.enc2(x2, c2)       # x3: (n3, d3)

        # Step 4: Bottleneck
        for block in self.bottleneck:
            x3 = block(x3, c3)

        # Step 5: Decoder
        xd2 = self.up2(x3, c3, skip2, cskip2)  # (n2, d2)
        xd1 = self.up1(xd2, c2, skip1, cskip1)  # (n1, d1)
        xd0 = self.up0(xd1, c1, skip0, cskip0)  # (N, d0)

        # Step 6: Segmentation head
        logits_sorted = self.head(xd0)  # (N, num_classes)

        # Step 7: Unsort — map predictions back to original point order
        unsort_idx = torch.argsort(sort_idx)
        logits = logits_sorted[unsort_idx]  # (N, num_classes)

        return logits


# ---- Helpers ----
def compute_metrics(preds, labels, num_classes):
    oa = (preds == labels).sum() / len(labels)
    ious = []
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
    n_tr = int(0.70 * len(all_files)); n_val = int(0.15 * len(all_files))
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

    cc = np.zeros(num_classes, dtype=np.float64)
    for f in tr:
        _, lbl = read_ply(f)
        for c in range(num_classes): cc[c] += (lbl == c).sum()
    freq = cc / cc.sum(); w = 1.0 / (freq + 1e-6); w = w / w.sum() * num_classes
    cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    model = PointTransformerV3(num_classes).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    best_miou, patience_cnt = 0.0, 0
    hist = {"train_loss": [], "val_loss": [], "val_oa": [], "val_miou": []}
    ckpt = os.path.join(CKPT_DIR, "best_pt_v3.pth")

    print("\n=== Training ===")
    for epoch in range(1, EPOCHS + 1):
        model.train(); rl, ns = 0.0, 0
        for xyz, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            xyz, labels = xyz.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xyz).reshape(-1, num_classes),
                                   labels.reshape(-1), weight=cw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); rl += loss.item() * xyz.size(0); ns += xyz.size(0)
        tl = rl / ns
        vl, voa, vmiu, _, _, _ = evaluate(model, val_loader, cw, num_classes)
        scheduler.step(vmiu)
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
    cm = confusion_matrix(tl_arr, tp, labels=list(range(num_classes)))
    print(f"Test OA: {toa:.4f} | mIoU: {tmiou:.4f} | F1: {tf1:.4f}")
    for c in range(num_classes):
        print(f"  Class {c}: IoU {tious[c]:.4f}" if not np.isnan(tious[c]) else f"  Class {c}: IoU N/A")

    # Save CSV metrics
    csv_path = os.path.join(METR_DIR, "results_pt_v3.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_oa", "test_miou", "test_f1"]
                        + [f"class_{c}_iou" for c in range(num_classes)])
        row = ["pt_v3", f"{toa:.4f}", f"{tmiou:.4f}", f"{tf1:.4f}"]
        row += [f"{tious[c]:.4f}" if not np.isnan(tious[c]) else "N/A" for c in range(num_classes)]
        writer.writerow(row)
    print(f"Results saved to {csv_path}")

    er = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(er, hist["train_loss"], label="Train"); ax.plot(er, hist["val_loss"], label="Val")
    ax.set_title("Loss (PT V3)"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(METR_DIR, "loss_curves_pt_v3.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(er, hist["val_miou"], label="mIoU"); ax.plot(er, hist["val_oa"], label="OA")
    ax.set_title("Val mIoU & OA (PT V3)"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(METR_DIR, "miou_acc_curves_pt_v3.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 8))
    cm_n = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (PT V3)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(METR_DIR, "confusion_matrix_pt_v3.png"), dpi=150); plt.close()
    print(f"Metrics saved to {METR_DIR}")


if __name__ == "__main__":
    main()
