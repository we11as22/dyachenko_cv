from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pointnet_data import DataConfig, prepare_dataloaders
from pointnet_model import PointNetClassifier


def show_point_cloud(point_cloud, title="cloud", color="b"):
    """Plot three orthogonal projections."""
    if point_cloud.shape[0] == 3:
        point_cloud = point_cloud.T

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(point_cloud[:, 0], point_cloud[:, 1], c=color, s=1, alpha=0.6)
    axes[0].set_title(f"{title} XY")
    axes[1].scatter(point_cloud[:, 0], point_cloud[:, 2], c=color, s=1, alpha=0.6)
    axes[1].set_title(f"{title} XZ")
    axes[2].scatter(point_cloud[:, 1], point_cloud[:, 2], c=color, s=1, alpha=0.6)
    axes[2].set_title(f"{title} YZ")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def visualize_predictions(
    point_clouds: np.ndarray,
    predictions: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
    num_samples: int = 9,
):
    """Render a small grid with predicted vs. true labels."""
    num_samples = min(num_samples, len(point_clouds))
    rows = (num_samples + 2) // 3
    fig, axes = plt.subplots(
        rows, 3, figsize=(14, 5 * rows), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten()

    for i in range(num_samples):
        pc = point_clouds[i]
        if pc.shape[0] == 3:
            pc = pc.T
        pred = predictions[i]
        true = labels[i]
        color = "green" if pred == true else "red"
        axes[i].scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=1, alpha=0.6)
        axes[i].set_title(f"{class_names[true]} → {class_names[pred]}", fontsize=10)
        axes[i].set_axis_off()

    for i in range(num_samples, len(axes)):
        axes[i].set_axis_off()

    plt.tight_layout()
    return fig


def load_checkpoint(model_path: Path, dropout: float, device: torch.device):
    payload = torch.load(model_path, map_location=device)
    classes = payload["classes"]
    model = PointNetClassifier(num_classes=len(classes), dropout=dropout)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, classes


def main():
    parser = argparse.ArgumentParser(description="Визуализация предсказаний PointNet")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Путь к чекпоинту из train_pointnet.py (по умолчанию artifacts/pointnet.pt)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).with_name("settings.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        root=Path(cfg.data.root),
        num_points=int(cfg.data.num_points),
        train_cap=cfg.data.get("train_cap"),
        test_cap=cfg.data.get("test_cap"),
        download_if_missing=bool(cfg.data.download_if_missing),
    )
    _, test_loader, classes = prepare_dataloaders(
        data_cfg,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
    )

    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt
        else Path(cfg.training.artifacts_dir) / "pointnet.pt"
    )
    model, classes = load_checkpoint(ckpt_path, float(cfg.model.dropout), device)

    collected_points, preds, labels = [], [], []
    with torch.no_grad():
        for points, lbls in test_loader:
            points = points.to(device)
            logits, _ = model(points.transpose(2, 1))
            batch_preds = logits.argmax(1).cpu().numpy()
            collected_points.extend(points.cpu().numpy())
            preds.extend(batch_preds)
            labels.extend(lbls.numpy())
            if len(collected_points) >= cfg.evaluation.get("visualize_samples", 9):
                break

    fig = visualize_predictions(
        np.array(collected_points),
        preds,
        labels,
        classes,
        num_samples=cfg.evaluation.get("visualize_samples", 9),
    )
    out_path = Path(cfg.training.artifacts_dir) / "predictions_visualization.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
