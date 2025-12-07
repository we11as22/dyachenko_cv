from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


@dataclass
class BBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@dataclass
class Config:
    input_path: str
    output_dir: str
    bbox: BBox
    height_threshold: float
    distance_center: Tuple[float, float, float]
    distance_radius: float
    class_filter: int
    viz: bool
    downsample_for_viz: int
    random_seed: int


def load_cloud(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 3:
        data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype=float)], axis=1)
    elif data.shape[1] >= 4:
        data = data[:, :4]
    else:
        raise ValueError("Need at least 3 columns (x y z)")
    data[:, 3] = data[:, 3].astype(int)
    return data


def save_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%.6f %.6f %.6f %d"
    np.savetxt(path, np.column_stack((points[:, :3], points[:, 3].astype(int))), fmt=fmt)


def clip_bbox(points: np.ndarray, box: BBox) -> np.ndarray:
    mask = (
        (points[:, 0] >= box.xmin)
        & (points[:, 0] <= box.xmax)
        & (points[:, 1] >= box.ymin)
        & (points[:, 1] <= box.ymax)
        & (points[:, 2] >= box.zmin)
        & (points[:, 2] <= box.zmax)
    )
    return points[mask]


def keep_above(points: np.ndarray, z_threshold: float) -> np.ndarray:
    return points[points[:, 2] > z_threshold]


def keep_near(points: np.ndarray, center: Tuple[float, float, float], radius: float) -> np.ndarray:
    center_arr = np.asarray(center, dtype=float)
    dist = np.linalg.norm(points[:, :3] - center_arr[None, :], axis=1)
    return points[dist <= radius]


def keep_class(points: np.ndarray, class_id: int) -> np.ndarray:
    return points[points[:, 3] == class_id]


def shrink(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx]


def plot_cloud(points: np.ndarray, path: Path, title: str, limit: int, seed: int) -> None:
    pts = shrink(points, limit, seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = pts[:, 3]
    if np.unique(colors).size > 1:
        im = ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=1.2, c=colors)
        fig.colorbar(im, ax=ax, label="class")
    else:
        ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=0.8, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=250)
    plt.close(fig)


def run(cfg: Config) -> None:
    log = logging.getLogger("region_filters")
    log.info("Config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    src = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    cloud = load_cloud(src)
    log.info("Loaded %d points", cloud.shape[0])

    bbox_pts = clip_bbox(cloud, cfg.bbox)
    height_pts = keep_above(cloud, cfg.height_threshold)
    radius_pts = keep_near(cloud, tuple(cfg.distance_center), cfg.distance_radius)

    save_cloud(out_dir / "bbox_filtered.asc", bbox_pts)
    save_cloud(out_dir / "high_points.asc", height_pts)
    save_cloud(out_dir / "near_center.asc", radius_pts)

    if cfg.class_filter >= 0:
        cls_pts = keep_class(cloud, cfg.class_filter)
        save_cloud(out_dir / f"class_{cfg.class_filter}.asc", cls_pts)
    else:
        cls_pts = None

    if cfg.viz:
        plot_cloud(cloud, out_dir / "viz_original.png", "Original cloud", cfg.downsample_for_viz, cfg.random_seed)
        plot_cloud(bbox_pts, out_dir / "viz_bbox.png", "BBox filtered", cfg.downsample_for_viz, cfg.random_seed)
        plot_cloud(height_pts, out_dir / "viz_high.png", "Z filter", cfg.downsample_for_viz, cfg.random_seed)
        plot_cloud(radius_pts, out_dir / "viz_near_center.png", "Radius filter", cfg.downsample_for_viz, cfg.random_seed)
        if cls_pts is not None and cls_pts.size > 0:
            plot_cloud(cls_pts, out_dir / "viz_class.png", "Class filter", cfg.downsample_for_viz, cfg.random_seed)

    summary = [
        f"Original: {cloud.shape[0]}",
        f"BBox: {bbox_pts.shape[0]}",
        f"High Z: {height_pts.shape[0]}",
        f"Radius: {radius_pts.shape[0]}",
    ]
    if cls_pts is not None:
        summary.append(f"Class {cfg.class_filter}: {cls_pts.shape[0]}")
    (out_dir / "report.txt").write_text("\n".join(summary))


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(cfg)


if __name__ == "__main__":
    main()
