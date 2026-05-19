from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


@dataclass
class Config:
    input_path: str
    output_dir: str

    # Subsampling options
    method: str
    n_samples: int
    voxel_size: float

    # Misc
    seed: int
    visualize: bool
    save_vis_to: str
    save_xyz: bool
    fps_init_index: int | None


PointCloud = np.ndarray


def load_cloud(path: Path) -> PointCloud:
    """Read ASCII cloud (x y z [class])."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 columns, got {data.shape[1]}")
    return data.astype(float)


def write_cloud(path: Path, points: PointCloud) -> None:
    fmt = "%.6f %.6f %.6f" if points.shape[1] == 3 else "%.6f %.6f %.6f %.6f"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt=fmt)


def subsample_random(points: PointCloud, cfg: Config) -> Tuple[PointCloud, float]:
    start = time.time()
    rng = np.random.default_rng(cfg.seed)
    n = points.shape[0]
    if cfg.n_samples >= n:
        return points.copy(), time.time() - start
    idx = rng.choice(n, size=cfg.n_samples, replace=False)
    return points[idx], time.time() - start


def subsample_voxel(points: PointCloud, cfg: Config) -> Tuple[PointCloud, float]:
    start = time.time()
    if cfg.voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    grid = np.floor(points[:, :3] / cfg.voxel_size).astype(np.int64)
    _, first_idx = np.unique(grid, axis=0, return_index=True)
    sampled = points[np.sort(first_idx)]
    return sampled, time.time() - start


def subsample_fps(points: PointCloud, cfg: Config) -> Tuple[PointCloud, float]:
    start = time.time()
    k = min(cfg.n_samples, points.shape[0])
    xyz = points[:, :3]

    chosen = np.empty(k, dtype=np.int64)
    init = 0 if cfg.fps_init_index is None else int(cfg.fps_init_index)
    chosen[0] = init

    dist2 = np.full(points.shape[0], np.inf)
    last = xyz[init]
    dist2 = np.minimum(dist2, np.sum((xyz - last) ** 2, axis=1))

    for i in range(1, k):
        nxt = int(np.argmax(dist2))
        chosen[i] = nxt
        last = xyz[nxt]
        dist2 = np.minimum(dist2, np.sum((xyz - last) ** 2, axis=1))

    return points[chosen], time.time() - start


SAMPLERS: Dict[str, Callable[[PointCloud, Config], Tuple[PointCloud, float]]] = {
    "random": subsample_random,
    "voxel": subsample_voxel,
    "fps": subsample_fps,
}


def plot_cloud(points: PointCloud, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    coords = points[:, :3]
    if points.shape[1] == 4:
        colors = points[:, 3]
        scatter = ax.scatter(coords[:, 1], coords[:, 0], coords[:, 2], s=1.5, c=colors)
        fig.colorbar(scatter, ax=ax, label="class")
    else:
        ax.scatter(coords[:, 1], coords[:, 0], coords[:, 2], s=1.0, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def run(cfg: Config) -> None:
    log = logging.getLogger("cloud_sampling")
    log.info("Config:\n%s", cfg)

    src = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud = load_cloud(src)
    log.info("Loaded %d points from %s", cloud.shape[0], src)

    sampler = SAMPLERS.get(cfg.method.lower())
    if sampler is None:
        raise ValueError(f"Unknown method '{cfg.method}', expected one of {list(SAMPLERS)}")

    sampled, duration = sampler(cloud, cfg)
    log.info("Method %s -> %d points (%.3f s)", cfg.method, sampled.shape[0], duration)

    if cfg.save_xyz:
        out_path = out_dir / f"{cfg.method}_subsampled_{sampled.shape[0]}.xyz"
        write_cloud(out_path, sampled)
        log.info("Saved cloud: %s", out_path)

    if cfg.visualize:
        s = max(0.1, 20_000.0 / max(sampled.shape[0], 1))
        plot_cloud(
            sampled,
            Path(cfg.save_vis_to),
            title=f"{cfg.method.upper()} | {sampled.shape[0]} pts (s={s:.2f})",
        )
        log.info("Saved preview: %s", cfg.save_vis_to)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(Config(**cfg))


if __name__ == "__main__":
    main()
