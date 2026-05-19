from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import hydra.utils as hy_utils
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from omegaconf import DictConfig, OmegaConf


@dataclass
class Config:
    N: int = 1000
    seed: int = 42
    save_dir: str = "./outputs"
    plot_matplotlib: bool = True
    plot_plotly: bool = True
    matplotlib_marker_size: int = 10
    plotly_marker_size: int = 3
    save_csv: bool = True
    save_ply: bool = False
    use_alt_colormap: bool = True
    alt_colormap: str = "viridis"


def make_points(cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.seed))
    return rng.random((int(cfg.N), 3))


def paint_by_coords(points: np.ndarray) -> np.ndarray:
    return points.copy()


def paint_by_distance(points: np.ndarray, cmap_name: str) -> np.ndarray:
    center = np.array([0.5, 0.5, 0.5], dtype=float)
    dist = np.linalg.norm(points - center, axis=1)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    cmap = cm.get_cmap(cmap_name)
    return cmap(dist_norm)[:, :3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = "x,y,z,r,g,b"
    data = np.hstack((points, colors))
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors255 = np.clip((colors * 255).astype(int), 0, 255)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")
        for (x, y, z), (r, g, b) in zip(points, colors255):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def plot_flat(points: np.ndarray, colors: np.ndarray, cfg: Config, out_dir: Path, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=cfg.matplotlib_marker_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"2D view ({suffix})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"points_2d_{suffix}.png", dpi=200)
    fig.savefig(out_dir / f"points_2d_{suffix}.svg")
    plt.close(fig)


def plot_3d(points: np.ndarray, colors: np.ndarray, cfg: Config, out_dir: Path, suffix: str) -> None:
    rgb_strings = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in colors]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=cfg.plotly_marker_size, color=rgb_strings),
            )
        ]
    )
    fig.update_layout(title=f"3D view ({suffix})", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    fig.write_html(out_dir / f"points_3d_{suffix}.html", include_plotlyjs="cdn")


def run(cfg: Config) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))

    points = make_points(cfg)
    base_colors = paint_by_coords(points)
    alt_colors = paint_by_distance(points, cfg.alt_colormap) if cfg.use_alt_colormap else None

    out_dir = ensure_dir(Path(hy_utils.get_original_cwd()) / cfg.save_dir)
    print(f"Outputs: {out_dir}")

    if cfg.save_csv:
        save_csv(out_dir / "points_rgb.csv", points, base_colors)
    if cfg.save_ply:
        save_ply(out_dir / "points_rgb.ply", points, base_colors)

    if cfg.plot_matplotlib:
        plot_flat(points, base_colors, cfg, out_dir, "matplotlib")
        if alt_colors is not None:
            plot_flat(points, alt_colors, cfg, out_dir, f"alt_{cfg.alt_colormap}")

    if cfg.plot_plotly:
        plot_3d(points, base_colors, cfg, out_dir, "plotly")
        if alt_colors is not None:
            plot_3d(points, alt_colors, cfg, out_dir, f"plotly_{cfg.alt_colormap}")


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg: Optional[DictConfig]) -> None:
    run(Config(**cfg))


if __name__ == "__main__":
    main()
