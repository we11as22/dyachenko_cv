from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import main
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


@dataclass
class Config:
    input_path: str = "data/cloud.txt"
    output_dir: str = "outputs"
    constant_value: float = 10.0
    multiply_factor: float = 2.0
    add_value: float = 5.0
    gaussian_sigma: float = 2.0
    moving_average_window: int = 5
    colormap: str = "viridis"
    filter_min: float = -1e9
    filter_max: float = 1e9
    use_scalar_axis: int = 2


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_asc(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 columns, got {data.shape[1]}")
    return data.astype(float)


def make_constant(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value), dtype=float)


def gauss_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(values, sigma=float(sigma), mode="nearest")


def moving_avg(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="same")


def gradient(values: np.ndarray) -> np.ndarray:
    return np.gradient(values)


def normalize(values: np.ndarray) -> np.ndarray:
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if np.isclose(vmin, vmax):
        return np.zeros_like(values)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def rgb_from_scalar(values: np.ndarray, cmap_name: str) -> np.ndarray:
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    return cmap(normalize(values))[:, :3]


def fill_nans(values: np.ndarray) -> np.ndarray:
    idx = np.arange(len(values))
    mask = np.isnan(values)
    if not mask.any():
        return values.copy()
    if mask.all():
        return np.zeros_like(values)
    interp = interp1d(idx[~mask], values[~mask], bounds_error=False, fill_value="extrapolate")
    return interp(idx)


def filter_by_range(points: np.ndarray, scalar: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (scalar >= lo) & (scalar <= hi)
    return points[mask], scalar[mask]


def replace_coord(points: np.ndarray, scalar: np.ndarray, axis: int) -> np.ndarray:
    out = points.copy()
    out[:, axis] = scalar
    return out


def drop_column(points: np.ndarray, idx: int) -> np.ndarray:
    if points.shape[1] <= idx:
        return points
    return np.delete(points, idx, axis=1)


def stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(np.sum(~np.isnan(values))),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
    }


def plot_series(values: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(values, linewidth=1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlabel("point index")
    ax.set_ylabel("scalar")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cloud(points: np.ndarray, scalar: np.ndarray, cmap: str, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    normalized = normalize(scalar)
    scatter = ax.scatter(points[:, 1], points[:, 0], points[:, 2], c=normalized, cmap=cmap, s=1)
    fig.colorbar(scatter, ax=ax, label="normalized scalar")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class ScalarFieldWorkflow:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(get_original_cwd())
        self.out = ensure_dir(self.root / cfg.output_dir)
        self.xyz, self.scalar = self._load_inputs()
        self.n_points = self.xyz.shape[0]

    def _load_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        data = load_asc(self.root / self.cfg.input_path)
        coords = data[:, :3]
        if data.shape[1] >= 4:
            scalar = data[:, 3].astype(float)
        else:
            scalar = make_constant(data.shape[0], self.cfg.constant_value)
        np.save(self.out / "xyz.npy", coords)
        np.save(self.out / "scalar_initial.npy", scalar)
        return coords, scalar

    def save_array(self, name: str, arr: np.ndarray) -> None:
        np.save(self.out / f"{name}.npy", arr)

    def save_line(self, name: str, arr: np.ndarray, title: str) -> None:
        self.save_array(name, arr)
        plot_series(arr, self.out / f"{name}.png", title)

    def save_cloud_plot(self, name: str, pts: np.ndarray, scalar: np.ndarray, title: str) -> None:
        self.save_array(name, scalar)
        plot_cloud(pts, scalar, self.cfg.colormap, self.out / f"{name}.png", title)

    def run(self) -> None:
        # Step 1: constant field
        const = make_constant(self.n_points, self.cfg.constant_value)
        self.save_line("scalar_task1_constant", const, "Constant scalar")

        # Step 2: multiply original/loaded scalar
        multiplied = self.scalar * self.cfg.multiply_factor
        self.save_line("scalar_task2_multiplied", multiplied, "Scalar * factor")

        # Step 3: add number
        added = multiplied + self.cfg.add_value
        self.save_line("scalar_task3_added", added, "Scalar + bias")

        # Step 4: gaussian smoothing
        smoothed = gauss_1d(added, self.cfg.gaussian_sigma)
        self.save_line("scalar_task4_gaussian", smoothed, f"Gaussian sigma={self.cfg.gaussian_sigma}")

        # Step 5: gradient
        grad = gradient(smoothed)
        self.save_array("scalar_task5_gradient", grad)

        # Step 6: moving average
        averaged = moving_avg(smoothed, self.cfg.moving_average_window)
        self.save_line("scalar_task6_moving_avg", averaged, f"Moving avg window={self.cfg.moving_average_window}")

        # Step 7: scalar -> RGB
        colors = rgb_from_scalar(averaged, self.cfg.colormap)
        self.save_array("colors_task7_rgb", colors)
        plot_cloud(self.xyz, averaged, self.cfg.colormap, self.out / "task7_colored_scatter.png", "Colored by scalar")

        # Step 8: stats
        df_stats = pd.DataFrame([stats(averaged)])
        df_stats.to_csv(self.out / "task8_stats.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.axis("off")
        tbl = ax.table(cellText=df_stats.round(5).values, colLabels=df_stats.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        fig.tight_layout()
        fig.savefig(self.out / "task8_stats_table.png", dpi=150)
        plt.close(fig)

        # Step 9: normalize
        normalized = normalize(averaged)
        self.save_line("scalar_task9_normalized", normalized, "Normalized to [0,1]")

        # Step 10: handle NaNs
        nany = normalized.copy()
        if not np.any(np.isnan(nany)):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(nany), size=max(1, len(nany) // 50), replace=False)
            nany[idx] = np.nan
        filled = fill_nans(nany)
        self.save_line("scalar_task10_with_nans", nany, "Injected NaNs")
        self.save_line("scalar_task10_interpolated", filled, "Interpolated NaNs")

        # Step 11: filter by scalar range
        f_pts, f_scalar = filter_by_range(self.xyz, filled, self.cfg.filter_min, self.cfg.filter_max)
        np.save(self.out / "task11_filtered_xyz.npy", f_pts)
        np.save(self.out / "task11_filtered_scalar.npy", f_scalar)
        if f_pts.size > 0:
            plot_cloud(f_pts, f_scalar, self.cfg.colormap, self.out / "task11_filtered_scatter.png", "Filtered by range")

        # Step 12: use scalar as coordinate
        swapped = replace_coord(self.xyz, filled, self.cfg.use_scalar_axis)
        self.save_cloud_plot("task12_used_scalar_as_coord", swapped, filled, "Scalar used as coordinate")

        # Step 13: remove scalar column if present
        raw_with_scalar = np.column_stack((self.xyz, self.scalar))
        trimmed = drop_column(raw_with_scalar, 3)
        np.save(self.out / "task13_removed_scalar_points.npy", trimmed)

        summary = {
            "n_points": int(self.n_points),
            "outputs_dir": str(self.out),
            "saved_files": sorted([p.name for p in self.out.iterdir()]),
        }
        pd.DataFrame([summary]).to_json(self.out / "run_summary.json", orient="records", indent=2)
        print("Pipeline finished. Outputs in", self.out)


@main(version_base=None, config_name="config", config_path=".")
def main(cfg: Config) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))
    ScalarFieldWorkflow(cfg).run()


if __name__ == "__main__":
    main()
