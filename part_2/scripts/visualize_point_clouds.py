from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LABEL_COLORS = np.array(
    [
        [0.12, 0.47, 0.71],
        [0.70, 0.87, 0.54],
        [0.98, 0.60, 0.60],
        [0.99, 0.75, 0.44],
        [0.79, 0.70, 0.84],
        [0.65, 0.81, 0.89],
        [0.20, 0.63, 0.17],
        [1.00, 0.50, 0.00],
        [0.42, 0.24, 0.60],
        [0.89, 0.10, 0.11],
    ],
    dtype=np.float64,
)


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"{path.name}: not a PLY file")

        vertex_count: int | None = None
        properties: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path.name}: unexpected end of header")
            line = line.strip()
            if line.startswith("format") and "ascii" not in line:
                raise ValueError(f"{path.name}: only ASCII PLY is supported")
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"{path.name}: invalid vertex count")

        required = ["x", "y", "z", "scalar_Label"]
        missing = [name for name in required if name not in properties]
        if missing:
            raise ValueError(f"{path.name}: missing properties {missing}")

        raw = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

    col = {name: idx for idx, name in enumerate(properties)}
    points = raw[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    labels = raw[:, col["scalar_Label"]].astype(np.int32)
    if not np.isfinite(points).all():
        raise ValueError(f"{path.name}: contains NaN or Inf coordinates")
    return points, labels


def cloud_signature(points: np.ndarray) -> tuple[float, float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    center = points.mean(axis=0)
    return float(len(points)), float(extent.max()), float(center[2]), float(extent[2])


def select_diverse_files(files: list[Path], count: int) -> list[Path]:
    """Pick files that differ in size/shape rather than evenly by filename."""
    if count >= len(files):
        return files

    scored: list[tuple[Path, tuple[float, float, float, float]]] = []
    for path in files:
        points, _ = read_ascii_ply(path)
        scored.append((path, cloud_signature(points)))

    # Sort by point count, then pick evenly spaced indices for diversity.
    scored.sort(key=lambda item: item[1][0])
    indices = np.linspace(0, len(scored) - 1, num=count, dtype=int)
    selected = [scored[i][0] for i in indices]

    # Avoid accidental duplicates while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in selected:
        if path.name not in seen:
            unique.append(path)
            seen.add(path.name)
    return unique


def subsample(points: np.ndarray, labels: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, labels
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], labels[idx]


def normalize_for_view(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        return centered
    return centered / scale


def view_angles(path: Path) -> tuple[float, float]:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()
    azim = int(digest[:4], 16) % 360
    elev = 15 + (int(digest[4:6], 16) % 35)
    return elev, azim


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    radius = max(float(np.linalg.norm(points, axis=1).max()), 1e-6)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)


def save_visualization(path: Path, points: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    view_points = normalize_for_view(points)
    elev, azim = view_angles(path)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    colors = LABEL_COLORS[labels % len(LABEL_COLORS)]
    ax1.scatter(view_points[:, 0], view_points[:, 1], view_points[:, 2], c=colors, s=2, linewidths=0)
    ax1.set_title("colored by label")
    set_equal_axes(ax1, view_points)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(view_points[:, 0], view_points[:, 1], view_points[:, 2], c="#666666", s=2, linewidths=0, alpha=0.7)
    ax2.set_title("geometry only")
    set_equal_axes(ax2, view_points)
    ax2.view_init(elev=elev + 12, azim=azim + 35)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    fig.suptitle(
        f"{path.name} | points={len(points)} | labels={len(np.unique(labels))} | view elev={elev}, azim={azim}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize sample point clouds from part_2/data.")
    parser.add_argument("--data-dir", type=Path, default=root / "data", help="Directory with source PLY files.")
    parser.add_argument("--output-dir", type=Path, default=root / "data_vis", help="Directory for PNG previews.")
    parser.add_argument("--max-files", type=int, default=6, help="How many PLY files to visualize.")
    parser.add_argument("--max-points", type=int, default=6000, help="Max points per cloud for plotting.")
    parser.add_argument("--files", nargs="*", default=None, help="Explicit list of PLY filenames to visualize.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    files = sorted(args.data_dir.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"no .ply files found in {args.data_dir}")

    if args.files:
        selected = [args.data_dir / name for name in args.files]
        for path in selected:
            if not path.exists():
                raise FileNotFoundError(path)
    else:
        selected = select_diverse_files(files, args.max_files)

    print(f"Visualizing {len(selected)} files from {args.data_dir}")
    for path in selected:
        points, labels = read_ascii_ply(path)
        points, labels = subsample(points, labels, args.max_points, rng)
        output_path = args.output_dir / f"{path.stem}.png"
        save_visualization(path, points, labels, output_path)
        print(f"Saved {output_path}")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
