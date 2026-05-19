from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


GEOMETRY_PLANE = "plane"
GEOMETRY_TUBE = "tube"
GEOMETRY_SPHERE = "sphere"
GEOMETRY_COMPLEX = "complex"

METHOD_POISSON = "poisson"
METHOD_ALPHA = "alpha_shape"
METHOD_BALL_PIVOTING = "ball_pivoting"


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    metrics_dir: Path
    visualization_dir: Path
    min_segment_points: int = 80
    voxel_size: float = 0.0
    max_points_per_segment: int = 3500
    normal_knn: int = 24
    poisson_depth: int = 7
    alpha_factor: float = 2.2
    ball_radius_factor: float = 1.8
    quality_sample_points: int = 5000
    max_files: int | None = None
    seed: int = 42
    workers: int = 1
    force_method: str | None = None
    isolate_files: bool = True
    save_segment_meshes: bool = True
    save_visualizations: bool = True
    log_files: bool = True


@dataclass
class PointCloudData:
    path: Path
    points: np.ndarray
    labels: np.ndarray


@dataclass
class Segment:
    source_file: str
    label: int
    points: np.ndarray
    normalized_points: np.ndarray
    center: np.ndarray
    scale: float


@dataclass
class SegmentFeatures:
    point_count: int
    bbox_volume: float
    density: float
    mean_nn_distance: float
    linearity: float
    planarity: float
    sphericity: float
    curvature: float
    normal_consistency: float
    connected_components: int
    largest_component_ratio: float


@dataclass
class SegmentResult:
    source_file: str
    label: int
    point_count: int
    geometry_type: str
    method: str
    mesh_path: str
    status: str
    error: str
    features: SegmentFeatures
    quality: dict[str, float | int | str]


def read_ascii_ply(path: Path) -> PointCloudData:
    """Read ASCII PLY files with x, y, z and scalar_Label fields."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError("not a PLY file")

        vertex_count: int | None = None
        properties: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected end of PLY header")
            line = line.strip()
            if line.startswith("format") and "ascii" not in line:
                raise ValueError("only ASCII PLY is supported")
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif line == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError("invalid or missing vertex count")

        required = ["x", "y", "z", "scalar_Label"]
        missing = [name for name in required if name not in properties]
        if missing:
            raise ValueError(f"missing PLY properties: {missing}")

        raw = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[0] != vertex_count:
            raise ValueError(f"expected {vertex_count} vertices, got {raw.shape[0]}")

    col = {name: idx for idx, name in enumerate(properties)}
    points = raw[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    labels = raw[:, col["scalar_Label"]].astype(np.int32)
    validate_points(points, labels)
    return PointCloudData(path=path, points=points, labels=labels)


def validate_points(points: np.ndarray, labels: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("labels must have shape [N]")
    if len(points) == 0:
        raise ValueError("point cloud is empty")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains NaN or Inf values")


def remove_statistical_noise(points: np.ndarray, labels: np.ndarray, nb_neighbors: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < nb_neighbors + 5:
        return points, labels
    pcd = to_point_cloud(points)
    _, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return points, labels
    return points[indices], labels[indices]


def voxel_downsample_by_label(
    points: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, labels

    kept_points: list[np.ndarray] = []
    kept_labels: list[np.ndarray] = []
    for label in np.unique(labels):
        mask = labels == label
        label_points = points[mask]
        if len(label_points) == 0:
            continue
        voxel_ids = np.floor(label_points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_ids, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        kept_points.append(label_points[unique_indices])
        kept_labels.append(np.full(len(unique_indices), int(label), dtype=np.int32))

    if not kept_points:
        return points, labels

    merged_points = np.vstack(kept_points)
    merged_labels = np.concatenate(kept_labels)
    order = rng.permutation(len(merged_points))
    return merged_points[order], merged_labels[order]


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        scale = 1.0
    return centered / scale, center, scale


def build_segments(cloud: PointCloudData, cfg: PipelineConfig, rng: np.random.Generator) -> list[Segment]:
    points, labels = remove_statistical_noise(cloud.points, cloud.labels)
    points, labels = voxel_downsample_by_label(points, labels, cfg.voxel_size, rng)

    segments: list[Segment] = []
    for label in sorted(np.unique(labels).tolist()):
        label_points = points[labels == label]
        if len(label_points) < cfg.min_segment_points:
            continue
        if len(label_points) > cfg.max_points_per_segment:
            idx = rng.choice(len(label_points), size=cfg.max_points_per_segment, replace=False)
            label_points = label_points[idx]
        normalized, center, scale = normalize_points(label_points)
        segments.append(
            Segment(
                source_file=cloud.path.stem,
                label=int(label),
                points=label_points,
                normalized_points=normalized,
                center=center,
                scale=scale,
            )
        )
    return segments


def to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, knn: int) -> o3d.geometry.PointCloud:
    if len(pcd.points) < 3:
        return pcd
    knn = max(3, min(knn, len(pcd.points) - 1))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=max(3, min(knn, len(pcd.points) - 1)))
    except RuntimeError:
        # Degenerate or very sparse segments can fail orientation; estimated normals are still useful.
        pass
    return pcd


def nearest_neighbor_stats(points: np.ndarray) -> tuple[float, np.ndarray]:
    if len(points) < 3:
        return 0.0, np.zeros(len(points), dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=min(2, len(points))).fit(points)
    distances, _ = nn.kneighbors(points)
    values = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    return float(np.mean(values)), values


def pca_features(points: np.ndarray) -> tuple[float, float, float, float]:
    cov = np.cov(points.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    if eigvals[0] <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    l1, l2, l3 = eigvals
    linearity = float((l1 - l2) / l1)
    planarity = float((l2 - l3) / l1)
    sphericity = float(l3 / l1)
    curvature = float(l3 / max(l1 + l2 + l3, 1e-12))
    return linearity, planarity, sphericity, curvature


def normal_consistency(points: np.ndarray, normals: np.ndarray, knn: int) -> float:
    if len(points) < 4 or len(normals) != len(points):
        return 0.0
    k = max(2, min(knn, len(points)))
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nn.kneighbors(points)
    dots: list[float] = []
    for row, normal in zip(indices, normals):
        neighbor_normals = normals[row[1:]]
        dot = np.abs(neighbor_normals @ normal)
        dots.extend(dot.tolist())
    return float(np.mean(dots)) if dots else 0.0


def connectivity(points: np.ndarray, radius: float) -> tuple[int, float]:
    if len(points) == 0:
        return 0, 0.0
    if len(points) == 1:
        return 1, 1.0
    radius = max(radius, 1e-6)
    nn = NearestNeighbors(radius=radius).fit(points)
    neighbors = nn.radius_neighbors(points, return_distance=False)
    visited = np.zeros(len(points), dtype=bool)
    component_sizes: list[int] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in neighbors[current]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        component_sizes.append(size)
    largest = max(component_sizes) if component_sizes else 0
    return len(component_sizes), float(largest / len(points))


def analyze_segment(segment: Segment, cfg: PipelineConfig) -> tuple[SegmentFeatures, o3d.geometry.PointCloud]:
    points = segment.normalized_points
    pcd = estimate_normals(to_point_cloud(points), cfg.normal_knn)
    normals = np.asarray(pcd.normals)

    bbox_extent = points.max(axis=0) - points.min(axis=0)
    bbox_volume = float(np.prod(np.maximum(bbox_extent, 1e-6)))
    density = float(len(points) / bbox_volume)
    mean_nn, nn_distances = nearest_neighbor_stats(points)
    linearity, planarity, sphericity, curvature = pca_features(points)
    consistency = normal_consistency(points, normals, cfg.normal_knn)
    radius = float(max(mean_nn * 2.8, np.percentile(nn_distances, 75) * 2.0, 1e-4))
    components, largest_ratio = connectivity(points, radius)

    return (
        SegmentFeatures(
            point_count=len(points),
            bbox_volume=bbox_volume,
            density=density,
            mean_nn_distance=mean_nn,
            linearity=linearity,
            planarity=planarity,
            sphericity=sphericity,
            curvature=curvature,
            normal_consistency=consistency,
            connected_components=components,
            largest_component_ratio=largest_ratio,
        ),
        pcd,
    )


def classify_segment(features: SegmentFeatures) -> str:
    if features.planarity > 0.55 and features.sphericity < 0.08 and features.normal_consistency > 0.72:
        return GEOMETRY_PLANE
    if features.linearity > 0.55 and features.planarity > 0.12:
        return GEOMETRY_TUBE
    if features.sphericity > 0.18 and features.linearity < 0.55 and features.planarity < 0.55:
        return GEOMETRY_SPHERE
    return GEOMETRY_COMPLEX


def choose_method(geometry_type: str, features: SegmentFeatures) -> str:
    if geometry_type == GEOMETRY_PLANE:
        return METHOD_ALPHA
    if geometry_type == GEOMETRY_TUBE:
        return METHOD_BALL_PIVOTING
    if geometry_type == GEOMETRY_SPHERE:
        return METHOD_POISSON
    if features.normal_consistency > 0.65 and features.largest_component_ratio > 0.75:
        return METHOD_POISSON
    if features.connected_components > 2:
        return METHOD_ALPHA
    return METHOD_BALL_PIVOTING


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
    return mesh


def reconstruct_poisson(pcd: o3d.geometry.PointCloud, cfg: PipelineConfig) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=cfg.poisson_depth)
    densities_np = np.asarray(densities)
    if len(densities_np) > 0:
        keep = densities_np > np.quantile(densities_np, 0.03)
        mesh.remove_vertices_by_mask(~keep)
    return clean_mesh(mesh)


def reconstruct_alpha_shape(
    pcd: o3d.geometry.PointCloud,
    features: SegmentFeatures,
    cfg: PipelineConfig,
) -> o3d.geometry.TriangleMesh:
    alpha = max(features.mean_nn_distance * cfg.alpha_factor, 0.02)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
    return clean_mesh(mesh)


def reconstruct_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    features: SegmentFeatures,
    cfg: PipelineConfig,
) -> o3d.geometry.TriangleMesh:
    radius = max(features.mean_nn_distance * cfg.ball_radius_factor, 0.01)
    radii = o3d.utility.DoubleVector([radius, radius * 1.5, radius * 2.0])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    return clean_mesh(mesh)


def reconstruct_segment(
    pcd: o3d.geometry.PointCloud,
    features: SegmentFeatures,
    method: str,
    cfg: PipelineConfig,
) -> o3d.geometry.TriangleMesh:
    if method == METHOD_POISSON:
        return reconstruct_poisson(pcd, cfg)
    if method == METHOD_ALPHA:
        return reconstruct_alpha_shape(pcd, features, cfg)
    if method == METHOD_BALL_PIVOTING:
        return reconstruct_ball_pivoting(pcd, features, cfg)
    raise ValueError(f"unknown reconstruction method: {method}")


def denormalize_mesh(mesh: o3d.geometry.TriangleMesh, center: np.ndarray, scale: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > 0:
        mesh.vertices = o3d.utility.Vector3dVector(vertices * scale + center)
    return mesh


def evaluate_mesh(mesh: o3d.geometry.TriangleMesh, segment: Segment, cfg: PipelineConfig) -> dict[str, float | int | str]:
    vertex_count = len(mesh.vertices)
    triangle_count = len(mesh.triangles)
    if vertex_count == 0 or triangle_count == 0:
        return {
            "mesh_vertices": vertex_count,
            "mesh_triangles": triangle_count,
            "mean_cloud_to_mesh": math.inf,
            "p95_cloud_to_mesh": math.inf,
            "max_cloud_to_mesh": math.inf,
            "accuracy_ratio": 0.0,
            "mesh_components": 0,
        }

    sample_count = min(max(cfg.quality_sample_points, 500), max(triangle_count * 3, 500))
    sampled = mesh.sample_points_uniformly(number_of_points=sample_count)
    sampled_points = np.asarray(sampled.points)
    nn = NearestNeighbors(n_neighbors=1).fit(sampled_points)
    distances, _ = nn.kneighbors(segment.points)
    distances = distances[:, 0]
    threshold = max(float(np.percentile(distances, 75)) * 1.5, segment.scale * 0.01, 1e-6)
    clusters, _, _ = mesh.cluster_connected_triangles()
    cluster_count = len(set(np.asarray(clusters).tolist())) if len(clusters) else 0
    return {
        "mesh_vertices": vertex_count,
        "mesh_triangles": triangle_count,
        "mean_cloud_to_mesh": float(np.mean(distances)),
        "p95_cloud_to_mesh": float(np.percentile(distances, 95)),
        "max_cloud_to_mesh": float(np.max(distances)),
        "accuracy_ratio": float(np.mean(distances <= threshold)),
        "mesh_components": int(cluster_count),
    }


def paint_mesh(mesh: o3d.geometry.TriangleMesh, label: int) -> o3d.geometry.TriangleMesh:
    palette = np.array(
        [
            [0.65, 0.81, 0.89],
            [0.12, 0.47, 0.71],
            [0.70, 0.87, 0.54],
            [0.20, 0.63, 0.17],
            [0.98, 0.60, 0.60],
            [0.89, 0.10, 0.11],
            [0.99, 0.75, 0.44],
            [1.00, 0.50, 0.00],
            [0.79, 0.70, 0.84],
            [0.42, 0.24, 0.60],
        ],
        dtype=np.float64,
    )
    mesh.paint_uniform_color(palette[label % len(palette)])
    return mesh


def save_segment_visualization(segment: Segment, mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(segment.points[:, 0], segment.points[:, 1], segment.points[:, 2], s=1)
    ax1.set_title(f"{segment.source_file} label={segment.label}")
    set_equal_axes(ax1, segment.points)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) > 0 and len(triangles) > 0:
        ax2.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=triangles,
            linewidth=0.05,
            antialiased=True,
            alpha=0.85,
        )
    ax2.set_title("Reconstructed mesh")
    set_equal_axes(ax2, vertices if len(vertices) else segment.points)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def merge_meshes(meshes: Iterable[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged += mesh
    return clean_mesh(merged)


def write_csv(path: Path, results: list[SegmentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "label",
        "point_count",
        "geometry_type",
        "method",
        "status",
        "error",
        "mesh_path",
        "bbox_volume",
        "density",
        "mean_nn_distance",
        "linearity",
        "planarity",
        "sphericity",
        "curvature",
        "normal_consistency",
        "connected_components",
        "largest_component_ratio",
        "mesh_vertices",
        "mesh_triangles",
        "mean_cloud_to_mesh",
        "p95_cloud_to_mesh",
        "max_cloud_to_mesh",
        "accuracy_ratio",
        "mesh_components",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "source_file": result.source_file,
                "label": result.label,
                "point_count": result.point_count,
                "geometry_type": result.geometry_type,
                "method": result.method,
                "status": result.status,
                "error": result.error,
                "mesh_path": result.mesh_path,
                **asdict(result.features),
                **result.quality,
            }
            writer.writerow(row)


def write_summary(path: Path, cfg: PipelineConfig, results: list[SegmentResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    by_type: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for result in results:
        by_type[result.geometry_type] = by_type.get(result.geometry_type, 0) + 1
        by_method[result.method] = by_method.get(result.method, 0) + 1
    mean_distance = float(np.mean([r.quality.get("mean_cloud_to_mesh", math.inf) for r in ok])) if ok else math.inf
    mean_accuracy = float(np.mean([r.quality.get("accuracy_ratio", 0.0) for r in ok])) if ok else 0.0
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "segments_total": len(results),
        "segments_ok": len(ok),
        "segments_failed": len(failed),
        "geometry_types": by_type,
        "methods": by_method,
        "mean_cloud_to_mesh": mean_distance,
        "mean_accuracy_ratio": mean_accuracy,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def persist_metrics(cfg: PipelineConfig, results: list[SegmentResult]) -> None:
    ordered = sorted(results, key=lambda item: (item.source_file, item.label))
    write_csv(cfg.metrics_dir / "segments_metrics.csv", ordered)
    write_summary(cfg.metrics_dir / "summary.json", cfg, ordered)


def process_cloud(path: Path, cfg: PipelineConfig, rng: np.random.Generator) -> list[SegmentResult]:
    cloud = read_ascii_ply(path)
    segments = build_segments(cloud, cfg, rng)
    results: list[SegmentResult] = []
    reconstructed_meshes: list[o3d.geometry.TriangleMesh] = []

    cloud_output_dir = cfg.output_dir / cloud.path.stem
    cloud_vis_dir = cfg.visualization_dir / cloud.path.stem
    cloud_output_dir.mkdir(parents=True, exist_ok=True)
    cloud_vis_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        features, pcd = analyze_segment(segment, cfg)
        geometry_type = classify_segment(features)
        method = cfg.force_method or choose_method(geometry_type, features)
        mesh_path = cloud_output_dir / f"label_{segment.label:02d}_{method}.ply"
        status = "ok"
        error = ""
        quality: dict[str, float | int | str]

        try:
            mesh_norm = reconstruct_segment(pcd, features, method, cfg)
            mesh = denormalize_mesh(mesh_norm, segment.center, segment.scale)
            mesh = paint_mesh(mesh, segment.label)
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise RuntimeError("reconstruction produced an empty mesh")
            quality = evaluate_mesh(mesh, segment, cfg)
            if cfg.save_segment_meshes:
                o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            if cfg.save_visualizations:
                save_segment_visualization(segment, mesh, cloud_vis_dir / f"label_{segment.label:02d}.png")
            reconstructed_meshes.append(mesh)
        except Exception as exc:  # noqa: BLE001 - every segment must be isolated from failures.
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            quality = {
                "mesh_vertices": 0,
                "mesh_triangles": 0,
                "mean_cloud_to_mesh": math.inf,
                "p95_cloud_to_mesh": math.inf,
                "max_cloud_to_mesh": math.inf,
                "accuracy_ratio": 0.0,
                "mesh_components": 0,
            }
            mesh_path = Path("")

        results.append(
            SegmentResult(
                source_file=cloud.path.name,
                label=segment.label,
                point_count=len(segment.points),
                geometry_type=geometry_type,
                method=method,
                mesh_path=str(mesh_path),
                status=status,
                error=error,
                features=features,
                quality=quality,
            )
        )

    if reconstructed_meshes:
        merged = merge_meshes(reconstructed_meshes)
        o3d.io.write_triangle_mesh(str(cloud_output_dir / f"{cloud.path.stem}_merged.ply"), merged)

    return results


def process_cloud_worker(path: Path, cfg: PipelineConfig, seed: int) -> list[SegmentResult]:
    rng = np.random.default_rng(seed)
    return process_cloud(path, cfg, rng)


def result_to_dict(result: SegmentResult) -> dict[str, object]:
    return {
        "source_file": result.source_file,
        "label": result.label,
        "point_count": result.point_count,
        "geometry_type": result.geometry_type,
        "method": result.method,
        "mesh_path": result.mesh_path,
        "status": result.status,
        "error": result.error,
        "features": asdict(result.features),
        "quality": result.quality,
    }


def result_from_dict(data: dict[str, object]) -> SegmentResult:
    return SegmentResult(
        source_file=str(data["source_file"]),
        label=int(data["label"]),
        point_count=int(data["point_count"]),
        geometry_type=str(data["geometry_type"]),
        method=str(data["method"]),
        mesh_path=str(data["mesh_path"]),
        status=str(data["status"]),
        error=str(data["error"]),
        features=SegmentFeatures(**data["features"]),  # type: ignore[arg-type]
        quality=data["quality"],  # type: ignore[arg-type]
    )


def failed_file_result(path: Path, error: str) -> SegmentResult:
    return SegmentResult(
        source_file=path.name,
        label=-1,
        point_count=0,
        geometry_type="file_error",
        method="none",
        mesh_path="",
        status="failed",
        error=error,
        features=SegmentFeatures(
            point_count=0,
            bbox_volume=0.0,
            density=0.0,
            mean_nn_distance=0.0,
            linearity=0.0,
            planarity=0.0,
            sphericity=0.0,
            curvature=0.0,
            normal_consistency=0.0,
            connected_components=0,
            largest_component_ratio=0.0,
        ),
        quality={
            "mesh_vertices": 0,
            "mesh_triangles": 0,
            "mean_cloud_to_mesh": math.inf,
            "p95_cloud_to_mesh": math.inf,
            "max_cloud_to_mesh": math.inf,
            "accuracy_ratio": 0.0,
            "mesh_components": 0,
        },
    )


def process_cloud_isolated(path: Path, cfg: PipelineConfig, seed: int) -> list[SegmentResult]:
    tmp_dir = cfg.metrics_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_json = tmp_dir / f"{path.stem}_{seed}.json"
    if output_json.exists():
        output_json.unlink()

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_reconstruction.py"
    command = [
        sys.executable,
        str(script_path),
        "--single-file",
        str(path),
        "--single-output-json",
        str(output_json),
        "--data-dir",
        str(cfg.data_dir),
        "--output-dir",
        str(cfg.output_dir),
        "--metrics-dir",
        str(cfg.metrics_dir),
        "--visualization-dir",
        str(cfg.visualization_dir),
        "--min-segment-points",
        str(cfg.min_segment_points),
        "--voxel-size",
        str(cfg.voxel_size),
        "--max-points-per-segment",
        str(cfg.max_points_per_segment),
        "--normal-knn",
        str(cfg.normal_knn),
        "--poisson-depth",
        str(cfg.poisson_depth),
        "--seed",
        str(seed),
        "--quiet-files",
        "--no-isolate-files",
    ]
    if cfg.force_method:
        command.extend(["--force-method", cfg.force_method])
    if not cfg.save_segment_meshes:
        command.append("--no-segment-meshes")
    if not cfg.save_visualizations:
        command.append("--no-visualizations")

    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip().splitlines()[-8:]
        stdout_tail = completed.stdout.strip().splitlines()[-4:]
        details = "\n".join(stdout_tail + stderr_tail).strip()
        return [failed_file_result(path, f"child process exited with code {completed.returncode}: {details}")]

    data = json.loads(output_json.read_text(encoding="utf-8"))
    output_json.unlink(missing_ok=True)
    return [result_from_dict(item) for item in data]


def log_progress(message: str, cfg: PipelineConfig) -> None:
    if cfg.log_files:
        tqdm.write(message)


def run_pipeline(cfg: PipelineConfig) -> list[SegmentResult]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.visualization_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cfg.data_dir.glob("*.ply"))
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise FileNotFoundError(f"no .ply files found in {cfg.data_dir}")

    all_results: list[SegmentResult] = []
    errors_path = cfg.metrics_dir / "errors.log"
    if errors_path.exists():
        errors_path.unlink()

    workers = max(1, min(cfg.workers, len(files)))
    if workers == 1:
        for index, path in enumerate(tqdm(files, desc="Reconstructing clouds")):
            log_progress(f"[start] {index + 1}/{len(files)} {path.name}", cfg)
            try:
                if cfg.isolate_files:
                    all_results.extend(process_cloud_isolated(path, cfg, cfg.seed + index))
                else:
                    all_results.extend(process_cloud_worker(path, cfg, cfg.seed + index))
                persist_metrics(cfg, all_results)
                log_progress(f"[done]  {index + 1}/{len(files)} {path.name}", cfg)
            except Exception as exc:  # noqa: BLE001 - keep processing other source files.
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n[{path.name}] {type(exc).__name__}: {exc}\n")
                    f.write(traceback.format_exc())
                log_progress(f"[fail]  {index + 1}/{len(files)} {path.name}: {type(exc).__name__}", cfg)
                all_results.append(failed_file_result(path, f"{type(exc).__name__}: {exc}"))
                persist_metrics(cfg, all_results)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {}
            next_index = 0

            def submit_next() -> None:
                nonlocal next_index
                if next_index >= len(files):
                    return
                index = next_index
                path = files[index]
                next_index += 1
                log_progress(f"[start] {index + 1}/{len(files)} {path.name}", cfg)
                if cfg.isolate_files:
                    future = executor.submit(process_cloud_isolated, path, cfg, cfg.seed + index)
                else:
                    future = executor.submit(process_cloud_worker, path, cfg, cfg.seed + index)
                future_to_item[future] = (index, path)

            for _ in range(workers):
                submit_next()

            with tqdm(total=len(files), desc=f"Reconstructing clouds ({workers} workers)") as pbar:
                while future_to_item:
                    for future in as_completed(list(future_to_item)):
                        index, path = future_to_item.pop(future)
                        break
                    try:
                        all_results.extend(future.result())
                        persist_metrics(cfg, all_results)
                        log_progress(f"[done]  {index + 1}/{len(files)} {path.name}", cfg)
                    except Exception as exc:  # noqa: BLE001 - keep processing other source files.
                        with errors_path.open("a", encoding="utf-8") as f:
                            f.write(f"\n[{path.name}] {type(exc).__name__}: {exc}\n")
                            f.write(traceback.format_exc())
                        log_progress(f"[fail]  {index + 1}/{len(files)} {path.name}: {type(exc).__name__}", cfg)
                        all_results.append(failed_file_result(path, f"{type(exc).__name__}: {exc}"))
                        persist_metrics(cfg, all_results)
                    pbar.update(1)
                    submit_next()

    all_results.sort(key=lambda item: (item.source_file, item.label))
    persist_metrics(cfg, all_results)
    return all_results


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Adaptive surface reconstruction for task 2.")
    parser.add_argument("--data-dir", type=Path, default=root.parent / "data", help="Directory with source PLY files.")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs", help="Directory for reconstructed meshes.")
    parser.add_argument("--metrics-dir", type=Path, default=root / "metrics", help="Directory for CSV/JSON metrics.")
    parser.add_argument("--visualization-dir", type=Path, default=root / "visualizations", help="Directory for PNG previews.")
    parser.add_argument("--min-segment-points", type=int, default=80)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsampling size in source units.")
    parser.add_argument("--max-points-per-segment", type=int, default=3500)
    parser.add_argument("--normal-knn", type=int, default=24)
    parser.add_argument("--poisson-depth", type=int, default=7)
    parser.add_argument("--max-files", type=int, default=None, help="Debug option: process only first N files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel file workers. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--force-method",
        choices=[METHOD_POISSON, METHOD_ALPHA, METHOD_BALL_PIVOTING],
        default=None,
        help="Use one fixed baseline method instead of adaptive selection.",
    )
    parser.add_argument("--no-segment-meshes", action="store_true", help="Do not save per-segment meshes.")
    parser.add_argument("--no-visualizations", action="store_true", help="Do not save PNG previews.")
    parser.add_argument("--quiet-files", action="store_true", help="Hide per-file start/done progress messages.")
    parser.add_argument("--no-isolate-files", action="store_true", help="Run files in the main process instead of isolated child processes.")
    parser.add_argument("--single-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-output-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        visualization_dir=args.visualization_dir,
        min_segment_points=args.min_segment_points,
        voxel_size=args.voxel_size,
        max_points_per_segment=args.max_points_per_segment,
        normal_knn=args.normal_knn,
        poisson_depth=args.poisson_depth,
        max_files=args.max_files,
        seed=args.seed,
        workers=args.workers,
        force_method=args.force_method,
        isolate_files=not args.no_isolate_files,
        save_segment_meshes=not args.no_segment_meshes,
        save_visualizations=not args.no_visualizations,
        log_files=not args.quiet_files,
    )

    if args.single_file is not None:
        if args.single_output_json is None:
            raise ValueError("--single-output-json is required with --single-file")
        single_rng = np.random.default_rng(cfg.seed)
        single_results = process_cloud(args.single_file, cfg, single_rng)
        args.single_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.single_output_json.write_text(
            json.dumps([result_to_dict(item) for item in single_results], ensure_ascii=False),
            encoding="utf-8",
        )
        return

    results = run_pipeline(cfg)
    ok = sum(1 for item in results if item.status == "ok")
    print(f"Done. Segments: {len(results)}, reconstructed: {ok}, failed: {len(results) - ok}")
    print(f"Metrics: {cfg.metrics_dir / 'segments_metrics.csv'}")
    print(f"Summary: {cfg.metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
