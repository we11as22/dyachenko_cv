from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


@dataclass
class AnalysisConfig:
    data_dir: Path
    metrics_dir: Path
    visualization_dir: Path
    figures_dir: Path
    min_segment_points: int = 80
    max_points_per_segment: int = 3500
    normal_knn: int = 24
    density_knn: int = 12
    max_files: int | None = None
    seed: int = 42
    save_visualizations: bool = True
    save_figures: bool = True
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


@dataclass
class SegmentAnalysis:
    source_file: str
    label: int
    point_count: int
    # 6.1 density
    mean_nn_distance: float
    nn_distance_std: float
    nn_uniformity: float
    sparse_region_ratio: float
    dense_region_ratio: float
    density_class: str
    density_interpretation: str
    # 6.2 shape
    linearity: float
    planarity: float
    sphericity: float
    anisotropy: float
    dominant_axes: int
    shape_class: str
    shape_interpretation: str
    # 6.3 surface
    curvature: float
    normal_variation: float
    surface_class: str
    surface_interpretation: str
    # 6.4 normals
    normal_consistency: float
    sharp_normal_ratio: float
    normal_class: str
    normal_interpretation: str
    # 6.5 topology
    connected_components: int
    largest_component_ratio: float
    isolated_clusters: int
    mean_neighbors: float
    integrity_class: str
    topology_interpretation: str
    # 6.6 summary
    summary: str
    status: str
    error: str


def read_ascii_ply(path: Path) -> PointCloudData:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "ply":
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
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError("invalid vertex count")

        required = ["x", "y", "z", "scalar_Label"]
        missing = [name for name in required if name not in properties]
        if missing:
            raise ValueError(f"missing PLY properties: {missing}")

        raw = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

    col = {name: idx for idx, name in enumerate(properties)}
    points = raw[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    labels = raw[:, col["scalar_Label"]].astype(np.int32)
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains NaN or Inf")
    return PointCloudData(path=path, points=points, labels=labels)


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        return centered
    return centered / scale


def build_segments(cloud: PointCloudData, cfg: AnalysisConfig, rng: np.random.Generator) -> list[Segment]:
    segments: list[Segment] = []
    for label in sorted(np.unique(cloud.labels).tolist()):
        label_points = cloud.points[cloud.labels == label]
        if len(label_points) < cfg.min_segment_points:
            continue
        if len(label_points) > cfg.max_points_per_segment:
            idx = rng.choice(len(label_points), size=cfg.max_points_per_segment, replace=False)
            label_points = label_points[idx]
        segments.append(
            Segment(
                source_file=cloud.path.name,
                label=int(label),
                points=label_points,
                normalized_points=normalize_points(label_points),
            )
        )
    return segments


def to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def estimate_normals(points: np.ndarray, knn: int) -> np.ndarray:
    pcd = to_point_cloud(points)
    if len(points) < 3:
        return np.zeros((len(points), 3))
    knn = max(3, min(knn, len(points) - 1))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=max(3, min(knn, len(points) - 1)))
    except RuntimeError:
        pass
    return np.asarray(pcd.normals)


def nn_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    nn = NearestNeighbors(n_neighbors=min(2, len(points))).fit(points)
    dist, _ = nn.kneighbors(points)
    return dist[:, 1] if dist.shape[1] > 1 else dist[:, 0]


def local_density_scores(points: np.ndarray, knn: int) -> np.ndarray:
    if len(points) < 3:
        return np.ones(len(points))
    k = max(2, min(knn, len(points)))
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    dist, _ = nn.kneighbors(points)
    return 1.0 / np.maximum(dist[:, 1:].mean(axis=1), 1e-8)


def pca_analysis(points: np.ndarray) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    if eigvals[0] <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, eigvals, eigvecs
    l1, l2, l3 = eigvals
    linearity = float((l1 - l2) / l1)
    planarity = float((l2 - l3) / l1)
    sphericity = float(l3 / l1)
    curvature = float(l3 / max(l1 + l2 + l3, 1e-12))
    anisotropy = float((l1 - l3) / l1)
    return linearity, planarity, sphericity, curvature, anisotropy, eigvals, eigvecs


def normal_stats(points: np.ndarray, normals: np.ndarray, knn: int) -> tuple[float, float]:
    if len(points) < 4:
        return 0.0, 0.0
    k = max(2, min(knn, len(points)))
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nn.kneighbors(points)
    dots: list[float] = []
    sharp = 0
    for row, normal in zip(indices, normals):
        neighbor_normals = normals[row[1:]]
        local_dots = np.abs(neighbor_normals @ normal)
        dots.extend(local_dots.tolist())
        sharp += int(np.mean(local_dots) < 0.55)
    consistency = float(np.mean(dots)) if dots else 0.0
    sharp_ratio = float(sharp / len(points))
    normal_variation = float(1.0 - consistency)
    return consistency, sharp_ratio if normal_variation == normal_variation else 0.0


def topology_stats(points: np.ndarray, mean_nn: float, nn_values: np.ndarray) -> tuple[int, float, float, int]:
    if len(points) == 0:
        return 0, 0.0, 0.0, 0
    if len(points) == 1:
        return 1, 1.0, 0.0, 0
    radius = float(max(mean_nn * 2.8, np.percentile(nn_values, 75) * 2.0, 1e-4))
    nn = NearestNeighbors(radius=radius).fit(points)
    neighbors = nn.radius_neighbors(points, return_distance=False)
    neighbor_counts = np.array([len(n) - 1 for n in neighbors], dtype=np.float64)
    visited = np.zeros(len(points), dtype=bool)
    sizes: list[int] = []
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
        sizes.append(size)
    largest = max(sizes) if sizes else 0
    isolated = sum(1 for s in sizes if s < max(5, int(0.05 * len(points))))
    return len(sizes), float(largest / len(points)), float(neighbor_counts.mean()), max(0, isolated - (1 if len(sizes) > 1 else 0))


def classify_density(mean_nn: float, uniformity: float, sparse_ratio: float, dense_ratio: float) -> tuple[str, str]:
    if mean_nn < 0.045 and uniformity < 0.55:
        cls = "high"
        text = "высокая плотность, равномерное распределение точек"
    elif mean_nn > 0.085 or sparse_ratio > 0.25:
        cls = "low"
        text = "низкая плотность или выраженные разреженные области"
    else:
        cls = "medium"
        text = "средняя плотность с умеренной неоднородностью"
    if dense_ratio > 0.20:
        text += "; есть локальные сгущения"
    return cls, text


def classify_shape(linearity: float, planarity: float, sphericity: float, anisotropy: float) -> tuple[int, str, str]:
    if linearity > 0.55 and planarity < 0.35:
        axes, cls = 1, "linear"
        text = "доминирует одна ось — линейная структура"
    elif planarity > 0.45 and linearity < 0.55:
        axes, cls = 2, "planar"
        text = "две доминирующие оси — плоская структура"
    elif sphericity > 0.18 and anisotropy < 0.75:
        axes, cls = 3, "volumetric"
        text = "равномерное распределение по осям — объёмная форма"
    else:
        axes, cls = 2, "anisotropic"
        text = "анизотропное распределение без чёткой доминирующей оси"
    return axes, cls, text


def classify_surface(curvature: float, normal_variation: float, sharp_ratio: float) -> tuple[str, str]:
    if curvature < 0.06 and normal_variation < 0.15 and sharp_ratio < 0.08:
        cls = "smooth"
        text = "гладкая поверхность"
    elif curvature < 0.12 and sharp_ratio < 0.15:
        cls = "weakly_curved"
        text = "слабо искривлённая поверхность"
    elif curvature >= 0.12 and sharp_ratio < 0.25:
        cls = "strongly_curved"
        text = "сильно искривлённая поверхность"
    else:
        cls = "complex"
        text = "сложная комбинированная геометрия с резкими изменениями формы"
    return cls, text


def classify_normals(consistency: float, sharp_ratio: float) -> tuple[str, str]:
    if consistency > 0.85:
        cls, text = "high", "высокая согласованность нормалей — плоская или гладкая поверхность"
    elif consistency > 0.65:
        cls, text = "medium", "средняя согласованность — цилиндрическая или сферическая форма"
    else:
        cls, text = "low", "низкая согласованность — сложная геометрия или шум"
    if sharp_ratio > 0.12:
        text += "; выявлены области резкого изменения нормалей"
    return cls, text


def classify_topology(components: int, largest_ratio: float, isolated_clusters: int, mean_neighbors: float) -> tuple[str, str]:
    if components == 1 and largest_ratio > 0.95:
        cls = "intact"
        text = "сегмент топологически целостный, разрывов нет"
    elif largest_ratio > 0.75 and isolated_clusters <= 1:
        cls = "mostly_intact"
        text = "преобладает одна связная компонента, возможны локальные разрывы"
    else:
        cls = "fragmented"
        text = "выявлены разрывы, изолированные кластеры или пропуски"
    text += f"; среднее число соседей в радиусе связности: {mean_neighbors:.1f}"
    return cls, text


def build_summary(result: SegmentAnalysis) -> str:
    return (
        f"Сегмент label={result.label}: {result.density_interpretation}. "
        f"Форма: {result.shape_interpretation}. "
        f"Поверхность: {result.surface_interpretation}. "
        f"Нормали: {result.normal_interpretation}. "
        f"Топология: {result.topology_interpretation}."
    )


def analyze_segment(segment: Segment, cfg: AnalysisConfig) -> SegmentAnalysis:
    points = segment.normalized_points
    nn_values = nn_distances(points)
    mean_nn = float(np.mean(nn_values)) if len(nn_values) else 0.0
    nn_std = float(np.std(nn_values)) if len(nn_values) else 0.0
    uniformity = float(nn_std / max(mean_nn, 1e-8))
    local_density = local_density_scores(points, cfg.density_knn)
    p20, p80 = np.percentile(local_density, [20, 80])
    sparse_ratio = float(np.mean(local_density <= p20))
    dense_ratio = float(np.mean(local_density >= p80))
    density_class, density_text = classify_density(mean_nn, uniformity, sparse_ratio, dense_ratio)

    linearity, planarity, sphericity, curvature, anisotropy, _, _ = pca_analysis(points)
    dominant_axes, shape_class, shape_text = classify_shape(linearity, planarity, sphericity, anisotropy)

    normals = estimate_normals(points, cfg.normal_knn)
    consistency, sharp_ratio = normal_stats(points, normals, cfg.normal_knn)
    normal_variation = float(1.0 - consistency)
    surface_class, surface_text = classify_surface(curvature, normal_variation, sharp_ratio)
    normal_class, normal_text = classify_normals(consistency, sharp_ratio)

    components, largest_ratio, mean_neighbors, isolated_clusters = topology_stats(points, mean_nn, nn_values)
    integrity_class, topology_text = classify_topology(components, largest_ratio, isolated_clusters, mean_neighbors)

    result = SegmentAnalysis(
        source_file=segment.source_file,
        label=segment.label,
        point_count=len(points),
        mean_nn_distance=mean_nn,
        nn_distance_std=nn_std,
        nn_uniformity=uniformity,
        sparse_region_ratio=sparse_ratio,
        dense_region_ratio=dense_ratio,
        density_class=density_class,
        density_interpretation=density_text,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        anisotropy=anisotropy,
        dominant_axes=dominant_axes,
        shape_class=shape_class,
        shape_interpretation=shape_text,
        curvature=curvature,
        normal_variation=normal_variation,
        surface_class=surface_class,
        surface_interpretation=surface_text,
        normal_consistency=consistency,
        sharp_normal_ratio=sharp_ratio,
        normal_class=normal_class,
        normal_interpretation=normal_text,
        connected_components=components,
        largest_component_ratio=largest_ratio,
        isolated_clusters=isolated_clusters,
        mean_neighbors=mean_neighbors,
        integrity_class=integrity_class,
        topology_interpretation=topology_text,
        summary="",
        status="ok",
        error="",
    )
    result.summary = build_summary(result)
    return result


def component_labels(points: np.ndarray, mean_nn: float, nn_values: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros(len(points), dtype=int)
    radius = float(max(mean_nn * 2.8, np.percentile(nn_values, 75) * 2.0, 1e-4))
    nn = NearestNeighbors(radius=radius).fit(points)
    neighbors = nn.radius_neighbors(points, return_distance=False)
    labels = -np.ones(len(points), dtype=int)
    comp_id = 0
    for start in range(len(points)):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = comp_id
        while stack:
            current = stack.pop()
            for nxt in neighbors[current]:
                if labels[nxt] == -1:
                    labels[nxt] = comp_id
                    stack.append(int(nxt))
        comp_id += 1
    return labels


def save_visualization(segment: Segment, result: SegmentAnalysis, output_path: Path, cfg: AnalysisConfig) -> None:
    points = segment.normalized_points
    nn_values = nn_distances(points)
    local_density = local_density_scores(points, cfg.density_knn)
    comp_ids = component_labels(points, result.mean_nn_distance, nn_values)
    normals = estimate_normals(points, cfg.normal_knn)
    consistency_map = np.zeros(len(points))
    k = max(2, min(cfg.normal_knn, len(points)))
    if len(points) >= 4:
        nn = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = nn.kneighbors(points)
        for i, row in enumerate(indices):
            consistency_map[i] = float(np.mean(np.abs(normals[row[1:]] @ normals[i])))

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    sc1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=local_density, s=3, cmap="viridis")
    ax1.set_title("6.1 Плотность точек")
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label="local density")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _, _, _, _, _, eigvals, eigvecs = pca_analysis(points)
    colors = plt.cm.tab10(comp_ids % 10)
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=3)
    center = points.mean(axis=0)
    for val, vec, color in zip(eigvals, eigvecs.T, ["r", "g", "b"]):
        if val <= 1e-12:
            continue
        direction = vec / max(np.linalg.norm(vec), 1e-8) * np.sqrt(val)
        ax2.plot([center[0], center[0] + direction[0]], [center[1], center[1] + direction[1]], [center[2], center[2] + direction[2]], color=color, linewidth=2)
    ax2.set_title("6.2 Форма и PCA-оси")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    sc3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=consistency_map, s=3, cmap="coolwarm", vmin=0, vmax=1)
    ax3.set_title("6.4 Согласованность нормалей")
    fig.colorbar(sc3, ax=ax3, shrink=0.6)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=comp_ids, s=3, cmap="tab20")
    ax4.set_title("6.5 Компоненты связности")

    text = (
        f"{result.source_file} | label={result.label}\n"
        f"6.1 {result.density_class}: {result.density_interpretation}\n"
        f"6.2 {result.shape_class}: {result.shape_interpretation}\n"
        f"6.3 {result.surface_class}: {result.surface_interpretation}\n"
        f"6.4 {result.normal_class}: {result.normal_interpretation}\n"
        f"6.5 {result.integrity_class}: {result.topology_interpretation}\n"
        f"6.6 {result.summary}"
    )
    fig.text(0.02, 0.01, text, fontsize=8, va="bottom", wrap=True)
    fig.suptitle(f"Геометрический анализ сегмента {segment.source_file} / label {segment.label}", fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(path: Path, results: list[SegmentAnalysis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, cfg: AnalysisConfig, results: list[SegmentAnalysis]) -> None:
    ok = [r for r in results if r.status == "ok"]
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "segments_total": len(results),
        "segments_ok": len(ok),
        "density_classes": {},
        "shape_classes": {},
        "surface_classes": {},
        "normal_classes": {},
        "integrity_classes": {},
    }
    for result in ok:
        for key, attr in [
            ("density_classes", "density_class"),
            ("shape_classes", "shape_class"),
            ("surface_classes", "surface_class"),
            ("normal_classes", "normal_class"),
            ("integrity_classes", "integrity_class"),
        ]:
            value = getattr(result, attr)
            summary[key][value] = summary[key].get(value, 0) + 1
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def process_cloud(path: Path, cfg: AnalysisConfig, rng: np.random.Generator) -> list[SegmentAnalysis]:
    cloud = read_ascii_ply(path)
    segments = build_segments(cloud, cfg, rng)
    results: list[SegmentAnalysis] = []
    vis_dir = cfg.visualization_dir / cloud.path.stem
    for segment in segments:
        try:
            result = analyze_segment(segment, cfg)
            if cfg.save_visualizations:
                save_visualization(segment, result, vis_dir / f"label_{segment.label:02d}_analysis.png", cfg)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append(
                SegmentAnalysis(
                    source_file=cloud.path.name,
                    label=segment.label,
                    point_count=len(segment.points),
                    mean_nn_distance=0,
                    nn_distance_std=0,
                    nn_uniformity=0,
                    sparse_region_ratio=0,
                    dense_region_ratio=0,
                    density_class="error",
                    density_interpretation="",
                    linearity=0,
                    planarity=0,
                    sphericity=0,
                    anisotropy=0,
                    dominant_axes=0,
                    shape_class="error",
                    shape_interpretation="",
                    curvature=0,
                    normal_variation=0,
                    surface_class="error",
                    surface_interpretation="",
                    normal_consistency=0,
                    sharp_normal_ratio=0,
                    normal_class="error",
                    normal_interpretation="",
                    connected_components=0,
                    largest_component_ratio=0,
                    isolated_clusters=0,
                    mean_neighbors=0,
                    integrity_class="error",
                    topology_interpretation="",
                    summary="",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def export_figures(results: list[SegmentAnalysis], cfg: AnalysisConfig) -> None:
    if not cfg.save_figures:
        return
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    picks: dict[str, SegmentAnalysis] = {}
    for target, key in [
        ("linear", "shape_class"),
        ("planar", "shape_class"),
        ("volumetric", "shape_class"),
        ("smooth", "surface_class"),
        ("complex", "surface_class"),
        ("fragmented", "integrity_class"),
    ]:
        for result in results:
            if result.status != "ok":
                continue
            if getattr(result, key) == target and target not in picks:
                picks[target] = result
    for name, result in picks.items():
        src = cfg.visualization_dir / Path(result.source_file).stem / f"label_{result.label:02d}_analysis.png"
        dst = cfg.figures_dir / f"{name}_{Path(result.source_file).stem}_label_{result.label:02d}.png"
        if src.exists():
            dst.write_bytes(src.read_bytes())


def run_pipeline(cfg: AnalysisConfig) -> list[SegmentAnalysis]:
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.visualization_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    files = sorted(cfg.data_dir.glob("*.ply"))
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise FileNotFoundError(f"no .ply files found in {cfg.data_dir}")

    all_results: list[SegmentAnalysis] = []
    errors_path = cfg.metrics_dir / "errors.log"
    if errors_path.exists():
        errors_path.unlink()

    for path in tqdm(files, desc="Analyzing segments"):
        try:
            all_results.extend(process_cloud(path, cfg, rng))
        except Exception as exc:  # noqa: BLE001
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{path.name}] {type(exc).__name__}: {exc}\n")
                f.write(traceback.format_exc())

    all_results.sort(key=lambda item: (item.source_file, item.label))
    write_csv(cfg.metrics_dir / "segments_analysis.csv", all_results)
    write_summary(cfg.metrics_dir / "summary.json", cfg, all_results)
    export_figures(all_results, cfg)
    return all_results


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Geometric segment analysis for task 3.")
    parser.add_argument("--data-dir", type=Path, default=root.parent / "data")
    parser.add_argument("--metrics-dir", type=Path, default=root / "metrics")
    parser.add_argument("--visualization-dir", type=Path, default=root / "visualizations")
    parser.add_argument("--figures-dir", type=Path, default=root / "figures")
    parser.add_argument("--min-segment-points", type=int, default=80)
    parser.add_argument("--max-points-per-segment", type=int, default=3500)
    parser.add_argument("--normal-knn", type=int, default=24)
    parser.add_argument("--density-knn", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig(
        data_dir=args.data_dir,
        metrics_dir=args.metrics_dir,
        visualization_dir=args.visualization_dir,
        figures_dir=args.figures_dir,
        min_segment_points=args.min_segment_points,
        max_points_per_segment=args.max_points_per_segment,
        normal_knn=args.normal_knn,
        density_knn=args.density_knn,
        max_files=args.max_files,
        seed=args.seed,
        save_visualizations=not args.no_visualizations,
        save_figures=not args.no_figures,
    )
    results = run_pipeline(cfg)
    ok = sum(1 for item in results if item.status == "ok")
    print(f"Done. Segments: {len(results)}, analyzed: {ok}, failed: {len(results) - ok}")
    print(f"Metrics: {cfg.metrics_dir / 'segments_analysis.csv'}")
    print(f"Summary: {cfg.metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
