from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MODELNET_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"


def _download_and_unpack(target_dir: Path) -> None:
    """Fetch ModelNet10 and unpack it next to the target directory."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir.parent / "ModelNet10.zip"

    print(f"Downloading ModelNet10 to {archive_path}...")
    urllib.request.urlretrieve(MODELNET_URL, archive_path)
    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir.parent)
    archive_path.unlink(missing_ok=True)

    extracted = target_dir.parent / "ModelNet10"
    if extracted.exists() and extracted != target_dir:
        extracted.rename(target_dir)


def read_off(path: Path) -> np.ndarray:
    """Читает файл формата .off и возвращает массив вершин меша.
    
    Формат .off начинается с заголовка "OFF", затем количество вершин и граней.
    Возвращает только координаты вершин как массив (N, 3).
    """
    with path.open("r") as src:
        if src.readline().strip() != "OFF":
            raise ValueError(f"{path} is not an OFF file")
        counts = src.readline().strip().split()
        if len(counts) < 2:
            raise ValueError(f"Corrupted header in {path}")
        n_vertices = int(counts[0])

        vertices = []
        for _ in range(n_vertices):
            vertices.append(list(map(float, src.readline().strip().split())))
    return np.asarray(vertices, dtype=np.float32)


def select_points(vertices: np.ndarray, num_points: int) -> np.ndarray:
    """Выбирает фиксированное количество точек из вершин меша.
    
    Если вершин меньше требуемого, используется выборка с возвратом.
    Иначе — случайная выборка без возврата.
    """
    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices to sample")
    replace = len(vertices) < num_points
    ids = np.random.choice(len(vertices), num_points, replace=replace)
    return vertices[ids]


def center_and_scale(points: np.ndarray) -> np.ndarray:
    """Центрирует облако точек и масштабирует до единичного радиуса.
    
    Сначала вычитает среднее по каждой координате, затем делит на максимальное
    расстояние от центра, чтобы все точки оказались в единичной сфере.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    max_dist = np.linalg.norm(centered, axis=1).max()
    return centered / max(max_dist, 1e-8)


def collect_split(
    root: Path, split: str, num_points: int, per_class_cap: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Собирает данные для одного сплита (train/test) из директории датасета.
    
    Обходит поддиректории классов, читает .off файлы, нормализует и сэмплирует точки.
    Возвращает массивы облаков, меток и список имён классов.
    """
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    samples, labels = [], []

    for class_idx, class_name in enumerate(classes):
        split_dir = root / class_name / split
        if not split_dir.exists():
            continue

        off_files = sorted(split_dir.glob("*.off"))
        # Ограничение количества примеров на класс для ускорения
        if per_class_cap:
            off_files = off_files[:per_class_cap]

        for off_file in off_files:
            verts = read_off(off_file)
            points = center_and_scale(select_points(verts, num_points))
            samples.append(points)
            labels.append(class_idx)

    if not samples:
        raise RuntimeError(f"No samples found in {root} for split '{split}'")

    return (
        np.stack(samples).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        classes,
    )


class PointCloudSet(Dataset):
    """Простой датасет для облаков точек.
    
    Хранит предобработанные облака и метки, возвращает их как тензоры PyTorch.
    Каждое облако имеет форму (N, 3) — N точек с координатами x, y, z.
    """

    def __init__(self, clouds: np.ndarray, labels: np.ndarray):
        self.clouds = clouds
        self.labels = labels

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.clouds[idx]), int(self.labels[idx])


@dataclass
class DataConfig:
    root: Path
    num_points: int
    train_cap: Optional[int]
    test_cap: Optional[int]
    download_if_missing: bool


def prepare_dataloaders(
    cfg: DataConfig, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    """Создаёт DataLoader'ы для обучения и тестирования.
    
    Загружает данные из указанной директории, применяет предобработку,
    создаёт датасеты и обёртки DataLoader с заданными параметрами батчинга.
    """
    root = Path(cfg.root).expanduser()
    if not root.exists():
        if cfg.download_if_missing:
            _download_and_unpack(root)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {root}. Set download_if_missing=true to auto-download."
            )

    train_data, train_labels, classes = collect_split(
        root, "train", cfg.num_points, cfg.train_cap
    )
    test_data, test_labels, _ = collect_split(
        root, "test", cfg.num_points, cfg.test_cap
    )

    train_ds = PointCloudSet(train_data, train_labels)
    test_ds = PointCloudSet(test_data, test_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, classes
