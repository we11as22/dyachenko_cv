"""Минимальная проверка работоспособности пайплайна PointNet.

Загружает небольшой батч данных, пропускает его через модель и проверяет,
что все компоненты (датасет, модель, конфигурация) работают корректно.
Не выполняет обучение, только проверяет целостность пайплайна.
"""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pointnet_data import DataConfig, prepare_dataloaders
from pointnet_model import PointNetClassifier


def run_smoke_test():
    """Проверяет, что датасет загружается, модель инициализируется и делает forward pass."""
    cfg = OmegaConf.load(Path(__file__).with_name("conf.yaml"))
    data_cfg = DataConfig(
        root=Path(cfg.data.root),
        num_points=int(cfg.data.num_points),
        train_cap=min(16, cfg.data.get("train_cap") or 16),
        test_cap=min(8, cfg.data.get("test_cap") or 8),
        download_if_missing=bool(cfg.data.download_if_missing),
    )

    train_loader, _, classes = prepare_dataloaders(
        data_cfg,
        batch_size=min(4, int(cfg.training.batch_size)),
        num_workers=0,
    )

    model = PointNetClassifier(num_classes=len(classes), dropout=float(cfg.model.dropout))
    points, labels = next(iter(train_loader))
    # Транспонируем для формата (B, 3, N)
    logits, _ = model(points.transpose(2, 1))

    print("Smoke test ok:")
    print(f"batch size: {len(points)}, classes: {classes}")
    print(f"logits shape: {tuple(logits.shape)}, labels shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    run_smoke_test()
