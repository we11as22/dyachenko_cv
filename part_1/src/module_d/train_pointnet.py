from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pointnet_data import DataConfig, prepare_dataloaders
from pointnet_model import PointNetClassifier, feature_alignment_penalty


def pick_device(requested: str) -> torch.device:
    """Выбирает доступное вычислительное устройство.
    
    Поддерживает CPU, CUDA (если доступно) и MPS для Apple Silicon.
    При недоступности запрошенного устройства возвращает CPU.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: PointNetClassifier,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    reg_weight: float,
) -> Tuple[float, float]:
    """Один эпох обучения модели.
    
    Вычисляет loss с регуляризацией матриц выравнивания и обновляет веса.
    Возвращает средний loss и точность на обучающем наборе.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for points, labels in tqdm(loader, desc="train", leave=False):
        # Транспонируем для формата (B, 3, N), ожидаемого моделью
        points = points.to(device).transpose(2, 1)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, align = model(points)
        base_loss = criterion(logits, labels)
        # Добавляем штраф за отклонение матриц выравнивания от ортогональности
        penalty = reg_weight * feature_alignment_penalty(align)
        loss = base_loss + penalty
        loss.backward()
        optimizer.step()

        running_loss += base_loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate_epoch(
    model: PointNetClassifier,
    loader,
    criterion,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, np.ndarray]:
    """Оценка модели на валидационном/тестовом наборе.
    
    Вычисляет loss, точность и строит матрицу ошибок (confusion matrix).
    Возвращает средний loss, точность и матрицу ошибок.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for points, labels in tqdm(loader, desc="eval", leave=False):
            points = points.to(device).transpose(2, 1)
            labels = labels.to(device)

            logits, _ = model(points)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Заполняем матрицу ошибок
            for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                conf[t, p] += 1

    return running_loss / max(len(loader), 1), correct / max(total, 1), conf


def plot_curves(train_losses, test_losses, train_accs, test_accs, save_path: Path):
    """Строит графики динамики loss и accuracy по эпохам.
    
    Создаёт два подграфика: один для loss, другой для accuracy.
    Сохраняет результат в указанный путь.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="train")
    ax1.plot(test_losses, label="test")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="train")
    ax2.plot(test_accs, label="test")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion(conf: np.ndarray, class_names: Sequence[str], save_path: Path):
    """Визуализирует матрицу ошибок (confusion matrix).
    
    Отображает количество правильных и неправильных предсказаний
    для каждого класса в виде тепловой карты.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    cfg = OmegaConf.load(Path(__file__).with_name("conf.yaml"))
    torch.manual_seed(int(cfg.seed))

    data_cfg = DataConfig(
        root=Path(cfg.data.root),
        num_points=int(cfg.data.num_points),
        train_cap=cfg.data.get("train_cap"),
        test_cap=cfg.data.get("test_cap"),
        download_if_missing=bool(cfg.data.download_if_missing),
    )

    device = pick_device(str(cfg.training.device))
    print(f"Running on {device}")

    train_loader, test_loader, classes = prepare_dataloaders(
        data_cfg, batch_size=int(cfg.training.batch_size), num_workers=int(cfg.training.num_workers)
    )

    model = PointNetClassifier(num_classes=len(classes), dropout=float(cfg.model.dropout)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(cfg.training.scheduler.step_size),
        gamma=float(cfg.training.scheduler.gamma),
    )

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
    best_acc = 0.0
    artifacts_dir = Path(cfg.training.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "pointnet.pt"

    for epoch in range(int(cfg.training.max_epochs)):
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            reg_weight=float(cfg.loss.feature_transform_reg_weight),
        )
        val_loss, val_acc, conf = evaluate_epoch(
            model, test_loader, criterion, device, num_classes=len(classes)
        )

        scheduler.step()

        train_losses.append(tr_loss)
        test_losses.append(val_loss)
        train_accs.append(tr_acc)
        test_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"test loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        # Сохраняем модель при улучшении точности на валидации
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes},
                checkpoint_path,
            )
            np.save(artifacts_dir / "confusion.npy", conf)

    plot_curves(
        train_losses,
        test_losses,
        train_accs,
        test_accs,
        Path(cfg.evaluation.curves_path),
    )
    if cfg.evaluation.save_confusion_matrix:
        plot_confusion(conf, classes, Path(cfg.evaluation.confusion_path))

    with (artifacts_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "test_loss": test_losses,
                "train_acc": train_accs,
                "test_acc": test_accs,
                "best_acc": best_acc,
            },
            f,
            indent=2,
        )

    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
