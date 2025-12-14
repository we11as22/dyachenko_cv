import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def read_ply_with_labels(filepath):
    """Чтение PLY-файла с координатами точек и метками классов"""
    with open(filepath, "r") as f:
        # Пропускаем заголовок до строки end_header
        line = f.readline()
        while line:
            if line.strip() == "end_header":
                break
            line = f.readline()

        # Читаем данные вершин (x, y, z, метка)
        coordinates = []
        class_labels = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # x, y, z, метка
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                label = (
                    int(parts[3]) if len(parts) > 3 else 0
                )  # метка по умолчанию 0, если не указана
                coordinates.append([x, y, z])
                class_labels.append(label)

        return np.array(coordinates), np.array(class_labels)


def downsample_points(coords, labels, target_count=1024):
    """Сэмплирование точек из облака до заданного количества"""
    if target_count is None:
        return coords, labels

    if len(coords) < target_count:
        selected_indices = np.random.choice(len(coords), target_count, replace=True)
    else:
        selected_indices = np.random.choice(len(coords), target_count, replace=False)

    sampled_coords = coords[selected_indices]
    sampled_labels = labels[selected_indices]
    return sampled_coords, sampled_labels


def center_and_scale(coords):
    """Нормализация облака точек в диапазон [-1, 1]"""
    center = np.mean(coords, axis=0)
    coords = coords - center

    # Масштабирование до [-1, 1]
    max_distance = np.max(np.sqrt(np.sum(coords**2, axis=1)))
    if max_distance > 0:
        coords = coords / max_distance

    return coords


class PointCloudSegmentationDataset(Dataset):
    def __init__(
        self, root, num_points=1024, split="train", val_sz=0.1, test_sz=0.1
    ):
        self.root = root
        self.num_points = num_points
        self.split = split

        self.files = np.array(os.listdir(root))
        train_idx = np.random.choice(
            len(self.files),
            int((1 - val_sz - test_sz) * len(self.files)),
            replace=False,
        )
        val_idx = np.random.choice(
            list(set(range(len(self.files))) - set(train_idx)),
            int(val_sz * len(self.files)),
            replace=False,
        )
        test_idx = np.array(
            list(set(range(len(self.files))) - set(train_idx) - set(val_idx))
        )

        if split == "full":
            self.files = self.files
        elif split == "train":
            self.files = self.files[train_idx]
        elif split == "val":
            self.files = self.files[val_idx]
        elif split == "test":
            self.files = self.files[test_idx]
        else:
            raise ValueError(f"Неизвестный сплит: {split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        coords, labels = read_ply_with_labels(os.path.join(self.root, file_path))

        coords, labels = downsample_points(coords, labels, self.num_points)

        coords = center_and_scale(coords)

        coords = torch.FloatTensor(coords)

        # Транспонируем в формат (3, N) для входа в PointNet
        coords = coords.transpose(0, 1)

        return coords, torch.LongTensor(labels)


class PointCloudSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        num_points=1024,
        num_classes=13,
        batch_size=2,
        num_workers=4,
        val_sz=0.1,
        test_sz=0.1,
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_sz = val_sz
        self.test_sz = test_sz

    def setup(self, stage=None):
        self.train_dataset = PointCloudSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="train",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.val_dataset = PointCloudSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="val",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.test_dataset = PointCloudSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="test",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = PointCloudSegmentationDataset(
        root="../../../datasets/ds4segmentation", num_points=1024, split="test"
    )
    print(f"Размер датасета: {len(dataset)}")

    # Тест загрузки образца
    coords, labels = dataset[0]
    print(f"Форма облака точек: {coords.shape}")
    print(f"Форма меток: {labels.shape}")
    print(f"Пример меток: {labels[:5]} (первые 5 меток)")
