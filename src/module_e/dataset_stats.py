from data_loader import PointCloudSegmentationDataset
from torch.utils.data import DataLoader
import numpy as np


def main():
    dataset = PointCloudSegmentationDataset(
        root="../../../datasets/ds4segmentation",
        num_points=None,
        split="full",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Количество образцов в датасете: {len(dataset)}")

    coords, labels = dataset[0]
    print(f"Форма координат: {coords.shape}")
    print(f"Форма меток: {labels.shape}")

    label_counts = {}
    num_classes = 0
    for _, labels in dataloader:
        for label in labels.squeeze(0).numpy():
            label_counts[label] = label_counts.get(label, 0) + 1
    print("Статистика меток в датасете:")
    for label, count in label_counts.items():
        print(f"Метка {label}: {count} точек")
        num_classes += 1
    weights = np.array(
        [1.0 / label_counts[i] for i in sorted(label_counts.keys())]
    )
    weights = weights / np.sum(weights)

    print(f"Количество классов: {num_classes}")
    print(f"Веса классов (по частоте): {weights}")
    print(
        f"Среднее количество точек на облако: {sum(label_counts.values()) / len(dataloader)}"
    )


if __name__ == "__main__":
    main()
