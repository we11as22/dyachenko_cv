from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentNet(nn.Module):
    """Сеть выравнивания для входных координат или признаков.
    
    Вычисляет матрицу преобразования k×k, которая применяется к входным данным
    для инвариантности к поворотам и сдвигам. Инициализируется единичной матрицей.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # Свёрточные слои для извлечения признаков из точек
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Полносвязные слои для генерации матрицы преобразования
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Извлечение признаков через свёртки
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Глобальная агрегация через максимум по точкам
        x = torch.max(x, dim=2)[0]

        # Генерация параметров матрицы
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Добавление единичной матрицы для стабильности обучения
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        identity = identity.repeat(batch_size, 1)

        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetClassifier(nn.Module):
    """Классификатор облаков точек на основе архитектуры PointNet.
    
    Использует два блока выравнивания: для входных координат и для признаков.
    После извлечения глобальных признаков применяется классификационная головка.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Выравнивание входных координат (3D)
        self.input_align = AlignmentNet(k=3)
        # Выравнивание признаков после первого блока свёрток
        self.feature_align = AlignmentNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, points: torch.Tensor):
        """Прямой проход через сеть.
        
        Args:
            points: Тензор формы (B, 3, N) — батч облаков точек
            
        Returns:
            logits: Логиты для классификации (B, num_classes)
            transform_feat: Матрица выравнивания признаков для регуляризации
        """
        # Применение выравнивания входных координат
        transform_in = self.input_align(points)
        x = torch.bmm(transform_in, points)

        # Первый блок свёрток
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Выравнивание признаков
        transform_feat = self.feature_align(x)
        x = torch.bmm(transform_feat, x)

        # Второй блок свёрток с увеличением размерности признаков
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Глобальная агрегация через максимум
        x = torch.max(x, dim=2)[0]

        # Классификационная головка
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits, transform_feat


def feature_alignment_penalty(transform: torch.Tensor) -> torch.Tensor:
    """Штраф за отклонение матрицы выравнивания от ортогональности.
    
    Поощряет матрицы преобразования быть близкими к ортогональным,
    что сохраняет геометрические свойства облака точек.
    """
    if transform is None:
        return torch.tensor(0.0)

    batch, k, _ = transform.shape
    eye = torch.eye(k, device=transform.device).unsqueeze(0).repeat(batch, 1, 1)
    # Вычисляем отклонение от единичной матрицы
    diff = torch.bmm(transform, transform.transpose(1, 2)) - eye
    return torch.mean(torch.norm(diff, dim=(1, 2)))
