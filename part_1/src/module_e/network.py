import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
import matplotlib.pyplot as plt


class TransformationNetwork(nn.Module):
    """T-Net: обучает матрицу трансформации для выравнивания входных данных или признаков"""

    def __init__(self, dim=3):
        super(TransformationNetwork, self).__init__()
        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim * dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        # Общая MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Максимальный пулинг
        x = torch.max(x, 2, keepdim=False)[0]

        # Полносвязные слои
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Инициализация как единичная матрица
        identity = (
            torch.eye(self.dim, device=x.device)
            .flatten()
            .view(1, -1)
            .repeat(batch_size, 1)
        )
        x = x + identity
        x = x.view(-1, self.dim, self.dim)

        return x


class PointNetBackbone(nn.Module):
    """Базовый экстрактор признаков PointNet"""

    def __init__(self, global_features=True, use_feature_transform=True):
        super(PointNetBackbone, self).__init__()
        self.global_features = global_features
        self.use_feature_transform = use_feature_transform

        # Трансформация входных данных
        self.input_transform = TransformationNetwork(dim=3)

        # Общие MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Трансформация признаков
        if self.use_feature_transform:
            self.feature_transform = TransformationNetwork(dim=64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_points = x.size(2)

        # Трансформация входных данных
        input_trans = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_trans)
        x = x.transpose(2, 1)

        # Общая MLP [64, 64]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Трансформация признаков
        if self.use_feature_transform:
            feat_trans = self.feature_transform(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feat_trans)
            x = x.transpose(2, 1)
        else:
            feat_trans = None

        # Сохраняем точечные признаки до max pooling
        point_features = x

        # Общая MLP [64, 128, 1024]
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling для глобального признака
        global_feat = torch.max(x, 2, keepdim=True)[0]

        if self.global_features:
            return global_feat, input_trans, feat_trans
        else:
            # Расширяем и конкатенируем с точечными признаками для сегментации
            global_feat = global_feat.repeat(1, 1, n_points)
            return torch.cat([point_features, global_feat], 1), input_trans, feat_trans


class PointNetSegmentationNetwork(nn.Module):
    """PointNet для семантической сегментации облаков точек"""

    def __init__(self, num_classes, dropout=0.3, use_feature_transform=True):
        super(PointNetSegmentationNetwork, self).__init__()
        self.num_classes = num_classes
        self.use_feature_transform = use_feature_transform

        # Экстрактор признаков
        self.backbone = PointNetBackbone(
            global_features=False, use_feature_transform=use_feature_transform
        )

        # Голова сегментации
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, 3, num_points]
        x, input_trans, feat_trans = self.backbone(x)

        # MLP сегментации
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        # x: [batch_size, num_classes, num_points]
        x = x.transpose(2, 1).contiguous()
        # x: [batch_size, num_points, num_classes]

        return F.log_softmax(x, dim=-1), input_trans, feat_trans


class PointNetSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning модуль для сегментации облаков точек на основе PointNet
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.network = PointNetSegmentationNetwork(
            num_classes=cfg.data.num_classes,
            dropout=cfg.model.dropout,
            use_feature_transform=cfg.model.feature_transform,
        )

        if cfg.data.class_weights is not None:
            weights = torch.tensor(cfg.data.class_weights, dtype=torch.float).to(
                self.cfg.training.device
            )

        self.loss_fn = nn.NLLLoss(weight=weights)

        self.accuracy_metric = Accuracy(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )

        self.iou_metric = JaccardIndex(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )

        self.test_predictions = []
        self.test_labels = []
        self.test_point_clouds = []

    def forward(self, x):
        return self.network(x)

    def compute_transform_regularization(self, transform_matrix):
        """
        Регуляризация для матрицы трансформации признаков
        Поощряет матрицу быть близкой к ортогональной
        """
        batch_size = transform_matrix.size(0)
        k = transform_matrix.size(1)

        identity = torch.eye(k).unsqueeze(0).repeat(batch_size, 1, 1)
        identity = identity.to(transform_matrix.device)

        matrix_diff = (
            torch.bmm(transform_matrix, transform_matrix.transpose(1, 2)) - identity
        )
        reg_loss = torch.mean(torch.norm(matrix_diff, dim=(1, 2)))

        return reg_loss

    def training_step(self, batch, batch_idx):
        coords, labels = batch
        # coords: (batch_size, 3, num_points)
        # labels: (batch_size, num_points)

        logits, input_trans, feat_trans = self(coords)
        # logits: (batch_size, num_points, num_classes)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.loss_fn(logits, labels)

        # Регуляризация трансформации признаков
        if self.cfg.model.feature_transform and feat_trans is not None:
            reg_loss = self.compute_transform_regularization(feat_trans)
            loss = loss + self.cfg.loss.feature_transform_reg_weight * reg_loss
            self.log("train_reg_loss", reg_loss, prog_bar=False)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy_metric(preds, labels)
        iou = self.iou_metric(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        coords, labels = batch
        logits, _, _ = self(coords)
        # logits: (batch_size, num_points, num_classes)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy_metric(preds, labels)
        iou = self.iou_metric(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        coords, labels = batch
        logits, _, _ = self(coords)
        # logits: (batch_size, num_points, num_classes)

        # Сохраняем исходную форму для визуализации
        original_batch_size = coords.shape[0]
        original_num_points = coords.shape[2]

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy_metric(preds, labels)
        iou = self.iou_metric(preds, labels)

        # Сохраняем образцы для визуализации
        if len(self.test_predictions) < self.cfg.evaluation.visualize_samples:
            # Восстанавливаем предсказания и метки к исходным размерам
            original_preds = (
                preds.view(original_batch_size, original_num_points).cpu().numpy()
            )
            original_labels = (
                labels.view(original_batch_size, original_num_points).cpu().numpy()
            )
            original_coords = coords.cpu().numpy()

            # Добавляем каждый образец отдельно для корректной визуализации
            for i in range(
                min(
                    len(original_preds),
                    self.cfg.evaluation.visualize_samples - len(self.test_predictions),
                )
            ):
                self.test_predictions.append(original_preds[i])
                self.test_labels.append(original_labels[i])
                self.test_point_clouds.append(original_coords[i])

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_iou", iou, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        """Визуализация результатов сегментации"""
        if self.cfg.evaluation.visualize_results and len(self.test_predictions) > 0:
            self.visualize_segmentation_results()

    def visualize_segmentation_results(self):
        """Визуализация результатов сегментации облаков точек"""
        num_samples = min(
            self.cfg.evaluation.visualize_samples, len(self.test_predictions)
        )

        fig = plt.figure(figsize=(15, 4 * num_samples))

        for i in range(num_samples):
            coords = self.test_point_clouds[i]  # (3, N)
            pred = self.test_predictions[i]  # (N,)
            gt = self.test_labels[i]  # (N,)

            # Транспонируем в (N, 3) для построения графиков
            coords = coords.transpose(1, 0)

            # Истинные метки
            ax1 = fig.add_subplot(num_samples, 2, 2 * i + 1, projection="3d")
            scatter1 = ax1.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], c=gt, cmap="tab10", s=1
            )
            ax1.set_title(f"Образец {i + 1}: Истинные метки")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            plt.colorbar(scatter1, ax=ax1, shrink=0.5)

            # Предсказания
            ax2 = fig.add_subplot(num_samples, 2, 2 * i + 2, projection="3d")
            scatter2 = ax2.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], c=pred, cmap="tab10", s=1
            )
            ax2.set_title(f"Образец {i + 1}: Предсказания")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            plt.colorbar(scatter2, ax=ax2, shrink=0.5)

        plt.tight_layout()
        self.logger.experiment.add_figure(
            "segmentation_results", plt.gcf(), self.current_epoch
        )
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.optimizer.lr,
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )

        if self.cfg.training.scheduler.name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.training.scheduler.step_size,
                gamma=self.cfg.training.scheduler.gamma,
            )
            return [optimizer], [scheduler]

        return optimizer
