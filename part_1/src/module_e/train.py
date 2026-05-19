import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_loader import PointCloudSegmentationDataModule
from network import PointNetSegmentationModule


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Конфигурация:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    datamodule = PointCloudSegmentationDataModule(
        root=cfg.data.root,
        num_points=cfg.data.num_points,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        val_sz=cfg.data.val_sz,
        test_sz=cfg.data.test_sz,
    )

    model = PointNetSegmentationModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="pointnet-seg-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        patience=cfg.training.early_stopping_patience,
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir, name="pointnet_segmentation", version=None
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.device,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print("\nОбучение завершено!")
    print(f"Лучшая модель сохранена в: {checkpoint_callback.best_model_path}")
    print(f"Логи TensorBoard в: {logger.log_dir}")


if __name__ == "__main__":
    main()
