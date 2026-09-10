from pathlib import Path
import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from sometria.dataset import MotionDataModule
from sometria.models.baseline import MAE


def build(config: DictConfig) -> tuple[L.LightningModule, L.LightningDataModule]:
    model_config = OmegaConf.to_container(config.model, resolve=True)
    return MAE(**model_config), MotionDataModule(config)  # type: ignore[arg-type]


def train(config: DictConfig, output: Path):
    output = Path(output)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.training.seed, workers=True)

    model, datamodule = build(config)
    monitor = config.training.get("monitor", "val/loss")

    trainer = L.Trainer(
        accelerator="auto",
        precision=config.training.get("precision", "bf16-mixed"),
        max_epochs=config.training.epochs,
        default_root_dir=output,
        check_val_every_n_epoch=config.training.get("check_val_every_n_epoch", 5),
        log_every_n_steps=config.training.get("log_every_n_steps", 1),
        limit_train_batches=config.training.get("limit_train_batches", None),
        limit_val_batches=config.training.get("limit_val_batches", None),
        overfit_batches=config.training.get("overfit_batches", 0),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor=monitor,
                mode="min" if monitor.endswith("loss") else "max",
                save_top_k=1,
                save_last=False,
            ),
            # last.ckpt is a *separate* callback on purpose. Lightning >=2.5 only writes it
            # when top-k saved on the same step, so a monitored callback owning both freezes
            # last.ckpt at the last epoch the monitor improved -- measured, not assumed: with
            # a monitor that stops improving at epoch 0, a single merged callback leaves
            # last.ckpt at epoch 0 while this pair leaves it at the final weights.
            # save_top_k=0 because the top-k file this used to write ("latest.ckpt") was a
            # byte-identical third copy that nothing reads.
            ModelCheckpoint(save_top_k=0, save_last=True),
        ],
        logger=CSVLogger(save_dir=output),
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline MAE.")
    parser.add_argument("--config", type=Path, nargs="+", default=[Path("config/pretrain_mae.yaml")])
    parser.add_argument("--output", type=Path, default=Path("runs/mae"))
    args, overrides = parser.parse_known_args()

    config = OmegaConf.merge(*(OmegaConf.load(path) for path in args.config), OmegaConf.from_dotlist(overrides))
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")
    train(config, args.output) # type: ignore


if __name__ == "__main__":
    main()
