
from pathlib import Path
import argparse

from omegaconf import OmegaConf, DictConfig

import lightning as L
import torch as t

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sometria.dataset import MotionDataModule
from sometria.model import MotionConvAutoencoder

EPOCHS = 200


def lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return t.optim.lr_scheduler.SequentialLR(
        optimizer,
        milestones=[warmup_steps],
        schedulers=[
            t.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, 
                total_iters=warmup_steps
            ),
            t.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps, 
                eta_min=1e-3
            )
        ]
    )

def train(config: DictConfig, output: Path):
    output = Path(output)

    L.seed_everything(config.training.seed, workers=True)
    datamodule = MotionDataModule(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model = MotionConvAutoencoder(**model_config)

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=config.training.epochs,
        default_root_dir=output,
        log_every_n_steps=1,
        limit_train_batches=config.training.get("limit_train_batches", None),
        limit_val_batches=config.training.get("limit_val_batches", None),
        overfit_batches=config.training.get("overfit_batches", 0),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
        ],
        logger=CSVLogger(
            save_dir=output,
        )
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, model

    

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Sometria smoke-test model.")
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/conv_autoencoder_smoke"))
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    train(config, args.output)


if __name__ == "__main__":
    main()
