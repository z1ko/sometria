
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

import lightning as L
import torch as t
import wandb
import tqdm

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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

    L.seed_everything(config.training.seed, workers=True)
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=config.training.epochs,
        default_root_dir=output,
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),

        ],
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir=output,
        )
    )

    

def main(cfg: dict):

    