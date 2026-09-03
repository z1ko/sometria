#!/usr/bin/env python
"""Train a BABEL60 probe on a baseline.py MAE checkpoint."""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch as t
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from sometria.downstream.classifier import MotionLinearClassifier
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.models.baseline import MAE


class BaselineBackbone(nn.Module):
    """Adapter: baseline.MAE encoder shaped like downstream classifier expects."""

    def __init__(self, mae: MAE) -> None:
        super().__init__()
        self.mae = mae
        self.spec = SimpleNamespace(d_model=mae.hparams.dim, num_dofs=mae.hparams.num_dofs)

    def embed_tokens(self, features: t.Tensor) -> t.Tensor:
        return self.mae.encode(features)


def latest_checkpoint(run: Path) -> Path:
    versions = sorted(
        run.glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.removeprefix("version_")),
    )
    for version in reversed(versions):
        checkpoint_dir = version / "checkpoints"
        best = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
        if best:
            return best[-1]
        last = checkpoint_dir / "last.ckpt"
        if last.exists():
            return last
    raise FileNotFoundError(f"no checkpoint under {run}")


def train_probe(config, checkpoint: Path, output: Path):
    t.set_float32_matmul_precision("high")
    L.seed_everything(config.training.seed, workers=True)

    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)
    mae = MAE.load_from_checkpoint(checkpoint, map_location="cpu")
    model = MotionLinearClassifier(
        BaselineBackbone(mae),
        num_labels=num_labels,
        pool=config.model.get("pool", "attentive_factorized"),
        lr=config.model.get("lr", 1e-3),
        min_lr_frac=config.model.get("min_lr_frac", 0.5),
        warmup=config.model.get("warmup", 0.03),
    )
    datamodule = LabelledMotionDataModule(config)
    monitor = config.training.get("monitor", "val/macro_map")

    trainer = L.Trainer(
        accelerator="auto",
        precision=config.training.get("precision", "bf16-mixed"),
        max_epochs=config.training.epochs,
        default_root_dir=output,
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
            ModelCheckpoint(save_top_k=1, save_last=True, filename="latest"),
        ],
        logger=CSVLogger(save_dir=output),
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, model


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="MAE checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args, overrides = parser.parse_known_args()

    checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")
    train_probe(config, checkpoint, args.output)


if __name__ == "__main__":
    main()
