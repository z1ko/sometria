
from pathlib import Path
import argparse

from omegaconf import OmegaConf, DictConfig

import lightning as L

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sometria.architecture.encoder import EncoderSpec
from sometria.dataset import MotionDataModule
from sometria.downstream.classifier import MotionWindowClassifier
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.models.jepa import MotionJEPA
from sometria.models.masked import MaskedMotionAutoencoder


def build(config: DictConfig) -> tuple[L.LightningModule, L.LightningDataModule]:
    """Turn one config into the objective and the loader it needs.

    ``config.encoder`` is the backbone spec, kept separate from ``config.model`` because
    the backbone is the thing that transfers: pretraining and downstream name the same
    architecture, and downstream then loads weights into it.
    """

    spec = EncoderSpec(**OmegaConf.to_container(config.encoder, resolve=True))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    name = model_config.pop("name")

    if name == "masked":
        return MaskedMotionAutoencoder(spec, **model_config), MotionDataModule(config)

    if name == "jepa":
        return MotionJEPA(spec, **model_config), MotionDataModule(config)

    if name == "classifier":
        checkpoint = model_config.pop("checkpoint", None)
        # How many labels there are is a property of the vocabulary, not of the run --
        # keeping it in the config too is just a second place for it to be wrong.
        _, model_config["num_labels"] = load_label_vocabulary_index(
            config.dataloader.root, config.dataloader.label_set
        )
        model = (
            MotionWindowClassifier(spec, **model_config)
            if checkpoint is None
            else MotionWindowClassifier.from_pretrained(checkpoint, **model_config)
        )
        return model, LabelledMotionDataModule(config)

    raise ValueError(f"Unknown model: {name}")


def train(config: DictConfig, output: Path):
    output = Path(output)

    L.seed_everything(config.training.seed, workers=True)
    model, datamodule = build(config)
    monitor = config.training.get("monitor", "val/loss")

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
                monitor=monitor,
                mode="max" if monitor.endswith("map") else "min",
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
    parser = argparse.ArgumentParser(description="Train a Sometria model.")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_mamp.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/mamp"))
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    train(config, args.output)


if __name__ == "__main__":
    main()
