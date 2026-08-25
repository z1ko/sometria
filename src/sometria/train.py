
from pathlib import Path
import argparse

from omegaconf import OmegaConf, DictConfig

import lightning as L

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder, load_encoder
from sometria.dataset import MotionDataModule
from sometria.downstream.classifier import MotionLinearClassifier
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.downstream.segmentation import MotionSegmenter
from sometria.masking import MaskSpec
from sometria.models.jepa import MotionJEPA
from sometria.models.mae import MaskedAutoencoder
from sometria.models.mamp import MaskedMotionPredictor

# Every pretext objective takes an EncoderSpec and a MaskSpec, and reads the same loader.
OBJECTIVES = {
    "mae": MaskedAutoencoder,
    "mamp": MaskedMotionPredictor,
    "jepa": MotionJEPA,
}


def build(config: DictConfig) -> tuple[L.LightningModule, L.LightningDataModule]:
    """Turn one config into the objective and the loader it needs.

    ``config.encoder`` is the backbone spec, kept separate from ``config.model`` because
    the backbone is the thing that transfers: pretraining and downstream name the same
    architecture, and downstream then loads weights into it. ``config.masking`` is the
    same story for what gets held out: every pretext objective masks, and none of them
    owns the policy. Omit the block and the objective's own default applies.
    """

    spec = EncoderSpec(**OmegaConf.to_container(config.encoder, resolve=True))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    name = model_config.pop("name")

    if name in OBJECTIVES:
        masking = config.get("masking")
        mask = None if masking is None else MaskSpec(**OmegaConf.to_container(masking, resolve=True))
        return OBJECTIVES[name](spec, mask, **model_config), MotionDataModule(config)

    if name in ("classifier", "segmenter"):
        # The mapping from a probe to what it measures is this path, and nothing else:
        # the pretraining checkpoint carries the spec, so the `encoder:` block below is
        # only read for the random-initialization control.
        checkpoint = model_config.pop("checkpoint", None)
        encoder = model_config.pop("encoder", "teacher")
        backbone = (
            MotionTransformerEncoder(spec)
            if checkpoint is None
            else load_encoder(checkpoint, encoder)
        )
        # How many labels there are is a property of the vocabulary, not of the run --
        # keeping it in the config too is just a second place for it to be wrong.
        _, model_config["num_labels"] = load_label_vocabulary_index(
            config.dataloader.root, config.dataloader.label_set
        )
        head = MotionSegmenter if name == "segmenter" else MotionLinearClassifier
        return head(backbone, **model_config), LabelledMotionDataModule(config)

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
                # Every metric here is better-when-higher except the losses; keying on
                # "map" silently made mode="min" for f1 monitors and saved the worst epoch.
                mode="min" if monitor.endswith("loss") else "max",
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
    # Unknown arguments are OmegaConf dotlist overrides -- `model.decoder_depth=6`,
    # `training.seed=1`. An ablation is then a shell loop over one key, and the config
    # file stays the baseline every run is a delta from.
    args, overrides = parser.parse_known_args()

    config = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)
    )
    # The resolved config beside the run, overrides already applied: an ablation's
    # output directory then says what produced it without trusting shell history.
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    train(config, args.output)


if __name__ == "__main__":
    main()
