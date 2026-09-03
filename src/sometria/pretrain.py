
import argparse
import lightning as L
import torch

from sometria.dataset import MotionDataModule
from sometria.models.baseline import MAE
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def pretrain(config: DictConfig, output: Path) -> None:
    L.seed_everything(config.training.seed, workers=True)

    # TF32 on the tensor cores. Ada defaults to "highest", which is plain fp32 matmul --
    # roughly half the throughput for a difference this model cannot see: the loss is an
    # MSE over normalized tokens, not an ill-conditioned solve.
    torch.set_float32_matmul_precision("high")

    # Create model from configuration
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model = MAE(**model_config) # type: ignore

    # Create datamodule from configuration
    datamodule = MotionDataModule(config)

    # Create lightning trainer
    monitor = config.training.get("monitor", "val/loss")
    trainer = L.Trainer(
        accelerator="auto",
        # bf16 is 4.3x, and almost none of it is the matmuls -- TF32 alone was 5%. At
        # 1290 tokens the cost is attention, and nn.TransformerEncoderLayer only reaches
        # the flash kernel in a half precision. Master weights stay fp32, so the EMA and
        # the optimizer are unaffected. Set training.precision=32 to fall back.
        precision=config.training.get("precision", "bf16-mixed"),
        max_epochs=config.training.epochs,
        default_root_dir=output,
        check_val_every_n_epoch=5,
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
                save_last=False,
            ),
            # last.ckpt is a *separate* callback on purpose. Lightning >=2.5 only writes
            # it when top-k saved on the same step, so a monitored callback that owns both
            # freezes last.ckpt at the last epoch the monitor improved. That is wrong for
            # any run whose monitor is not the quality signal -- JEPA drives val/loss to
            # zero by collapsing, so the epochs worth probing are exactly the ones after
            # it stops improving. Unmonitored, this one saves every epoch and last.ckpt is
            # the final weights, as the name says.
            ModelCheckpoint(save_top_k=1, save_last=True, filename="latest"),
        ],
        logger=CSVLogger(
            save_dir=output,
        )
    )

    trainer.fit(model, datamodule=datamodule)


def main() -> None:

    parser = argparse.ArgumentParser(description="Train a Sometria model.")
    parser.add_argument("--config", type=Path, default=Path("config/pretrain_mae.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/mae"))

    # Unknown arguments are OmegaConf dotlist overrides -- `model.decoder_depth=6`,
    # `training.seed=1`. An ablation is then a shell loop over one key, and the config
    # file stays the baseline every run is a delta from.
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides) 
    )

    # The resolved config beside the run, overrides already applied: an ablation's
    # output directory then says what produced it without trusting shell history.
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")
    pretrain(config, args.output) # type: ignore

if __name__ == "__main__":
    main()