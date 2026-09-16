#!/usr/bin/env python
"""Fine-tune a pretrained checkpoint on a labelled benchmark, backbone included.

    uv run python scripts/finetune_baseline_mae.py \
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed42/in_pk__loss_pk \
        --config config/experiment_finetune.yaml \
        --output runs/finetune/babel_60/amass_clean/medium_100ep/seed42/in_pk__loss_pk

Sibling of ``scripts/probe_baseline_mae.py`` and deliberately a thin one: same checkpoint
loading, same adapter, same loader, same metrics. The head is
:class:`~sometria.downstream.finetune.MotionFinetuneClassifier` instead of the probe's, and
that class *is* the difference -- backbone in train mode, gradient reaching it, and in the
optimizer under layer-wise learning-rate decay.

Why this exists rather than ``config/experiment_finetune.yaml`` through ``sometria.train``:
that path builds its backbone with
:func:`~sometria.architecture.encoder.load_encoder`, which rebuilds an ``EncoderSpec`` from
``hparams["spec"]``. The objectives in ``sometria.models.baseline``/``.simmim``/``.jepa2``
save their constructor arguments flat and have no ``spec`` key, so every checkpoint the
pretraining matrix has ever written is unreadable by it. The probes already solved this with
``load_pretrained``, which dispatches on state_dict prefixes; this reuses that rather than
teaching a second loader the same trick.

A fine-tune is the protocol the masked objectives were designed for. SimMIM's own paper
(arXiv:2111.09886 S3.5) declines to be judged on linear probing at all -- 56.7 against 83.8
fine-tuned -- so a frozen probe that cannot separate these four objectives is weak evidence
that they are the same, and this is the measurement that would be strong evidence either way.

``metrics.json`` is written in the shape ``scripts/probe_convex_mae.py`` uses, so a results
folder can read the two side by side, and so ``scripts/finetune_matrix.sh`` has a marker to
skip a finished cell on.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import torch as t
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from probe_baseline_mae import BaselineBackbone, latest_checkpoint, load_pretrained
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.finetune import MotionFinetuneClassifier
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.models.baseline import MAE
from sometria.models.jepa2 import JEPA
from sometria.models.simmim import SimMIM

#: The one submodule each objective's ``encode`` reads. Everything else it carries --
#: decoder, prediction head, JEPA's student and predictor -- was scaffolding for training
#: those weights and is dead here.
ENCODER_OF = {MAE: "encoder", SimMIM: "encoder", JEPA: "encoder_teacher"}

#: Reported for every run. The probe's names exactly, because the point of the exercise is
#: reading a fine-tune against the probe of the same checkpoint.
METRICS = ("macro_map", "micro_map", "macro_f1s", "top_1_rec", "top_3_rec", "top_5_rec")


def drop_scaffolding(objective: L.LightningModule) -> L.LightningModule:
    """Keep only the submodule ``encode`` reads, so a finetune does not carry the rest.

    A probe can leave the scaffolding attached: the backbone is frozen, so a decoder it never
    calls costs memory and nothing else. A finetune cannot. ``MotionFinetuneClassifier``
    un-freezes everything the adapter wraps, which puts the decoder -- and, for JEPA, a whole
    second encoder and a predictor -- into the optimizer holding AdamW state for a gradient
    that never arrives.

    Layer-wise learning-rate decay is the part that makes this load-bearing rather than
    tidy: it reads the ladder's depth off parameter names, and a decoder's ``blocks.layers.*``
    are indistinguishable from an encoder's. Left attached, they would shift every rung.

    The objective is *not* rebuilt around the kept submodule, because ``encode`` also applies
    ``channels_input``, and a cell probed at ``in_p`` would silently be fed all five channels
    without it.
    """

    keep = ENCODER_OF[type(objective)]
    for name, _ in list(objective.named_children()):
        if name != keep:
            setattr(objective, name, None)
    return objective


def finetune(config, checkpoint: Path, output: Path) -> dict:
    t.set_float32_matmul_precision("high")
    L.seed_everything(config.training.seed, workers=True)

    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)

    model = MotionFinetuneClassifier(
        BaselineBackbone(drop_scaffolding(load_pretrained(checkpoint))),
        num_labels=num_labels,
        pool=config.model.get("pool", "attentive"),
        lr=config.model.get("lr", 1e-3),
        backbone_lr=config.model.get("backbone_lr", 1e-4),
        layer_decay=config.model.get("layer_decay"),
        weight_decay=config.model.get("weight_decay", 0.05),
        dropout=config.model.get("dropout"),
        min_lr_frac=config.model.get("min_lr_frac", 0.01),
        warmup=config.model.get("warmup", 0.05),
    )

    datamodule = LabelledMotionDataModule(config)
    monitor = config.training.get("monitor", "val/macro_map")

    best = ModelCheckpoint(
        monitor=monitor,
        mode="min" if monitor.endswith("loss") else "max",
        save_top_k=1,
        save_last=False,
    )
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
            best,
            ModelCheckpoint(save_top_k=1, save_last=True, filename="latest"),
        ],
        logger=CSVLogger(save_dir=output),
    )
    trainer.fit(model, datamodule=datamodule)

    # One extra validation pass against the *monitored* checkpoint rather than reading
    # `callback_metrics` off the end of training. Those hold the last epoch, and the last
    # epoch of a fine-tune is routinely not its best -- reporting it would quietly report a
    # different model than the one this run selected and saved.
    scores = trainer.validate(model, datamodule=datamodule, ckpt_path=best.best_model_path)[0]

    return {
        "metrics": {name: scores[f"val/{name}"] for name in METRICS if f"val/{name}" in scores},
        "monitor": monitor,
        "best_checkpoint": str(Path(best.best_model_path).relative_to(output))
        if best.best_model_path
        else None,
        "epochs": config.training.epochs,
        "lr": config.model.get("lr", 1e-3),
        "layer_decay": config.model.get("layer_decay"),
        "backbone_lr": config.model.get("backbone_lr", 1e-4),
        "pool": config.model.get("pool", "attentive"),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_finetune.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args, overrides = parser.parse_known_args()

    checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    result = finetune(config, checkpoint, args.output)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))

    for name, value in result["metrics"].items():
        print(f"val/{name}: {value:.4f}")


if __name__ == "__main__":
    main()
