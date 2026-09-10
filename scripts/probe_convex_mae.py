#!/usr/bin/env python
"""Evaluate a BABEL60 convex probe (mean-pool + L-BFGS logistic regression) on a MAE checkpoint.

Same config/checkpoint contract as ``probe_baseline_mae.py``, but the head is fit to
convergence with :class:`~sometria.downstream.classifier_convex.MotionConvexClassifier`
instead of trained epoch-by-epoch with AdamW -- see that module's docstring for why.
Only ``model.pool: mean`` / ``mean_max`` are valid here.

Sweeps ``weight_decay`` by default (``model.weight_decays``, a list; ``model.weight_decay``
still works as a single value for backward compat) and reports the best. The backbone
forward pass is the expensive part and does not depend on ``weight_decay``, so this costs
one feature extraction pass plus one cheap L-BFGS solve per candidate, not one run per
candidate.

Also re-draws the training crop ``training.train_passes`` times before fitting (default
``DEFAULT_TRAIN_PASSES``): the training split is one random window per sample per epoch,
so a single extraction pass is one crop per sample, while the AdamW probe sees a new one
every one of its 100 epochs. Fewer passes trades this off for a cheaper/faster run.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from sometria.downstream.classifier_convex import (
    DEFAULT_TRAIN_PASSES,
    DEFAULT_WEIGHT_DECAYS,
    MotionConvexClassifier,
)
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.models.baseline import MAE


class BaselineBackbone(nn.Module):
    """Adapter: baseline.MAE encoder shaped like the downstream classifier expects."""

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


def run_probe(config, checkpoint: Path, device: str) -> dict:
    # The L-BFGS fit is deterministic given features, but the features are not: the
    # training split draws one random crop per sample per pass, so an unseeded run refits
    # different data every time. Measured on `in_pk__loss_pk`, that alone moved macro mAP
    # by ~0.001 -- the same size as the corpus-to-corpus deltas the matrix is used to
    # compare, which is why this is seeded rather than left to the global RNG.
    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")
    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)

    mae = MAE.load_from_checkpoint(checkpoint, map_location="cpu")
    model = MotionConvexClassifier(
        BaselineBackbone(mae),
        num_labels=num_labels,
        pool=config.model.get("pool", "mean"),
    )

    datamodule = LabelledMotionDataModule(config)
    datamodule.setup()

    # A single `model.weight_decay` still means "skip the sweep, use this one value".
    single_wd = config.model.get("weight_decay")
    weight_decays = [single_wd] if single_wd is not None else config.model.get("weight_decays", DEFAULT_WEIGHT_DECAYS)

    best_wd, best_metrics, all_metrics = model.sweep(
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        weight_decays=list(weight_decays),
        device=device,
        max_iter=config.training.get("max_iter", 200),
        tol=config.training.get("tol", 1e-7),
        train_passes=config.training.get("train_passes", DEFAULT_TRAIN_PASSES),
    )
    return {"weight_decay": best_wd, "metrics": best_metrics, "sweep": all_metrics}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="MAE checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    args, overrides = parser.parse_known_args()

    checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    result = run_probe(config, checkpoint, args.device)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"best weight_decay: {result['weight_decay']:g}")
    for name, value in result["metrics"].items():
        print(f"val/{name}: {value:.4f}")


if __name__ == "__main__":
    main()
