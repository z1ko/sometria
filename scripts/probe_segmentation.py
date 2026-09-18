#!/usr/bin/env python
"""Fit a BABEL-60 convex *segmentation* probe on a pretrained checkpoint.

The same contract as ``probe_convex_mae.py`` -- ``--checkpoint`` names a run directory or
a checkpoint file, the config supplies everything else, ``metrics.json`` lands in
``--output`` -- with the target resolved per time patch instead of collapsed over the
window. See :mod:`sometria.downstream.segmentation_convex` for the head, and
``config/experiment_segmentation.yaml`` for the data settings, which are not revisited
here: ``label_min_coverage`` 0.5, ``label_types: ["frame"]``, 240-frame windows, 30 time
patches.

Three keys of that config are deliberately ignored. ``model.checkpoint`` pins one dead
path from before the matrix tree existed, and the cell being probed has to arrive on the
command line -- hence ``--checkpoint``, required, exactly as in ``probe_convex_mae.py``.
``model.pool`` and ``model.lr`` configure the AdamW/attentive ``MotionSegmenter``, which is
the optional ceiling run rather than this ranking head; pooling here is ``mean`` over the
DOF axis and there is no learning rate.

Reported: per-patch ``macro_map`` is the ranking number, ``boundary_f1s`` the gate beside
it. Both go in ``metrics.json``, and both are named as ``scripts/moments_baseline.py``
names them, so the dumb baseline run under this same config lands in the same table.

    uv run python scripts/probe_segmentation.py --self-check
    uv run python scripts/probe_segmentation.py \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed1/in_pkd__loss_pkd \\
        --output runs/probe/babel_60_segmentation/amass_clean/medium_100ep/seed1/in_pkd__loss_pkd
"""

import json
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

import lightning as L
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.downstream.classifier_convex import DEFAULT_WEIGHT_DECAYS
from sometria.downstream.segmentation_convex import (
    DEFAULT_TRAIN_PASSES,
    MotionConvexSegmenter,
    check_feature_memory,
)

from probe_baseline_mae import BaselineBackbone, latest_checkpoint, load_pretrained


def run_probe(config, checkpoint: Path, device: str, limit_batches: int | None = None) -> dict:
    # Seeded for the reason probe_convex_mae.py gives: the L-BFGS fit is deterministic
    # given features, but the training split draws a fresh random crop per sample per
    # pass, so an unseeded run fits different data every time.
    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    if not config.dataloader.get("label_patches"):
        raise SystemExit(
            f"{config.dataloader.label_set}: this config has no dataloader.label_patches, "
            "so its target is one multi-hot per window -- use scripts/probe_convex_mae.py "
            "for that. config/experiment_segmentation.yaml is the config this script wants."
        )

    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)

    model = MotionConvexSegmenter(BaselineBackbone(load_pretrained(checkpoint)), num_labels=num_labels)

    datamodule = LabelledMotionDataModule(config)
    datamodule.setup()

    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    if limit_batches:
        # Smoke path only, never used by segmentation_matrix.sh. Materialized rather than
        # islice'd lazily so a second pass re-reads it -- which does mean the extra passes
        # see identical crops, and is why this is not a knob for a real run.
        train_loader = list(islice(train_loader, limit_batches))
        val_loader = list(islice(val_loader, limit_batches))

    # A single `model.weight_decay` still means "skip the sweep, use this one value".
    single_wd = config.model.get("weight_decay")
    weight_decays = [single_wd] if single_wd is not None else config.model.get("weight_decays", DEFAULT_WEIGHT_DECAYS)

    train_passes = config.training.get("train_passes", DEFAULT_TRAIN_PASSES)
    best_wd, best_metrics, all_metrics = model.sweep(
        train_loader,
        val_loader,
        weight_decays=list(weight_decays),
        device=device,
        max_iter=config.training.get("max_iter", 200),
        tol=config.training.get("tol", 1e-7),
        train_passes=train_passes,
    )
    return {
        "weight_decay": best_wd,
        "metrics": best_metrics,
        "sweep": all_metrics,
        # Recorded because it is the one setting that trades score against memory here,
        # and a table mixing 2-pass and 20-pass cells would not be one table.
        "train_passes": train_passes,
        "pool": "mean",
    }


def demo() -> None:
    """Self-check: a grid with a known label written into every time patch must come back.

    Two things go wrong here silently and neither changes a shape. The *fold*: the token
    grid flattens as ``t * D + d``, so reshaping on the wrong axis pools over time and
    hands every patch of a window the same vector. The *alignment*: features and targets
    are concatenated through separate lists, so a flatten in a different order
    desynchronizes rows from labels. Both leave the fit at chance while every shape assert
    passes -- so this asserts on the pooled vectors and on the fitted mAP, not on shapes.
    """

    patches, dofs, width, num_labels = 6, 4, 8, 4   # TP != D, so a swapped fold cannot hide

    class _IdentityBackbone(nn.Module):
        spec = type("Spec", (), {"d_model": width, "num_dofs": dofs})()

        def embed_tokens(self, features: t.Tensor) -> t.Tensor:
            return features

    def _batches(n_batches: int, batch_size: int = 16) -> list[dict]:
        # Label of window b, patch p is (b + p) % num_labels: it varies along both axes, so
        # a transposed flatten desynchronizes rather than coincidentally matching.
        batches = []
        for b in range(n_batches):
            index = (t.arange(batch_size)[:, None] + t.arange(patches)[None, :] + b * batch_size) % num_labels
            labels = t.nn.functional.one_hot(index, num_labels).float()          # (B, TP, L)
            planted = t.nn.functional.one_hot(index, width).float()              # (B, TP, C)
            # Every DOF token of a patch carries that patch's label, so a DOF mean recovers
            # it exactly and a time mean destroys it.
            features = planted[:, :, None, :].expand(batch_size, patches, dofs, width)
            batches.append({"features": features.reshape(batch_size, patches * dofs, width).clone(), "labels": labels})
        return batches

    model = MotionConvexSegmenter(_IdentityBackbone(), num_labels=num_labels)
    train, validation = _batches(8), _batches(4)

    feats, labels = model._extract(validation, "cpu")
    assert feats.shape == (64, patches, width), feats.shape
    assert labels.shape == (64, patches, num_labels), labels.shape

    # Patches 0 and 4 of window 0 carry the same planted label, patches 0 and 1 do not.
    # A fold that pooled time instead of DOFs makes all three equal.
    assert t.allclose(feats[0, 0], feats[0, 4]), "same-label patches pooled to different vectors"
    assert not t.allclose(feats[0, 0], feats[0, 1]), "the DOF pooling folded the grid wrong"

    best_wd, metrics, sweep = model.sweep(train, validation, weight_decays=(1e-2, 1e-6))
    assert set(sweep) == {1e-2, 1e-6}
    assert metrics == sweep[best_wd]
    assert {"macro_map", "micro_map", "boundary_f1s", "boundary_best_f1s"} <= set(metrics), metrics
    # Separable by construction. Anything below this means rows and labels came apart.
    assert metrics["macro_map"] > 0.99, metrics
    # The label changes at every patch boundary and the head can see it, so the gate fires.
    assert metrics["boundary_best_f1s"] > 0.9, metrics

    # The guard has to refuse before extraction rather than let L-BFGS discover it.
    try:
        check_feature_memory(10**12, 256, 60, "cpu")
    except MemoryError as error:
        assert "train_passes" in str(error), error
    else:
        raise AssertionError("check_feature_memory accepted a 1e12-row matrix")

    print("probe_segmentation self-check ok:", {k: round(v, 4) for k, v in metrics.items()})


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="pretrained checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_segmentation.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--limit-batches", type=int, help="smoke runs only: truncate both loaders")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args, overrides = parser.parse_known_args()

    if args.self_check:
        return demo()
    if args.checkpoint is None or args.output is None:
        raise SystemExit("--checkpoint and --output are required (or pass --self-check)")

    checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    result = run_probe(config, checkpoint, args.device, args.limit_batches)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"best weight_decay: {result['weight_decay']:g}")
    for name, value in result["metrics"].items():
        print(f"val/{name}: {value:.4f}")


if __name__ == "__main__":
    main()
