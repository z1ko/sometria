#!/usr/bin/env python
"""Per-label average precision for finished convex probes, one column per backbone.

    uv run python scripts/label_breakdown.py
    uv run python scripts/label_breakdown.py --runs runs/probe/babel_60_convex/amass_clean/*/seed42/in_pk*

A probe's `metrics.json` reports macro mAP, which is the mean of 60 per-label APs. That mean
answers "how good is this representation" and cannot answer "at what", and on a vocabulary
whose head-to-tail support ratio is 44.6:1 those are very different questions.

There is nothing to read back: the convex probe writes its metrics and discards its head, so
the per-label numbers have to be recomputed. The refit is cheap relative to a probe sweep --
features are extracted once and the head is fit once at the `weight_decay` the run already
selected, rather than once per candidate -- but it is not free, because extraction runs
`train_passes` times over the training split.

The backbone is taken from the *pretraining* tree at the matching path, not from the
`model.checkpoint` a probe recorded. Those paths were written before the runs/ restructure
and name flat directories that are now under `runs/archive/`; they are a record of what was
probed, not a path that opens today.

Two checks, and they are not the same question. The per-label APs are averaged over the
labels that occur, exactly as `WindowMeanAveragePrecision` does, and that mean must equal the
`macro_map` the sweep just computed -- same head, same data, so any gap is an arithmetic
error here and the script stops. It is then *compared*, without asserting, against the
`macro_map` stored in `metrics.json`. That second number was produced by whatever the code
looked like on the day the run finished, and a probe from before
`eeabbf6 Refactor feature extraction and scoring` reproduces about 0.0013 high. Printed
rather than enforced, because it says something about the run's age, not about this table.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
import polars as pl
import torch as t
from omegaconf import OmegaConf
from torchmetrics.classification import MultilabelAveragePrecision

from probe_baseline_mae import BaselineBackbone, latest_checkpoint, load_pretrained
from sometria.downstream.classifier_convex import (
    DEFAULT_TRAIN_PASSES,
    MotionConvexClassifier,
    extract_features,
)
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index

def pretrain_dir(probe_run: Path) -> Path:
    """The backbone a probe run scored, at its current location.

    `runs/probe/<benchmark>/<corpus>/<arch>/<seed>/<cell>`
      -> `runs/pretrain/<corpus>/<arch>/<seed>/<cell>`

    Derived from the path rather than read from the probe's config, because the two trees are
    kept in step by the matrix scripts and the recorded checkpoint paths are not.
    """

    parts = probe_run.parts
    root = parts.index("probe")
    # Everything after the benchmark segment addresses the same cell in either tree.
    return Path(*parts[:root], "pretrain", *parts[root + 2 :])


def per_label(run: Path, device: str) -> tuple[pl.DataFrame, float, float]:
    """``(frame, reported macro mAP, refit macro mAP)`` for one finished probe run."""

    config = OmegaConf.load(run / "config.yaml")
    reported = json.loads((run / "metrics.json").read_text())

    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    names, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)
    model = MotionConvexClassifier(
        BaselineBackbone(load_pretrained(latest_checkpoint(pretrain_dir(run)))),
        num_labels=num_labels,
        pool=config.model.get("pool", "mean"),
    )
    model.to(device)

    datamodule = LabelledMotionDataModule(config)
    datamodule.setup()

    # The run's own sweep, not a single fit at the recorded weight decay. L-BFGS is
    # deterministic given features, but `sweep` calls `head.reset_parameters()` before each
    # candidate, so the winning fit starts from a point that depends on how many candidates
    # preceded it. Refitting once from the constructor's init lands on a different corner of
    # the same optimum -- 0.0012 macro mAP away on `in_pk__loss_pk`, which is the size of the
    # corpus-to-corpus deltas this matrix exists to compare. Replaying the sweep costs six
    # cheap fits on features that were extracted once anyway.
    # One candidate: the weight decay the run selected. `sweep` is still the call, because it
    # is what produced the original head and it resets the head before fitting -- refitting
    # from the constructor's init instead lands 0.0013 macro mAP away, the size of the deltas
    # this matrix exists to compare.
    #
    # Re-deriving the selection would be a different experiment, and an unstable one: under
    # current code JEPA's sweep picks 1e-6 where its run recorded 3e-6, because adjacent
    # candidates sit within the drift. Pinning keeps this a breakdown of a head that was
    # reported, not a fresh probe wearing its name.
    _, best_metrics, _ = model.sweep(
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        weight_decays=[reported["weight_decay"]],
        device=device,
        max_iter=config.training.get("max_iter", 200),
        tol=config.training.get("tol", 1e-7),
        train_passes=config.training.get("train_passes", DEFAULT_TRAIN_PASSES),
        by_sample=config.training.get("pool_val_by_take", False),
        f1_subsets=[list(subset) for subset in config.training.get("f1_label_subsets", [])],
    )

    # `sweep` leaves the winning head in place but keeps no features, so the validation split
    # is read once more. Its crops are deterministic, so this is the same data it scored.
    val_feats, val_labels, _ = extract_features(
        model.backbone, model.pooler, model.norm, datamodule.val_dataloader(), device
    )
    with t.no_grad():
        logits = model.head(val_feats.to(device)).cpu()

    metric = MultilabelAveragePrecision(num_labels=num_labels, average=None)
    ap = metric(logits, val_labels.int()).numpy()
    positives = val_labels.sum(dim=0).int().numpy()

    # `WindowMeanAveragePrecision` averages over the labels that occur, so reproducing its
    # number means using the same subset.
    present = positives > 0
    refit = float(ap[present].mean())
    assert abs(refit - best_metrics["macro_map"]) < 1e-6, (
        f"per-label APs average to {refit:.6f} where the sweep that produced this head "
        f"reported {best_metrics['macro_map']:.6f}; the breakdown does not describe the head."
    )

    return (
        pl.DataFrame(
            {
                "label": names["label"].to_list(),
                "positives": positives,
                "ap": ap,
                # A label's AP floor is its prevalence, so a rare label scores low partly by
                # construction. Lift is what compares across the tail.
                "prevalence": positives / len(val_labels),
            }
        ).with_columns((pl.col("ap") / pl.col("prevalence")).alias("lift")),
        float(reported["metrics"]["macro_map"]),
        refit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", default=[
        Path("runs/probe/babel_60_convex/amass_clean/medium_100ep/seed42/in_pk__loss_pk"),
        Path("runs/probe/babel_60_convex/amass_clean/medium_100ep_jepa/seed42/in_pk"),
        Path("runs/probe/babel_60_convex/amass_clean/medium_100ep_simmim/seed42/in_pk__loss_pk"),
    ])
    parser.add_argument("--output", type=Path, default=Path("results/probe/babel_60_convex/labels.csv"))
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    merged = None
    for run in arguments.runs:
        # The architecture segment names the objective, and is the only part of the path that
        # differs between the runs being compared.
        arch = run.parts[run.parts.index("probe") + 3]
        frame, reported, refit = per_label(run, arguments.device)
        age = datetime.fromtimestamp((run / "metrics.json").stat().st_mtime).date()
        drift = refit - reported
        print(f"{arch:22s} stored {reported:.4f} ({age})  refit {refit:.4f}  drift {drift:+.4f}")

        columns = frame.select("label", "positives", pl.col("ap").alias(arch))
        merged = columns if merged is None else merged.join(columns.drop("positives"), on="label")

    arms = [c for c in merged.columns if c not in ("label", "positives")]
    merged = merged.sort("positives", descending=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(arguments.output)
    print(f"\n{merged.height} labels -> {arguments.output}\n")

    pl.Config.set_tbl_rows(-1)
    print(merged.with_columns([pl.col(a).round(4) for a in arms]).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
