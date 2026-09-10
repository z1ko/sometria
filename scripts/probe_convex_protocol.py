#!/usr/bin/env python
"""Score one frozen checkpoint against many split sets, extracting its features exactly once.

``probe_convex_mae.py`` answers "how does this checkpoint do on this split", and re-extracts
the backbone's features every time it is called. That is the right shape for a benchmark with
one split and the wrong one for cross-validation: CARE-PD's leave-one-subject-out protocol is
110 folds, and at ~80 s of feature extraction each it costs a day per checkpoint to learn
almost nothing new -- because the backbone is frozen, so all 110 folds see *the same features*
and differ only in which rows they keep.

So: build one view over every take the requested split sets mention, extract from it once, and
reduce each fold to an L-BFGS solve on a row subset. LOSO drops from hours to the cost of a
single run plus some cheap solves.

    uv run python scripts/probe_convex_protocol.py \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed42/in_pkd__loss_pkd \\
        --config config/experiment_probe_carepd.yaml \\
        --split-sets 'carepd_*_loso_*' \\
        --tune-on carepd_BMCLab_6fold_1 --select-on macro_f1_012 \\
        --output runs/probe/carepd_updrs_convex/loso/amass_clean/medium_100ep

``--split-sets`` takes names or ``fnmatch`` patterns, resolved against what is actually in
``splits.parquet``. Results are grouped by stripping the trailing fold index, so a run over
110 LOSO folds reports one number per cohort rather than 110 numbers nobody can read.

Three things about that grouped number are worth knowing before reading one.

``pooled`` is the headline, not ``per_fold.mean``. A LOSO fold holds out one patient, who
often walks at a single severity, so per-fold metrics are dominated by artefacts of that --
mean average precision is 1.0 whenever the held-out subject carries one class, because the
only class present is positive everywhere. Pooling every fold's held-out predictions and
scoring them once is what leave-one-out cross-validation normally means. The per-fold mean
is kept beside it because its *spread* is real between-subject variability even where its
level is not meaningful.

``--select-on`` is which metric ``weight_decay`` is chosen by, and defaults to ``macro_map``
for consistency with ``probe_convex_mae.py``. On single-subject folds that default is the
degenerate metric above; use ``macro_f1_012``. It is not a cosmetic choice -- on
``carepd_lodo_BMCLab`` selecting by ``macro_map`` picked a weight decay scoring macro F1
0.089, against 0.182 for one tuned properly.

``--tune-on`` is not optional in spirit. Sweeping ``weight_decay`` per fold and keeping the
best selects a hyperparameter on the evaluation set, so the reported number is optimistic;
the paper tunes once on BMCLab and reuses that setting everywhere. Left unset, this script
sweeps per fold and records ``weight_decay_tuned_per_fold: true`` so the number is never
mistaken for a clean one.
"""

import json
from argparse import ArgumentParser
from fnmatch import fnmatch
from pathlib import Path
from statistics import mean, stdev

import lightning as L
import polars as pl
import torch as t
from omegaconf import OmegaConf

from sometria.catalog import load_splits
from sometria.downstream.classifier_convex import (
    DEFAULT_TRAIN_PASSES,
    DEFAULT_WEIGHT_DECAYS,
    MotionConvexClassifier,
    extract_features,
    pool_by_sample,
    score_predictions,
    vote_macro_f1,
)
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.models.baseline import MAE

from probe_convex_mae import BaselineBackbone, latest_checkpoint


def resolve_split_sets(root: str | Path, patterns: list[str]) -> list[str]:
    """Expand names and glob patterns against the split sets that exist.

    Patterns rather than an explicit list because the interesting selections are families --
    every LOSO fold of every cohort is 110 names nobody should type. Resolved against the
    table so a typo is an error here instead of an empty result three minutes into extraction.
    """

    available = sorted(load_splits(root)["split_set"].unique().to_list())
    selected = [name for name in available if any(fnmatch(name, p) for p in patterns)]
    if not selected:
        raise SystemExit(
            f"no split set matches {patterns}. {len(available)} available, e.g. "
            f"{available[:5]}"
        )
    return selected


def fold_group(split_set: str) -> str:
    """The family a split set belongs to: its name with any trailing fold index removed.

    ``carepd_BMCLab_loso_7`` and ``carepd_BMCLab_loso_8`` are two folds of one experiment and
    are averaged together; ``carepd_lodo_BMCLab`` is its own experiment and stays alone.
    """

    head, _, tail = split_set.rpartition("_")
    return head if tail.isdigit() else split_set


def membership(splits: pl.DataFrame, split_set: str) -> tuple[set[str], set[str]]:
    """The ``(train, eval)`` sample ids of one split set."""

    rows = splits.filter(pl.col("split_set") == split_set)
    return (
        set(rows.filter(pl.col("split") == "train")["sample_id"]),
        set(rows.filter(pl.col("split") == "eval")["sample_id"]),
    )


def rows_in(sample_ids: list[str], keep: set[str]) -> t.Tensor:
    """Indices of the extracted rows whose sample is in ``keep``."""

    return t.tensor([i for i, sample_id in enumerate(sample_ids) if sample_id in keep])


def union_config(config):
    """A copy of ``config`` whose train and val views are *every* take, not one split.

    ``MotionViewSpec`` with no ``split_set`` means "the whole labelled corpus for these
    sources", which is exactly the union any set of CARE-PD folds can draw from. Reusing the
    datamodule this way keeps normalization, windowing, the broken filter and the label
    vocabulary on one code path with the single-split probe, rather than a second copy that
    drifts.
    """

    merged = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    for side in ("train", "val"):
        merged.dataloader[side].split_set = None
        merged.dataloader[side].split = None
    return merged


def pooled_metrics(folds: list[dict], f1_subsets) -> dict[str, float]:
    """Score every fold's held-out predictions together, as one set.

    This is the number to quote for leave-one-subject-out, and it is not the mean of the
    per-fold numbers. A LOSO fold holds out *one patient*, who often walks at a single
    severity -- so most classes have no support in that fold, macro F1 is divided by classes
    that could never score, and mean average precision is 1.0 by construction because the one
    class present is positive everywhere. Averaging 14 such numbers averages 14 artefacts.

    Pooling instead scores each held-out take exactly once against the full label set, which
    is what leave-one-out cross-validation normally means. The per-fold mean is still
    reported beside it: its spread is real between-subject variability even where its level
    is not meaningful.
    """

    probabilities = t.cat([fold["probabilities"] for fold in folds])
    labels = t.cat([fold["labels"] for fold in folds])
    sample_ids = [sample_id for fold in folds for sample_id in fold["ids"]]

    scores, targets = pool_by_sample(probabilities, labels, sample_ids)
    return score_predictions(scores, targets.int()) | vote_macro_f1(
        probabilities, labels, sample_ids, f1_subsets
    )


def summarize(results: dict[str, dict], predictions: dict[str, dict], f1_subsets) -> dict[str, dict]:
    """Per fold family: metrics pooled over folds, and the mean/sd of the per-fold numbers.

    ``sd`` is the spread *between folds*, which for LOSO is between-subject variability. See
    :func:`pooled_metrics` for why ``pooled`` rather than ``per_fold_mean`` is the number to
    report.
    """

    grouped: dict[str, list[str]] = {}
    for split_set in results:
        grouped.setdefault(fold_group(split_set), []).append(split_set)

    summary = {}
    for group, names in grouped.items():
        folds = [results[name]["metrics"] for name in names]
        shared = set.intersection(*(set(f) for f in folds))
        summary[group] = {
            "folds": len(folds),
            "pooled": pooled_metrics([predictions[name] for name in names], f1_subsets),
            "per_fold": {
                name: {
                    "mean": mean(values),
                    "sd": stdev(values) if len(values) > 1 else None,
                }
                for name in sorted(shared)
                for values in [[f[name] for f in folds]]
            },
        }
    return summary


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="MAE checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_probe_carepd.yaml"))
    parser.add_argument("--split-sets", nargs="+", required=True, help="names or fnmatch patterns")
    parser.add_argument("--tune-on", type=str, default=None, help="split set to pick weight_decay on")
    parser.add_argument(
        "--select-on",
        type=str,
        default="macro_map",
        help="metric weight_decay is chosen by; macro_map is degenerate on single-subject folds",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    args, overrides = parser.parse_known_args()

    checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    root = config.dataloader.root
    splits = load_splits(root)
    split_sets = resolve_split_sets(root, args.split_sets)
    if args.tune_on and args.tune_on not in set(splits["split_set"]):
        raise SystemExit(f"--tune-on {args.tune_on!r} is not a split set in {root}")

    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    _, num_labels = load_label_vocabulary_index(root, config.dataloader.label_set)
    mae = MAE.load_from_checkpoint(checkpoint, map_location="cpu")
    model = MotionConvexClassifier(
        BaselineBackbone(mae), num_labels=num_labels, pool=config.model.get("pool", "mean")
    )
    model.to(args.device)

    # One view over every take any requested fold can draw from, extracted twice: with random
    # crops for fitting and tiled for scoring, exactly as the single-split probe does.
    datamodule = LabelledMotionDataModule(union_config(config))
    datamodule.setup()
    passes = config.training.get("train_passes", DEFAULT_TRAIN_PASSES)
    print(f"extracting {len(datamodule.train_dataset.motions.samples):,} takes "
          f"({passes} train passes) for {len(split_sets)} split sets")

    train_feats, train_labels, train_ids = extract_features(
        model.backbone, model.pooler, model.norm, datamodule.train_dataloader(), args.device, passes
    )
    val_feats, val_labels, val_ids = extract_features(
        model.backbone, model.pooler, model.norm, datamodule.val_dataloader(), args.device
    )
    train_feats, train_labels = train_feats.to(args.device), train_labels.to(args.device)
    print(f"features: {tuple(train_feats.shape)} train windows, {tuple(val_feats.shape)} val windows")

    by_sample = config.training.get("pool_val_by_take", False)
    f1_subsets = [list(subset) for subset in config.training.get("f1_label_subsets", [])]
    max_iter = config.training.get("max_iter", 200)
    tol = config.training.get("tol", 1e-7)
    single_wd = config.model.get("weight_decay")
    candidates = (
        [single_wd] if single_wd is not None
        else list(config.model.get("weight_decays", DEFAULT_WEIGHT_DECAYS))
    )

    def run(split_set: str, weight_decays: list[float]) -> tuple[dict, dict]:
        """Fit and score one split set over ``weight_decays``, keeping the best.

        Also returns the winning head's per-window predictions, so a family of folds can be
        rescored as one set afterwards -- see :func:`pooled_metrics`.
        """

        train_rows = rows_in(train_ids, membership(splits, split_set)[0])
        val_rows = rows_in(val_ids, membership(splits, split_set)[1])
        if train_rows.numel() == 0 or val_rows.numel() == 0:
            raise SystemExit(
                f"{split_set} resolves to {train_rows.numel()} train and {val_rows.numel()} "
                "eval windows; check source_datasets and min_frames in the config"
            )

        feats, labels = train_feats[train_rows], train_labels[train_rows]
        fold_val_feats, fold_val_labels = val_feats[val_rows], val_labels[val_rows]
        fold_val_ids = [val_ids[i] for i in val_rows.tolist()]

        best, best_predictions = None, None
        for weight_decay in weight_decays:
            model.fit_features(feats, labels, max_iter, tol, weight_decay)
            with t.no_grad():
                probabilities = model.head(fold_val_feats.to(args.device)).cpu().sigmoid()
            metrics = model.score_features(
                fold_val_feats, fold_val_labels, fold_val_ids, args.device, by_sample, f1_subsets
            )
            if metrics.get(args.select_on) is None:
                raise SystemExit(
                    f"--select-on {args.select_on!r} is not among the metrics "
                    f"{sorted(metrics)}"
                )
            if best is None or metrics[args.select_on] > best["metrics"][args.select_on]:
                best = {
                    "weight_decay": weight_decay,
                    "metrics": metrics,
                    "train_takes": len({train_ids[i] for i in train_rows.tolist()}),
                    "eval_takes": len(set(fold_val_ids)),
                }
                best_predictions = {
                    "probabilities": probabilities,
                    "labels": fold_val_labels,
                    "ids": fold_val_ids,
                }
        assert best is not None and best_predictions is not None  # never empty
        return best, best_predictions

    tuned = None
    if args.tune_on:
        tuned = run(args.tune_on, candidates)[0]["weight_decay"]
        print(f"weight_decay {tuned:g}, tuned once on {args.tune_on} by {args.select_on}")

    results, predictions = {}, {}
    for i, split_set in enumerate(split_sets, start=1):
        results[split_set], predictions[split_set] = run(
            split_set, [tuned] if tuned is not None else candidates
        )
        reported = " ".join(
            f"{k}={v:.4f}" for k, v in results[split_set]["metrics"].items()
            if k in ("macro_map", "macro_f1_0123", "macro_f1_012")
        )
        print(f"[{i:>3}/{len(split_sets)}] {split_set:<34} {reported}")

    summary = summarize(results, predictions, f1_subsets)
    payload = {
        "tuned_on": args.tune_on,
        "selected_on": args.select_on,
        "weight_decay": tuned,
        "weight_decay_tuned_per_fold": tuned is None,
        "summary": summary,
        "split_sets": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2))

    reported = ("macro_map", "macro_f1_0123", "macro_f1_012")
    print(f"\n{'group':<34} {'n':>4}  {'metric':<14} {'pooled':>8}  per-fold mean")
    for group, stats in sorted(summary.items()):
        for name in reported:
            if name not in stats["pooled"]:
                continue
            per_fold = stats["per_fold"][name]
            # A one-fold group has nothing to average, so the two columns are the same
            # number twice. Say it once.
            shown = (
                ""
                if stats["folds"] == 1
                else f"{per_fold['mean']:.4f} +-{per_fold['sd']:.4f}"
            )
            print(
                f"{group:<34} {stats['folds']:>4}  {name:<14} "
                f"{stats['pooled'][name]:>8.4f}  {shown}"
            )


if __name__ == "__main__":
    main()
