#!/usr/bin/env python
"""Per-label view of a 3x3 baseline-MAE probe matrix against matching moments.

Usage:

    uv run python scripts/baseline_matrix_label_analysis.py \
      --probe-root runs/baseline_matrix_probe_mean_100ep+100ep_medium

The moments baseline is trained once per input arm and compared to probe cells with the
same input channels:

    in_p__loss_*   vs moments_p
    in_pk__loss_*  vs moments_pk
    in_pkd__loss_* vs moments_pkd
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch as t
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import label_analysis as legacy_label_analysis  # noqa: E402
from moments_baseline import channel_slice, collect as collect_moments, evaluate, train_head  # noqa: E402
from probe_baseline_mae import BaselineBackbone  # noqa: E402

from sometria.downstream.classifier import MotionLinearClassifier  # noqa: E402
from sometria.downstream.dataset import LabelledMotionDataModule  # noqa: E402
from sometria.downstream.labels import load_label_vocabulary_index  # noqa: E402
from sometria.models.baseline import MAE  # noqa: E402
from sometria.viz import collect_predictions, label_names, per_label_ap  # noqa: E402

INPUTS = ("p", "pk", "pkd")
LOSSES = ("p", "pk", "pkd")
CHANNELS = {"p": (0, 1), "pk": (0, 1, 2, 3), "pkd": (0, 1, 2, 3, 4)}
METRICS = (
    "val/macro_map",
    "val/micro_map",
    "val/top_1_rec",
    "val/top_3_rec",
    "val/top_5_rec",
    "val/macro_f1s",
    "val/loss",
)
GAP_THRESHOLD = 0.03


def latest_checkpoint(run: Path) -> Path:
    versions = sorted(
        run.glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.removeprefix("version_")),
    )
    for version in reversed(versions):
        best = list((version / "checkpoints").glob("epoch=*.ckpt"))
        if best:
            return max(best, key=lambda p: p.stat().st_mtime)
        last = version / "checkpoints" / "last.ckpt"
        if last.exists():
            return last
    raise FileNotFoundError(f"no checkpoint under {run}")


def checkpoint_epoch(checkpoint: Path) -> int | None:
    match = re.search(r"epoch=(\d+)", checkpoint.name)
    return int(match.group(1)) if match else None


def metric_row(run: Path, checkpoint: Path) -> dict[str, float]:
    files = sorted(run.glob("lightning_logs/version_*/metrics.csv"))
    if not files:
        return {metric: np.nan for metric in METRICS}

    frame = pl.read_csv(files[-1], infer_schema_length=None)
    epoch = checkpoint_epoch(checkpoint)
    rows = frame if epoch is None else frame.filter(pl.col("epoch") == epoch)
    have = [metric for metric in METRICS if metric in rows.columns]
    rows = rows.select(have).drop_nulls()
    if rows.height == 0:
        rows = frame.select(have).drop_nulls().sort("val/macro_map", descending=True).head(1)
    return {metric: (float(rows[metric][0]) if metric in have else np.nan) for metric in METRICS}


def load_probe(run: Path, device: str) -> tuple[MotionLinearClassifier, LabelledMotionDataModule, Path]:
    config = OmegaConf.load(run / "config.yaml")
    OmegaConf.update(config, "dataloader.num_workers", 0)
    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)

    mae = MAE.load_from_checkpoint(config.model.checkpoint, map_location="cpu")
    model = MotionLinearClassifier(
        BaselineBackbone(mae),
        num_labels=num_labels,
        pool=config.model.get("pool", "mean"),
        lr=config.model.get("lr", 1e-3),
        min_lr_frac=config.model.get("min_lr_frac", 0.5),
        warmup=config.model.get("warmup", 0.03),
    )
    checkpoint = latest_checkpoint(run)
    state = t.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval().to(device)

    data = LabelledMotionDataModule(config)
    data.setup("fit")
    return model, data, checkpoint


@t.no_grad()
def collect_probe(run: Path, device: str) -> tuple[np.ndarray, dict[str, float]]:
    model, data, checkpoint = load_probe(run, device)
    logits, targets = collect_predictions(model, data.val_dataloader())
    ap = per_label_ap(logits, targets).numpy()
    scores = metric_row(run, checkpoint)
    scores["val/macro_map_from_labels"] = float(np.nanmean(ap))
    del model, data, logits, targets
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    return ap, scores


def collect_moment_arms(config, *, passes: int, epochs: int, batch_size: int, lr: float, seed: int, device: str):
    OmegaConf.update(config, "dataloader.num_workers", 4)
    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)
    data = LabelledMotionDataModule(config)
    data.setup("fit")

    print(f"collecting moments ({passes} train passes, 1 val pass)")
    train_x, _, train_y = collect_moments(data.train_dataloader(), passes, None)
    val_x, _, val_y = collect_moments(data.val_dataloader(), 1, None)

    num_channels = config.encoder.num_features
    aps, scores = [], []
    for name in INPUTS:
        channels = CHANNELS[name]
        if not all(channel < num_channels for channel in channels):
            raise SystemExit(f"{name} channels {channels} out of range for {num_channels} channels")

        train_subset = (channel_slice(train_x, channels, num_channels), train_y)
        val_subset = (channel_slice(val_x, channels, num_channels), val_y)
        print(f"moments_{name}: {train_subset[0].shape[1]} features")
        t.manual_seed(seed)
        head = train_head(
            train_subset,
            num_labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        with t.no_grad():
            logits = head(val_subset[0].to(device)).cpu()
            arm_ap = per_label_ap(logits, val_subset[1]).numpy()
        arm_scores = evaluate(head, val_subset, num_labels, device)
        arm_scores["val/macro_map_from_labels"] = float(np.nanmean(arm_ap))
        aps.append(arm_ap)
        scores.append([arm_scores.get(metric, np.nan) for metric in METRICS])
        del head, logits

    return np.stack(aps), np.asarray(scores, dtype=float)


def build_cache(arguments) -> dict:
    cache = arguments.cache or (arguments.output / "cache.npz")
    if cache.exists() and not arguments.force:
        return dict(np.load(cache, allow_pickle=True))

    device = arguments.device
    if device == "auto":
        device = "cuda" if t.cuda.is_available() else "cpu"
    t.manual_seed(arguments.seed)

    first_config = OmegaConf.load(arguments.probe_root / "in_p__loss_p" / "config.yaml")
    names = label_names(first_config.dataloader.root, first_config.dataloader.label_set)
    moments_epochs = arguments.moments_epochs or int(first_config.training.epochs)

    # Reuse the old coverage routine, but point its globals at this run's data/config.
    legacy_label_analysis.ROOT = str(first_config.dataloader.root)
    legacy_label_analysis.LABEL_SET = str(first_config.dataloader.label_set)
    legacy_label_analysis.MIN_COVERAGE = float(first_config.dataloader.label_min_coverage)
    mean_coverage, positives, sequence_share = legacy_label_analysis.window_coverage(len(names))

    probe_ap = np.full((len(INPUTS), len(LOSSES), len(names)), np.nan, dtype=float)
    probe_scores = np.full((len(INPUTS), len(LOSSES), len(METRICS)), np.nan, dtype=float)
    for i, input_name in enumerate(INPUTS):
        for j, loss_name in enumerate(LOSSES):
            run = arguments.probe_root / f"in_{input_name}__loss_{loss_name}"
            print(f"probe {run.name}")
            ap, scores = collect_probe(run, device)
            probe_ap[i, j] = ap
            probe_scores[i, j] = [scores.get(metric, np.nan) for metric in METRICS]

    moments_ap, moments_scores = collect_moment_arms(
        first_config,
        passes=arguments.passes,
        epochs=moments_epochs,
        batch_size=arguments.moments_batch_size,
        lr=arguments.moments_lr,
        seed=arguments.seed,
        device=device,
    )

    data = {
        "names": np.asarray(names),
        "inputs": np.asarray(INPUTS),
        "losses": np.asarray(LOSSES),
        "metrics": np.asarray(METRICS),
        "probe_ap": probe_ap,
        "probe_scores": probe_scores,
        "moments_ap": moments_ap,
        "moments_scores": moments_scores,
        "mean_coverage": mean_coverage,
        "positives": positives,
        "sequence_share": sequence_share,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def matrix_md(values: np.ndarray) -> str:
    lines = ["| input \\ loss | p | pk | pkd |", "| :--- | ---: | ---: | ---: |"]
    for input_name, row in zip(INPUTS, values):
        lines.append(f"| **{input_name}** | " + " | ".join(f"{value:.4f}" for value in row) + " |")
    return "\n".join(lines)


def cell_name(i: int, j: int) -> str:
    return f"in_{INPUTS[i]}__loss_{LOSSES[j]}"


def safe_argmax(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.unravel_index(np.nanargmax(values), values.shape))


def _style(axes, *, grid_axis: str = "y") -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis=grid_axis, color="#e4e3de", lw=0.8, zorder=0)
    axes.set_axisbelow(True)


def best_label_frame(data: dict) -> tuple[pl.DataFrame, int, int]:
    """Per-label AP for the best macro-mAP cell against its matching moments arm."""

    names = data["names"].astype(str)
    probe_ap = data["probe_ap"].astype(float)
    moments_ap = data["moments_ap"].astype(float)
    scores = data["probe_scores"].astype(float)
    best_i, best_j = safe_argmax(scores[:, :, METRICS.index("val/macro_map")])
    gap = probe_ap[best_i, best_j] - moments_ap[best_i]

    return (
        pl.DataFrame(
            {
                "label": names,
                "coverage": data["mean_coverage"],
                "sequence": data["sequence_share"],
                "positives": data["positives"].astype(int),
                "moments": moments_ap[best_i],
                "probe": probe_ap[best_i, best_j],
                "gap": gap,
                "group": ["earns" if value > GAP_THRESHOLD else "no gain" for value in gap],
            }
        ).sort("gap", descending=True),
        best_i,
        best_j,
    )


def plot_group_summary(data: dict):
    frame, best_i, best_j = best_label_frame(data)
    groups = [
        ("all labels", frame),
        (f"earns ({frame.filter(pl.col('group') == 'earns').height})", frame.filter(pl.col("group") == "earns")),
        (f"no gain ({frame.filter(pl.col('group') == 'no gain').height})", frame.filter(pl.col("group") == "no gain")),
    ]

    figure, axes = plt.subplots(figsize=(8, 4))
    width = 0.34
    positions = np.arange(len(groups))
    for offset, (name, colour) in enumerate((("moments", "#2a78d6"), (cell_name(best_i, best_j), "#eb6834"))):
        values = [float(subset["moments" if name == "moments" else "probe"].mean()) for _, subset in groups]
        bars = axes.bar(positions + (offset - 0.5) * width, values, width=width - 0.02, color=colour, label=name, zorder=3)
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    axes.set_xticks(positions, [name for name, _ in groups])
    axes.set_ylabel("macro mAP")
    axes.set_ylim(0, max(0.62, float(frame.select(pl.max_horizontal("moments", "probe")).to_series().max()) * 1.18))
    axes.set_title("Reported macro mAP averages two populations")
    axes.legend(frameon=False, ncols=2, loc="upper left")
    _style(axes)
    figure.tight_layout()
    return figure


def plot_scatter(data: dict, annotate: int = 8):
    frame, best_i, best_j = best_label_frame(data)
    figure, axes = plt.subplots(figsize=(6.4, 6.2))
    axes.plot([0, 1], [0, 1], color="#8a8880", lw=1.2, ls="--", zorder=2)
    for group, colour in (("earns", "#eb6834"), ("no gain", "#2a78d6")):
        subset = frame.filter(pl.col("group") == group)
        axes.scatter(subset["moments"], subset["probe"], s=28 + subset["positives"].to_numpy() / 12, color=colour, edgecolor="white", linewidth=1.0, label=f"{group} ({subset.height})", zorder=3)

    extremes = pl.concat([frame.head(annotate), frame.tail(3)])
    for index, row in enumerate(extremes.iter_rows(named=True)):
        axes.annotate(row["label"], (row["moments"], row["probe"]), textcoords="offset points", xytext=(8, 4) if index % 2 == 0 else (8, -10), fontsize=7)

    axes.set_xlabel(f"moments_{INPUTS[best_i]} AP")
    axes.set_ylabel(f"{cell_name(best_i, best_j)} AP")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.set_title("Per label: what the representation adds")
    axes.legend(frameon=False, loc="lower right")
    _style(axes, grid_axis="both")
    figure.tight_layout()
    return figure


def plot_gap_dumbbell(data: dict, top: int | None = 15):
    frame, best_i, best_j = best_label_frame(data)
    if top is not None:
        frame = pl.concat([frame.head(top), frame.tail(top)])
    frame = frame.sort("gap")

    figure, axes = plt.subplots(figsize=(8.5, 0.26 * frame.height + 1.4))
    positions = np.arange(frame.height)
    axes.hlines(positions, frame["moments"], frame["probe"], color="#e4e3de", lw=2.5, zorder=2)
    axes.scatter(frame["moments"], positions, s=44, color="#2a78d6", label=f"moments_{INPUTS[best_i]}", zorder=3, edgecolor="white", linewidth=1.0)
    axes.scatter(frame["probe"], positions, s=44, color="#eb6834", label=cell_name(best_i, best_j), zorder=4, edgecolor="white", linewidth=1.0)

    for position, row in zip(positions, frame.iter_rows(named=True)):
        axes.text(max(row["moments"], row["probe"]) + 0.015, position, f"{row['gap']:+.2f}", fontsize=7.5, va="center")

    boundary = int((frame["gap"] <= GAP_THRESHOLD).sum())
    if 0 < boundary < frame.height:
        axes.axhline(boundary - 0.5, color="#8a8880", lw=1, ls=":", zorder=1)
        axes.text(0.015, boundary - 0.42, f"gap = {GAP_THRESHOLD}", color="#8a8880", fontsize=7.5, va="bottom", ha="left")

    axes.set_yticks(positions, frame["label"].to_list(), fontsize=8)
    axes.set_xlabel("average precision")
    axes.set_xlim(0, 1.06)
    axes.set_title("Where the representation earns, and where it does not")
    axes.legend(frameon=False, loc="lower right")
    _style(axes, grid_axis="x")
    figure.tight_layout()
    return figure


def plot_coverage_null(data: dict):
    frame, _, _ = best_label_frame(data)
    frame = frame.drop_nulls("coverage")
    coverage = frame["coverage"].to_numpy()
    gap = frame["gap"].to_numpy()

    figure, axes = plt.subplots(figsize=(7.6, 4.2))
    for group, colour in (("earns", "#eb6834"), ("no gain", "#2a78d6")):
        subset = frame.filter(pl.col("group") == group)
        axes.scatter(subset["coverage"], subset["gap"], s=30, color=colour, alpha=0.85, edgecolor="white", linewidth=1.0, label=group, zorder=3)

    axes.axhline(0, color="#8a8880", lw=1, zorder=2)
    for low, high in ((legacy_label_analysis.MIN_COVERAGE, 0.45), (0.45, 0.75), (0.75, 1.01)):
        inside = (coverage >= low) & (coverage < high)
        if inside.any():
            mean = gap[inside].mean()
            axes.hlines(mean, low, high, color="#0b0b0b", lw=2.5, zorder=5)
            axes.text((low + high) / 2, mean + 0.012, f"{mean:+.3f}", ha="center", fontsize=8.5, zorder=6)

    corr = float(np.corrcoef(coverage, gap)[0, 1]) if len(coverage) > 1 else float("nan")
    axes.set_xlabel("mean coverage of positive 4 s windows")
    axes.set_ylabel("probe AP − moments AP")
    axes.set_title(f"Window dilution check (r = {corr:+.2f})")
    axes.legend(frameon=False, loc="upper left")
    _style(axes)
    figure.tight_layout()
    return figure


def plot_heatmap(values: np.ndarray, title: str, *, center_zero: bool = False):
    figure, axes = plt.subplots(figsize=(4.8, 3.6))
    kwargs = {}
    if center_zero:
        vmax = float(np.nanmax(np.abs(values)))
        kwargs = {"vmin": -vmax, "vmax": vmax}
    image = axes.imshow(values, cmap="coolwarm" if center_zero else "viridis", **kwargs)
    axes.set_xticks(range(len(LOSSES)), LOSSES)
    axes.set_yticks(range(len(INPUTS)), INPUTS)
    axes.set_xlabel("loss channels")
    axes.set_ylabel("input channels")
    axes.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            axes.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def all_cell_names() -> list[str]:
    return [cell_name(i, j) for i in range(3) for j in range(3)]


def plot_all_cells_lift_heatmap(data: dict, top: int | None = None):
    """Labels x cells heatmap: where each pretraining setup beats matching moments."""

    names = data["names"].astype(str)
    lift = (data["probe_ap"].astype(float) - data["moments_ap"].astype(float)[:, None, :]).reshape(9, len(names)).T
    best = np.nanmax(lift, axis=1)
    order = np.argsort(np.nan_to_num(best, nan=-999.0))[::-1]
    if top is not None:
        order = order[:top]
    shown = lift[order]

    figure, axes = plt.subplots(figsize=(9.5, 0.24 * len(order) + 2.0))
    vmax = float(np.nanmax(np.abs(shown)))
    image = axes.imshow(shown, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes.set_xticks(range(9), all_cell_names(), rotation=45, ha="right", fontsize=7)
    axes.set_yticks(range(len(order)), names[order], fontsize=7)
    axes.set_title(f"Per-label lift over matching moments, top {len(order)} labels")
    axes.set_xlabel("probe cell")
    axes.set_ylabel("label")
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="AP lift")
    figure.tight_layout()
    return figure


def plot_winner_counts(data: dict):
    """How often each cell gives the largest per-label lift."""

    lift = (data["probe_ap"].astype(float) - data["moments_ap"].astype(float)[:, None, :]).reshape(9, -1)
    winners = np.nanargmax(lift, axis=0)
    counts = np.bincount(winners, minlength=9)

    figure, axes = plt.subplots(figsize=(8.5, 3.4))
    positions = np.arange(9)
    axes.bar(positions, counts, color="#eb6834", zorder=3)
    axes.set_xticks(positions, all_cell_names(), rotation=45, ha="right", fontsize=7)
    axes.set_ylabel("labels won")
    axes.set_title("Which cell gives the best per-label lift?")
    for position, count in zip(positions, counts):
        axes.text(position, count + 0.4, str(int(count)), ha="center", fontsize=8)
    _style(axes)
    figure.tight_layout()
    return figure


def plot_all_cells_dumbbell(data: dict, top: int | None = None):
    """Range of all 9 probe APs per label, best cell highlighted."""

    names = data["names"].astype(str)
    probe = data["probe_ap"].astype(float).reshape(9, len(names)).T
    best = np.nanmax(probe, axis=1)
    worst = np.nanmin(probe, axis=1)
    spread = best - worst
    order = np.argsort(np.nan_to_num(spread, nan=-1.0))[::-1]
    if top is not None:
        order = order[:top]
    probe = probe[order]

    figure, axes = plt.subplots(figsize=(9, 0.24 * len(order) + 1.6))
    positions = np.arange(len(order))
    axes.hlines(positions, np.nanmin(probe, axis=1), np.nanmax(probe, axis=1), color="#e4e3de", lw=2.5, zorder=1)
    for cell_index, name in enumerate(all_cell_names()):
        axes.scatter(probe[:, cell_index], positions, s=18, alpha=0.65, label=name, zorder=2)
    best_indices = np.nanargmax(probe, axis=1)
    axes.scatter(probe[np.arange(len(order)), best_indices], positions, s=52, facecolors="none", edgecolors="#0b0b0b", linewidth=1.2, label="label winner", zorder=4)

    axes.set_yticks(positions, names[order], fontsize=7)
    axes.set_xlabel("average precision")
    axes.set_xlim(0, 1.02)
    suffix = "all labels" if top is None else f"top {len(order)} by disagreement"
    axes.set_title(f"All probe cells per label, {suffix}")
    axes.legend(frameon=False, fontsize=6, ncols=3, loc="lower right")
    _style(axes, grid_axis="x")
    figure.tight_layout()
    return figure


def save_figures(data: dict, output: Path, macro_lift: np.ndarray) -> None:
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "macro_map_heatmap": plot_heatmap(data["probe_scores"][:, :, METRICS.index("val/macro_map")], "macro mAP"),
        "macro_lift_heatmap": plot_heatmap(macro_lift, "macro AP lift over matching moments", center_zero=True),
        "group_summary": plot_group_summary(data),
        "scatter": plot_scatter(data),
        "gap_dumbbell": plot_gap_dumbbell(data),
        "coverage_null": plot_coverage_null(data),
        "all_cells_lift_heatmap": plot_all_cells_lift_heatmap(data),
        "winner_counts": plot_winner_counts(data),
        "all_cells_dumbbell": plot_all_cells_dumbbell(data),
    }
    for name, figure in figures.items():
        figure.savefig(image_dir / f"{name}.png", dpi=160)
        plt.close(figure)


def write_outputs(data: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = data["names"].astype(str)
    probe_ap = data["probe_ap"].astype(float)
    moments_ap = data["moments_ap"].astype(float)
    probe_scores = data["probe_scores"].astype(float)
    moments_scores = data["moments_scores"].astype(float)

    lift = probe_ap - moments_ap[:, None, :]
    macro_lift = np.nanmean(lift, axis=2)
    macro_from_labels = np.nanmean(probe_ap, axis=2)
    best_i, best_j = safe_argmax(probe_scores[:, :, METRICS.index("val/macro_map")])
    best_lift_i, best_lift_j = safe_argmax(macro_lift)

    summary_rows = []
    for i, input_name in enumerate(INPUTS):
        for j, loss_name in enumerate(LOSSES):
            row = {"input": input_name, "loss": loss_name, "cell": cell_name(i, j)}
            row.update({metric: probe_scores[i, j, k] for k, metric in enumerate(METRICS)})
            row["macro_map_from_labels"] = macro_from_labels[i, j]
            row["macro_lift_over_matching_moments"] = macro_lift[i, j]
            summary_rows.append(row)
    pl.DataFrame(summary_rows).write_csv(output / "summary_metrics.csv")

    moment_rows = []
    for i, input_name in enumerate(INPUTS):
        row = {"input": input_name, "arm": f"moments_{input_name}"}
        row.update({metric: moments_scores[i, k] for k, metric in enumerate(METRICS) if metric != "val/loss"})
        row["macro_map_from_labels"] = float(np.nanmean(moments_ap[i]))
        moment_rows.append(row)
    pl.DataFrame(moment_rows).write_csv(output / "moments_metrics.csv")

    label_rows = []
    flat_lift = lift.reshape(9, len(names))
    flat_ap = probe_ap.reshape(9, len(names))
    for label_index, label in enumerate(names):
        row = {
            "label": label,
            "coverage": float(data["mean_coverage"][label_index]),
            "positives": int(data["positives"][label_index]),
            "sequence_share": float(data["sequence_share"][label_index]),
        }
        for i, input_name in enumerate(INPUTS):
            row[f"moments_{input_name}"] = float(moments_ap[i, label_index])
            for j, loss_name in enumerate(LOSSES):
                row[f"probe_in_{input_name}__loss_{loss_name}"] = float(probe_ap[i, j, label_index])
                row[f"lift_in_{input_name}__loss_{loss_name}"] = float(lift[i, j, label_index])

        label_ap = flat_ap[:, label_index]
        label_lift = flat_lift[:, label_index]
        if np.isfinite(label_ap).any():
            best = int(np.nanargmax(label_ap))
            bi, bj = divmod(best, 3)
            row["best_probe_cell"] = cell_name(bi, bj)
            row["best_probe_ap"] = float(probe_ap[bi, bj, label_index])
            row["best_probe_lift"] = float(lift[bi, bj, label_index])
        if np.isfinite(label_lift).any():
            best = int(np.nanargmax(label_lift))
            bi, bj = divmod(best, 3)
            row["best_lift_cell"] = cell_name(bi, bj)
            row["best_lift"] = float(lift[bi, bj, label_index])
        label_rows.append(row)
    label_frame = pl.DataFrame(label_rows)
    label_frame.write_csv(output / "per_label.csv")

    best_lift = lift[best_i, best_j]
    best_frame = label_frame.with_columns(
        pl.Series("current_best_lift", best_lift),
        pl.Series("current_best_ap", probe_ap[best_i, best_j]),
        pl.Series("current_best_moments", moments_ap[best_i]),
    )
    gained = best_frame.sort("current_best_lift", descending=True).head(10)
    lost = best_frame.sort("current_best_lift").head(10)

    win_counts = np.zeros((3, 3), dtype=int)
    for label_index in range(len(names)):
        label_lift = flat_lift[:, label_index]
        if np.isfinite(label_lift).any():
            winner = int(np.nanargmax(label_lift))
            wi, wj = divmod(winner, 3)
            win_counts[wi, wj] += 1

    def label_table(frame: pl.DataFrame) -> str:
        rows = ["| label | AP | moments | lift | positives |", "| :--- | ---: | ---: | ---: | ---: |"]
        for row in frame.select("label", "current_best_ap", "current_best_moments", "current_best_lift", "positives").iter_rows(named=True):
            rows.append(
                f"| {row['label']} | {row['current_best_ap']:.3f} | {row['current_best_moments']:.3f} | "
                f"{row['current_best_lift']:+.3f} | {int(row['positives'])} |"
            )
        return "\n".join(rows)

    metric = lambda name: probe_scores[:, :, METRICS.index(name)]
    moments_macro = np.nanmean(moments_ap, axis=1)
    earn_mask = best_lift > GAP_THRESHOLD

    report = f"""# Baseline matrix label analysis

Probe root analysed against matching moments arms.

## Probe metrics

### Macro mAP

{matrix_md(metric("val/macro_map"))}

### Micro mAP

{matrix_md(metric("val/micro_map"))}

### Recall@1

{matrix_md(metric("val/top_1_rec"))}

### Recall@3

{matrix_md(metric("val/top_3_rec"))}

### Recall@5

{matrix_md(metric("val/top_5_rec"))}

## Lift over matching moments, macro AP

Each row subtracts its own moments baseline: `p` vs `moments_p`, `pk` vs `moments_pk`, `pkd` vs `moments_pkd`.

{matrix_md(macro_lift)}

## Moments macro AP

| arm | macro AP |
| :--- | ---: |
| moments_p | {moments_macro[0]:.4f} |
| moments_pk | {moments_macro[1]:.4f} |
| moments_pkd | {moments_macro[2]:.4f} |

## Winners

- Best probe macro mAP: `{cell_name(best_i, best_j)}` = {metric("val/macro_map")[best_i, best_j]:.4f}
- Best macro lift over matching moments: `{cell_name(best_lift_i, best_lift_j)}` = {macro_lift[best_lift_i, best_lift_j]:+.4f}
- Current best `{cell_name(best_i, best_j)}` beats `moments_{INPUTS[best_i]}` on {int(np.nansum(best_lift > 0))}/{len(names)} labels, and clears +{GAP_THRESHOLD:.2f} on {int(np.nansum(earn_mask))}/{len(names)} labels.

## Per-label lift winner counts

{matrix_md(win_counts.astype(float))}

## Current best: labels gained most over matching moments

{label_table(gained)}

## Current best: labels lost most against matching moments

{label_table(lost)}
"""
    (output / "overview.md").write_text(report)
    save_figures(data, output, macro_lift)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--probe-root", type=Path, default=Path("runs/baseline_matrix_probe_mean_100ep+100ep_medium"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--moments-epochs", type=int, default=None, help="default: probe training.epochs")
    parser.add_argument("--moments-batch-size", type=int, default=256)
    parser.add_argument("--moments-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    args.output = args.output or (args.probe_root / "label_analysis")
    data = build_cache(args)
    write_outputs(data, args.output)
    print(f"wrote {args.output / 'overview.md'}")
    print(f"wrote {args.output / 'per_label.csv'}")
    print(f"wrote {args.output / 'summary_metrics.csv'}")
    print(f"wrote {args.output / 'images'}")


if __name__ == "__main__":
    main()
