"""Where does the representation actually beat summary statistics? Per label, not on average.

``scripts/moments_baseline.py`` answers "how much of the probe's score was available
without learning a representation" with one number, and one number hides the answer. A
macro mAP averages 60 labels equally, so a vocabulary carrying labels that no
joint-angle model can read -- and labels that four arithmetic moments already read
perfectly -- reports a small gap whatever the representation is worth on the rest.

This decomposes that gap. Every arm is scored per label on the same windows, then the
vocabulary is split by whether the representation beat the moments on that label at all.

    uv run python scripts/label_analysis.py            # collect, cache, write figures
    uv run python scripts/label_analysis.py --force    # recollect, ignoring the cache

Collection is the expensive half -- one forward pass per run plus a moments head trained
from scratch -- so it caches to ``results/long/label_analysis.npz`` and every figure
reads the cache. ``notebooks/label_analysis.ipynb`` is the viewer.

Two things the numbers depend on:

- the moments arm is trained with ``--passes 4``. Training windows are drawn at a random
  offset per epoch, so a probe sees a fresh crop every epoch and a moments head caching
  once sees one. Collecting a single pass understates the baseline by 0.065 macro mAP,
  which is larger than the effect being measured.
- the moments arm here is a *linear head*, the probe's own head on different inputs. The
  higher statistics-only number is ``moments k=50`` in ``scripts/knn_probe.py``, which
  fits nothing and gets the whole training set's geometry. :func:`statistics_ceiling`
  reads it back out of the kNN report so the ladder figure quotes the stronger baseline
  rather than the more convenient one.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch as t

from matplotlib.figure import Figure
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moments_baseline import collect as collect_moments  # noqa: E402
from moments_baseline import train_head  # noqa: E402

from sometria.catalog import load_annotations, load_label_vocabulary
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.viz import collect_predictions, label_names, load_run, per_label_ap

#: Arms scored against each other. The key is the legend label and the order is the
#: categorical slot order, so it must not be shuffled between figures -- colour follows
#: the arm, never its rank.
RUNS = {
    "probe": "runs/mae_100ep_probe_attentive_hard",
    "finetune": "runs/mae_100ep_finetune_attentive",
}

#: The kNN report the ladder figure quotes its strongest baseline from.
KNN_REPORT = "results/long/mae_100ep_knn.txt"

CACHE = Path("results/long/label_analysis.npz")

ROOT = "data/processed"
LABEL_SET = "babel_action_60"

#: A window is 240 frames at 60 Hz for every sample in the catalog.
WINDOW_S = 4.0

#: Must match ``dataloader.label_min_coverage`` in the configs the arms were trained on,
#: or the coverage reported for a label is not the coverage that produced its positives.
MIN_COVERAGE = 0.15

#: Below this the representation is called "no gain" over the moments on that label.
#: Not a significance test -- a line drawn where the per-label AP noise sits, wide enough
#: that a label on the wrong side of it is there for a reason.
GAP_THRESHOLD = 0.03

# Categorical slots 1-3 of the validated default palette. Three is the cap that clears
# the all-pairs CVD floors, which is what a scatter needs; aqua sits under 3:1 on a light
# surface, so every figure using it carries direct labels.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e4e3de"

ARM_COLOUR = {"moments": BLUE, "probe": ORANGE, "finetune": AQUA}


def _style(axes, *, grid_axis: str = "y") -> None:
    """Recessive frame: no top/right spines, one faint grid axis, ink-token text."""

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
    axes.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    axes.set_axisbelow(True)


# --------------------------------------------------------------------------- collection


def window_coverage(num_labels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per label over ``babel_official/val`` tiles: mean coverage, count, sequence share.

    Recomputes the tiling of :class:`~sometria.downstream.dataset.LabelledWindows` with
    ``tiles=True`` rather than reading the targets back, because the target is already
    thresholded and the quantity wanted here is the thing it thresholded: how much of the
    4 s window a label's positives actually occupy.

    ``sequence share`` is the fraction of a label's positives attributable to a
    sequence-type BABEL annotation, which spans ``0..dur`` and therefore covers every
    window of the sample whole. A label whose positives are mostly sequence annotations
    needs no localization at all to be predicted.
    """

    vocabulary = load_label_vocabulary(ROOT, LABEL_SET).select("label", "label_index")
    catalog = pl.read_parquet(f"{ROOT}/tables/motion_catalog.parquet")
    validation = pl.read_parquet(f"{ROOT}/tables/splits.parquet").filter(
        (pl.col("split_set") == "babel_official") & (pl.col("split") == "val")
    )
    annotations = (
        load_annotations(ROOT)
        .filter(pl.col("ontology") == "act_cat")
        .join(vocabulary, on="label", how="inner")
        .join(validation.select("sample_id"), on="sample_id", how="inner")
    )

    segments = defaultdict(list)
    for row in annotations.iter_rows(named=True):
        segments[row["sample_id"]].append(
            (row["start_t"], row["end_t"], row["label_index"], row["label_type"])
        )
    frames = {
        row["sample_id"]: (row["n_frames"], row["hz"])
        for row in catalog.select("sample_id", "hz", "n_frames").iter_rows(named=True)
    }

    covered, from_sequence = defaultdict(list), defaultdict(list)
    for sample_id, rows in segments.items():
        if sample_id not in frames:
            continue
        n_frames, hz = frames[sample_id]
        window = int(WINDOW_S * hz)
        if n_frames < window:
            continue

        starts = list(range(0, n_frames - window + 1, window))
        if starts and starts[-1] + window < n_frames:
            starts.append(n_frames - window)

        for start in starts:
            begin = start / hz
            end = begin + WINDOW_S
            coverage, kind = defaultdict(float), {}
            for segment_start, segment_end, index, label_type in rows:
                overlap = max(0.0, min(segment_end, end) - max(segment_start, begin))
                if overlap <= 0.0:
                    continue
                coverage[index] += overlap / WINDOW_S
                # A sequence span covers the window whole, so where both kinds of
                # annotation contribute it is the sequence one that made the positive.
                if label_type == "sequence":
                    kind[index] = True
                else:
                    kind.setdefault(index, False)

            for index, value in coverage.items():
                value = min(value, 1.0)
                if value >= MIN_COVERAGE:
                    covered[index].append(value)
                    from_sequence[index].append(kind[index])

    mean_coverage = np.full(num_labels, np.nan)
    positives = np.zeros(num_labels)
    sequence_share = np.full(num_labels, np.nan)
    for index, values in covered.items():
        mean_coverage[index] = float(np.mean(values))
        positives[index] = len(values)
        sequence_share[index] = float(np.mean(from_sequence[index]))
    return mean_coverage, positives, sequence_share


def moments_arm(config, num_labels: int, device: str, passes: int = 4) -> t.Tensor:
    """Per-label AP for a linear head over the four moments, on the probe's own config."""

    OmegaConf.update(config, "dataloader.num_workers", 4)
    data = LabelledMotionDataModule(config)
    data.setup("fit")

    train_x, _, train_y = collect_moments(data.train_dataloader(), passes, None)
    val_x, _, val_y = collect_moments(data.val_dataloader(), 1, None)
    head = train_head(
        (train_x, train_y), num_labels, epochs=40, batch_size=256, lr=1e-3, device=device
    )
    with t.no_grad():
        return per_label_ap(head(val_x.to(device)).cpu(), val_y)


def statistics_ceiling(report: str | Path = KNN_REPORT) -> dict[str, float]:
    """``{source: macro_map}`` at the largest k in a ``scripts/knn_probe.py`` report.

    The kNN rows are the strongest statistics-only number available and the only place
    the random-init control is recorded, so the ladder reads them rather than restating
    them by hand.
    """

    rows = {}
    for line in Path(report).read_text().splitlines():
        match = re.match(r"\s+(\S+(?: \S+)*?) k=(\d+)\s+([\d.]+)", line)
        if match:
            rows.setdefault(match.group(1), {})[int(match.group(2))] = float(match.group(3))
    return {source: values[max(values)] for source, values in rows.items()}


def build_cache(cache: Path = CACHE, *, force: bool = False, passes: int = 4) -> dict:
    """Collect every arm's per-label AP and the label statistics, or load the cache."""

    if cache.exists() and not force:
        return dict(np.load(cache, allow_pickle=True))

    t.manual_seed(13)
    device = "cuda" if t.cuda.is_available() else "cpu"
    names = label_names(ROOT, LABEL_SET)

    arms = {}
    for name, run in RUNS.items():
        model, data = load_run(run)
        model.eval()
        with t.no_grad():
            logits, targets = collect_predictions(model, data.val_dataloader())
        arms[name] = per_label_ap(logits, targets).numpy()
        print(f"  {name:<10} macro mAP {np.nanmean(arms[name]):.4f}")
        del model, data

    config = OmegaConf.load(f"{RUNS['probe']}/config.yaml")
    arms["moments"] = moments_arm(config, len(names), device, passes).numpy()
    print(f"  {'moments':<10} macro mAP {np.nanmean(arms['moments']):.4f}")

    mean_coverage, positives, sequence_share = window_coverage(len(names))
    data = {
        **arms,
        "names": np.array(names),
        "mean_coverage": mean_coverage,
        "positives": positives,
        "sequence_share": sequence_share,
        "knn": np.array([statistics_ceiling()], dtype=object),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def label_frame(data: dict) -> pl.DataFrame:
    """One row per label: the statistics, every arm's AP, and the gap that groups it."""

    gap = data["probe"] - data["moments"]
    return pl.DataFrame(
        {
            "label": list(data["names"]),
            "coverage": data["mean_coverage"],
            "sequence": data["sequence_share"],
            "positives": data["positives"].astype(int),
            "moments": data["moments"],
            "probe": data["probe"],
            "finetune": data["finetune"],
            "gap": gap,
            "group": ["earns" if value > GAP_THRESHOLD else "no gain" for value in gap],
        }
    ).sort("gap", descending=True)


# ------------------------------------------------------------------------------ figures


def plot_ladder(data: dict) -> Figure:
    """The headline: every readout against the statistics-only baselines it has to beat.

    Read the distance from ``moments k=50`` upward, not the distance from zero. The
    random-init row is the floor the same architecture reaches with no pretraining at
    all, and the two moments rows bracket what four arithmetic summaries are worth with
    and without a fitted head.
    """

    knn = data["knn"][0] if data["knn"].dtype == object else {}
    rows = [
        ("random-init kNN k=50", knn.get("random-init", np.nan), INK_3),
        ("moments head (linear)", float(np.nanmean(data["moments"])), BLUE),
        ("moments kNN k=50", knn.get("moments", np.nan), BLUE),
        ("attentive probe", float(np.nanmean(data["probe"])), ORANGE),
        ("finetune", float(np.nanmean(data["finetune"])), AQUA),
    ]

    figure, axes = plt.subplots(figsize=(8, 3.4))
    positions = np.arange(len(rows))
    axes.barh(
        positions,
        [value for _, value, _ in rows],
        color=[colour for _, _, colour in rows],
        height=0.62,
        zorder=3,
    )
    ceiling = knn.get("moments", np.nan)
    axes.axvline(ceiling, color=INK_2, lw=1.2, ls="--", zorder=4)
    axes.text(
        ceiling + 0.008,
        -0.92,
        "best statistics-only",
        color=INK_2,
        fontsize=8,
        va="center",
    )

    for position, (_, value, _) in zip(positions, rows):
        axes.text(
            value + 0.008, position, f"{value:.3f}", color=INK, fontsize=9, va="center",
            # the ceiling line crosses this label on the rows either side of it
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
            zorder=6,
        )

    axes.set_yticks(positions, [name for name, _, _ in rows], fontsize=9)
    axes.set_xlabel("macro mAP, babel_official/val", color=INK_2, fontsize=9)
    axes.set_xlim(0, max(value for _, value, _ in rows) * 1.18)
    # room under the last bar for the ceiling annotation, which has nowhere to sit inside
    axes.set_ylim(-1.25, len(rows) - 0.4)
    axes.set_title("Every readout, against what statistics alone already deliver", color=INK, fontsize=11)
    _style(axes, grid_axis="x")
    figure.tight_layout()
    return figure


def plot_group_summary(data: dict) -> Figure:
    """Macro mAP recomputed over the two halves of the vocabulary, per arm.

    The left group is the number every table reports. The right two are what it averages:
    35 labels where the representation is worth 0.10 over the moments, and 25 where it is
    worth nothing -- and the 25 pull the reported number down by a third.
    """

    frame = label_frame(data)
    groups = [
        ("all 60 labels", frame),
        (f"earns ({frame.filter(pl.col('group') == 'earns').height} labels)", frame.filter(pl.col("group") == "earns")),
        (f"no gain ({frame.filter(pl.col('group') == 'no gain').height} labels)", frame.filter(pl.col("group") == "no gain")),
    ]
    arms = ("moments", "probe", "finetune")

    figure, axes = plt.subplots(figsize=(8, 4))
    width = 0.26
    positions = np.arange(len(groups))
    for offset, arm in enumerate(arms):
        values = [float(subset[arm].mean()) for _, subset in groups]
        bars = axes.bar(
            positions + (offset - 1) * width,
            values,
            width=width - 0.02,  # a 2px surface gap between adjacent bars
            color=ARM_COLOUR[arm],
            label=arm,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                color=INK,
                fontsize=8.5,
            )

    axes.set_xticks(positions, [name for name, _ in groups], fontsize=9)
    axes.set_ylabel("macro mAP", color=INK_2, fontsize=9)
    axes.set_ylim(0, 0.62)
    axes.set_title(
        "The reported macro mAP averages two different populations", color=INK, fontsize=11
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncols=3, loc="upper left")
    _style(axes)
    figure.tight_layout()
    return figure


def plot_scatter(data: dict, annotate: int = 5) -> Figure:
    """Every label as moments AP against probe AP. The diagonal is "pretraining bought nothing".

    Distance above the line is the representation's contribution on that label, in the
    units the metric is reported in. Points *below* the line are labels the moments read
    better than the transformer.
    """

    frame = label_frame(data)
    figure, axes = plt.subplots(figsize=(6.4, 6.2))
    axes.plot([0, 1], [0, 1], color=INK_3, lw=1.2, ls="--", zorder=2)
    axes.text(0.72, 0.70, "no gain over moments", color=INK_3, fontsize=8, rotation=45, ha="center")

    for group, colour in (("earns", ORANGE), ("no gain", BLUE)):
        subset = frame.filter(pl.col("group") == group)
        axes.scatter(
            subset["moments"],
            subset["probe"],
            s=28 + subset["positives"].to_numpy() / 12,
            color=colour,
            edgecolor="white",  # 2px surface ring, so overlapping points stay separable
            linewidth=1.2,
            label=f"{group} ({subset.height})",
            zorder=3,
        )

    extremes = pl.concat([frame.head(annotate), frame.tail(3)])
    for index, row in enumerate(extremes.iter_rows(named=True)):
        axes.annotate(
            row["label"],
            (row["moments"], row["probe"]),
            textcoords="offset points",
            # alternated, because the largest gaps cluster and a fixed offset stacks them
            xytext=(8, 4) if index % 2 == 0 else (8, -10),
            fontsize=7.5,
            color=INK_2,
        )

    axes.set_xlabel("moments head AP", color=INK_2, fontsize=9)
    axes.set_ylabel("attentive probe AP", color=INK_2, fontsize=9)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.set_title("Per label: what the representation adds", color=INK, fontsize=11)
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _style(axes, grid_axis="both")
    figure.tight_layout()
    return figure


def plot_gap_dumbbell(data: dict, top: int | None = 15) -> Figure:
    """Moments AP to probe AP, one row per label, ordered by the distance between them.

    ``top`` keeps the ``top`` largest gains and the ``top`` smallest, which is the pair of
    populations the finding is about; ``None`` draws the whole vocabulary.
    """

    frame = label_frame(data)
    if top is not None:
        frame = pl.concat([frame.head(top), frame.tail(top)])
    frame = frame.sort("gap")

    figure, axes = plt.subplots(figsize=(8.5, 0.26 * frame.height + 1.4))
    positions = np.arange(frame.height)
    axes.hlines(
        positions, frame["moments"], frame["probe"], color=GRID, lw=2.5, zorder=2
    )
    axes.scatter(frame["moments"], positions, s=44, color=BLUE, label="moments", zorder=3,
                 edgecolor="white", linewidth=1.2)
    axes.scatter(frame["probe"], positions, s=44, color=ORANGE, label="probe", zorder=4,
                 edgecolor="white", linewidth=1.2)

    for position, row in zip(positions, frame.iter_rows(named=True)):
        axes.text(
            max(row["moments"], row["probe"]) + 0.015,
            position,
            f"{row['gap']:+.2f}",
            color=INK_2 if row["gap"] > GAP_THRESHOLD else INK_3,
            fontsize=7.5,
            va="center",
        )

    boundary = int((frame["gap"] <= GAP_THRESHOLD).sum())
    if 0 < boundary < frame.height:
        axes.axhline(boundary - 0.5, color=INK_3, lw=1, ls=":", zorder=1)
        axes.text(
            0.015, boundary - 0.42, f"gap = {GAP_THRESHOLD}", color=INK_3, fontsize=7.5,
            va="bottom", ha="left",
        )

    axes.set_yticks(positions, frame["label"].to_list(), fontsize=8)
    axes.set_xlabel("average precision", color=INK_2, fontsize=9)
    axes.set_xlim(0, 1.06)
    axes.set_title(
        "Where the representation earns, and where it does not", color=INK, fontsize=11
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _style(axes, grid_axis="x")
    figure.tight_layout()
    return figure


def plot_coverage_null(data: dict) -> Figure:
    """The falsified hypothesis: gap against how much of the window a label's positives fill.

    If the 4 s window were diluting short actions into unlearnable targets, the gap would
    be largest where coverage is lowest. It is not: the correlation is weakly *positive*,
    and the bucket means are lowest in the middle. The window length is not what is
    capping the score.
    """

    frame = label_frame(data).drop_nulls("coverage")
    coverage = frame["coverage"].to_numpy()
    gap = frame["gap"].to_numpy()

    figure, axes = plt.subplots(figsize=(7.6, 4.2))
    for group, colour in (("earns", ORANGE), ("no gain", BLUE)):
        subset = frame.filter(pl.col("group") == group)
        axes.scatter(
            subset["coverage"], subset["gap"], s=30, color=colour, alpha=0.85,
            edgecolor="white", linewidth=1.2, label=group, zorder=3,
        )

    axes.axhline(0, color=INK_3, lw=1, zorder=2)
    for low, high in ((MIN_COVERAGE, 0.45), (0.45, 0.75), (0.75, 1.01)):
        inside = (coverage >= low) & (coverage < high)
        if not inside.any():
            continue
        mean = gap[inside].mean()
        axes.hlines(mean, low, high, color=INK, lw=2.5, zorder=5)
        axes.text(
            (low + high) / 2, mean + 0.012, f"{mean:+.3f}", ha="center",
            color=INK, fontsize=8.5, zorder=6,
        )

    correlation = float(np.corrcoef(coverage, gap)[0, 1])
    axes.set_xlabel(
        "mean coverage of the 4 s window across the label's positive windows",
        color=INK_2, fontsize=9,
    )
    axes.set_ylabel("probe AP − moments AP", color=INK_2, fontsize=9)
    axes.set_title(
        f"Window dilution does not explain the gap  (r = {correlation:+.2f}, "
        "black bars are bucket means)",
        color=INK, fontsize=10.5,
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _style(axes)
    figure.tight_layout()
    return figure


FIGURES = {
    "ladder": plot_ladder,
    "group_summary": plot_group_summary,
    "scatter": plot_scatter,
    "gap_dumbbell": plot_gap_dumbbell,
    "coverage_null": plot_coverage_null,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="recollect, ignoring the cache")
    parser.add_argument("--passes", type=int, default=4, help="train collections for the moments head")
    parser.add_argument("--out", type=Path, default=Path("images/label_analysis"))
    arguments = parser.parse_args()

    data = build_cache(force=arguments.force, passes=arguments.passes)
    frame = label_frame(data)

    with pl.Config(tbl_rows=-1, tbl_width_chars=140):
        print(frame)

    for group in ("earns", "no gain"):
        subset = frame.filter(pl.col("group") == group)
        print(
            f"{group:<8} {subset.height:>3} labels   "
            f"moments {subset['moments'].mean():.3f}   "
            f"probe {subset['probe'].mean():.3f}   "
            f"finetune {subset['finetune'].mean():.3f}   "
            f"finetune - probe {subset['finetune'].mean() - subset['probe'].mean():+.3f}"
        )

    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, figure_function in FIGURES.items():
        figure = figure_function(data)
        figure.savefig(arguments.out / f"{name}.png", dpi=160)
        plt.close(figure)
    print(f"figures written to {arguments.out}")


if __name__ == "__main__":
    main()
