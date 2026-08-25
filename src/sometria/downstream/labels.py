"""What a window is: label coverage turned into a multi-hot target.

A window is not cut to segment boundaries. At a median ``act_cat`` segment length of
1.1 s a 4 s window spans several of them, so the target is the set of every action
category that covers enough of the crop -- ``LABEL_MIN_COVERAGE`` of its duration. One
definition serves the loss and the metric, so "present" never means two things.

This is deliberately not BABEL's official protocol, which scores one label per chunk and
duplicates a *k*-label segment into *k* samples, capping such a chunk at 50% Top-1.
Comparability with published BABEL numbers is given up on purpose, in exchange for a
loss with no ceiling and a downstream path that reuses the pretraining one.

Labels outside the chosen vocabulary contribute nothing, so a window covering only
out-of-scope segments -- ``transition`` is 19% of rows and has no index at all -- trains
as an all-negative example rather than being dropped. Dropping would bias evaluation
toward the segments that happen to carry scoreable labels.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch as t

from sometria.catalog import load_annotations, load_label_vocabulary

LABEL_MIN_COVERAGE = 0.15

# Sequence-level annotations are kept alongside frame-level ones. They are spans too --
# BABEL emits them over 0..dur -- and roughly 40% of sequences carry no frame annotation
# at all, so dropping them would make those samples silently all-negative.
#
# That is the right default for window classification and the wrong one for segmentation.
# A sequence label runs 0..dur exactly, so it covers every patch of every window and can
# never produce a boundary: on babel_official/val it is 40% of samples contributing full
# gradient to *what* and none to *when*, and it is why 42% of segmentation windows carry
# a constant target. Pass ``label_types=("frame",)`` there -- and filter the samples to
# match, or the sequence-only ones stay in as all-negative windows.
LABEL_TYPES = ("frame", "sequence")


def load_label_vocabulary_index(
    root: str | Path,
    label_set: str,
) -> tuple[pl.DataFrame, int]:
    """Return the vocabulary rows for ``label_set`` and how many labels it scores."""

    vocabulary = load_label_vocabulary(root, label_set)
    if vocabulary.is_empty():
        raise ValueError(
            f"no label vocabulary {label_set!r} in {root}; "
            "import it with sometria.babel.import_babel_action_vocabulary"
        )
    return vocabulary, vocabulary["label_index"].max() + 1 # type: ignore


def load_label_segments(
    root: str | Path,
    *,
    label_set: str,
    sample_ids: list[str] | None = None,
    label_types: tuple[str, ...] = LABEL_TYPES,
) -> tuple[dict[str, np.ndarray], int]:
    """Return ``{sample_id: (n, 3) array of [start_t, end_t, label_index]}``.

    Annotations whose label is absent from ``label_set`` are dropped here rather than in
    the dataset: they are out of scope for this benchmark, not errors, and the windows
    covering them still train as negatives.

    Numpy rather than torch, and this is not cosmetic. The dataset holding this dict is
    pickled into every DataLoader worker, and torch's pickler moves each tensor into
    shared memory, which costs one file descriptor per tensor. ``forkserver`` refuses to
    pass more than 256 to a child, so a few thousand samples killed the run outright with
    ``ValueError: too many fds``. An array pickles as bytes; :class:`LabelledWindows`
    wraps one in a tensor per item, which is a free view.
    """

    vocabulary, num_labels = load_label_vocabulary_index(root, label_set)
    ontology = vocabulary["ontology"][0]

    annotations = load_annotations(root).filter(
        (pl.col("ontology") == ontology)
        & pl.col("label_type").is_in(list(label_types))
        & pl.col("start_t").is_not_null()
        & pl.col("end_t").is_not_null()
    )
    if sample_ids is not None:
        annotations = annotations.filter(pl.col("sample_id").is_in(sample_ids))

    joined = annotations.join(
        vocabulary.select("label", "label_index"), on="label", how="inner"
    ).select("sample_id", "start_t", "end_t", "label_index")

    segments = {
        sample_id: np.asarray(rows, dtype=np.float32).reshape(-1, 3)
        for sample_id, rows in (
            joined.group_by("sample_id")
            .agg(pl.concat_list("start_t", "end_t", "label_index").flatten())
            .iter_rows()
        )
    }
    return segments, num_labels


def annotated_sample_ids(
    root: str | Path,
    ontology: str = "act_cat",
    label_types: tuple[str, ...] | None = None,
) -> set[str]:
    """Sample ids carrying at least one annotation in ``ontology``.

    Not the same question as "has a label in the vocabulary". A sample whose only
    ``act_cat`` segments are ``transition`` is annotated and scores as all-negative --
    that is decision 12, and at 19% of rows dropping it would bias the benchmark. A
    sample with no ``act_cat`` at all is *unknown*, and calling it negative would be
    inventing labels: BABEL withholds ``act_cat`` for its whole test split, and carries
    free-text-only annotations elsewhere.

    ``label_types`` narrows what counts as annotated, and must match whatever
    :func:`load_label_segments` was narrowed to: a sample kept here whose every segment
    is dropped there becomes an all-negative window, which is the silent-negative failure
    this function exists to prevent.
    """

    annotations = load_annotations(root).filter(pl.col("ontology") == ontology)
    if label_types is not None:
        annotations = annotations.filter(pl.col("label_type").is_in(list(label_types)))
    return set(annotations["sample_id"].unique().to_list())


def window_multi_hot(
    segments: t.Tensor,
    start_t: float,
    end_t: float,
    num_labels: int,
    min_coverage: float = LABEL_MIN_COVERAGE,
) -> t.Tensor:
    """Multi-hot target for the window ``[start_t, end_t)``, from one sample's segments.

    Coverage is the fraction of the window's duration a category spans. Float rather
    than bool because it is fed straight to ``binary_cross_entropy_with_logits``.
    """

    target = t.zeros(num_labels)
    if segments.numel() == 0 or end_t <= start_t:
        return target

    overlap = (
        segments[:, 1].clamp(max=end_t) - segments[:, 0].clamp(min=start_t)
    ).clamp(min=0.0)

    # ponytail: two segments of the same category that overlap each other are counted
    # twice, which can only push a borderline window over the threshold. Take the union
    # per label if that ever measurably matters.
    coverage = t.zeros(num_labels).index_add_(
        0, segments[:, 2].long(), overlap / (end_t - start_t)
    )
    return target.masked_fill(coverage >= min_coverage, 1.0)

def window_multi_hot_ex(
    segments: t.Tensor,
    start_t: float,
    end_t: float,
    num_labels: int,
    min_coverage: float = LABEL_MIN_COVERAGE,
) -> t.Tensor:
    """Multi-hot target for the window ``[start_t, end_t)``, from one sample's segments."""

    target = t.zeros(num_labels, dtype=t.float32)
    duration = end_t - start_t

    if segments.numel() == 0 or duration <= 0.0:
        return target

    # Calculate interval intersections with window [start_t, end_t]
    starts = segments[:, 0].clamp(min=start_t)
    ends = segments[:, 1].clamp(max=end_t)
    overlaps = (ends - starts).clamp(min=0.0)

    # Filter out non-overlapping segments early
    valid_mask = overlaps > 0.0
    if not valid_mask.any():
        return target

    valid_overlaps = overlaps[valid_mask]
    label_ids = segments[valid_mask, 2].long()

    # Guard against invalid label indices
    label_mask = (label_ids >= 0) & (label_ids < num_labels)
    if not label_mask.any():
        return target

    valid_overlaps = valid_overlaps[label_mask]
    label_ids = label_ids[label_mask]

    # Compute maximum coverage per label without double-counting overlapping intervals of the same class
    # For disjoint segments of the same label, sum per unique class index
    coverage = t.zeros(num_labels, dtype=t.float32)
    coverage.index_add_(0, label_ids, valid_overlaps / duration)

    # Clamp coverage to 1.0 max in case overlapping annotations of the same class pushed it over 100%
    coverage = coverage.clamp(max=1.0)

    return (coverage >= min_coverage).float()


def window_patch_labels(
    segments: t.Tensor,
    start_t: float,
    end_t: float,
    num_patches: int,
    num_labels: int,
    min_coverage: float = LABEL_MIN_COVERAGE,
) -> t.Tensor:
    """``(num_patches, num_labels)`` -- :func:`window_multi_hot`, resolved in time.

    The same coverage rule applied to each time patch separately instead of to the window
    as a whole, so a label marks the patches it actually spans rather than the whole
    window it touches. This is the target that makes the task temporal: a summary
    statistic over the window cannot produce it, because the answer differs per patch
    while the statistic does not.

    ``min_coverage`` means the same thing it does for a window and should usually be
    higher here: a patch is short, so a segment either covers most of it or misses.
    """

    target = t.zeros(num_patches, num_labels, dtype=t.float32)
    duration = end_t - start_t
    if segments.numel() == 0 or duration <= 0.0 or num_patches <= 0:
        return target

    # Patch p spans [edges[p], edges[p + 1]).
    edges = t.linspace(start_t, end_t, num_patches + 1, dtype=t.float64)
    patch_duration = float(duration) / num_patches

    starts = segments[:, 0].double().unsqueeze(1)            # (n, 1)
    ends = segments[:, 1].double().unsqueeze(1)
    overlap = (
        t.minimum(ends, edges[1:]) - t.maximum(starts, edges[:-1])
    ).clamp(min=0.0) / patch_duration                        # (n, num_patches)

    labels = segments[:, 2].long()
    keep = (labels >= 0) & (labels < num_labels) & (overlap > 0.0).any(dim=1)
    if not keep.any():
        return target

    # index_add_ over the label axis, so two disjoint segments of one label sum into the
    # same row rather than the later one replacing the earlier.
    coverage = t.zeros(num_patches, num_labels, dtype=t.float64)
    coverage.index_add_(1, labels[keep], overlap[keep].T)
    return (coverage.clamp(max=1.0) >= min_coverage).float()
