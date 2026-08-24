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
    return vocabulary, vocabulary["label_index"].max() + 1


def load_label_segments(
    root: str | Path,
    *,
    label_set: str,
    sample_ids: list[str] | None = None,
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
        & pl.col("label_type").is_in(LABEL_TYPES)
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


def annotated_sample_ids(root: str | Path, ontology: str = "act_cat") -> set[str]:
    """Sample ids carrying at least one annotation in ``ontology``.

    Not the same question as "has a label in the vocabulary". A sample whose only
    ``act_cat`` segments are ``transition`` is annotated and scores as all-negative --
    that is decision 12, and at 19% of rows dropping it would bias the benchmark. A
    sample with no ``act_cat`` at all is *unknown*, and calling it negative would be
    inventing labels: BABEL withholds ``act_cat`` for its whole test split, and carries
    free-text-only annotations elsewhere.
    """

    annotations = load_annotations(root).filter(pl.col("ontology") == ontology)
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
