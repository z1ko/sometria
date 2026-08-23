
"""Catalog utilities for building trainable views over processed motion data.

The processed data root is intentionally split into separate concerns:

- ``tables/motion_catalog.parquet`` describes one row per motion sample.
- ``tables/annotations.parquet`` optionally describes labels over samples.
- ``tables/splits.parquet`` describes named dataset views such as BABEL's
  official split or a custom pretraining split.

The training code should not load these tables directly. Instead, construct a
``MotionViewSpec`` and call ``build_motion_view`` so filtering rules stay in one
place as more datasets and label sources are added.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

CATALOG = "motion_catalog.parquet"
ANNOTATIONS = "annotations.parquet"
SPLITS = "splits.parquet"
VOCABULARY = "label_vocabulary.parquet"
TABLES_DIR = "tables"

@dataclass(frozen=True)
class MotionViewSpec:
    """Declarative request for a subset of the motion catalog.

    ``split_set`` names a family of splits, for example ``"babel_official"`` or
    ``"pretrain_v1"``. ``split`` selects one member of that family. Label filters
    are separate from split filters because a sample can belong to a split even
    when labels are unavailable or withheld.
    """

    split_set: str | None = None            # "babel_official", "pretrain_v1"
    split: str | None = None                # "train", "val", "test"
    source_datasets: tuple[str, ...] = ()   # ("AMASS", "MotionX")
    label_sources: tuple[str, ...] = ()     # ("BABEL",)
    require_labels: bool = False
    exclude_broken: bool = True


def stable_sample_id(source_dataset: str, source_path: str) -> str:
    """Return the stable catalog key for one source dataset-relative file."""
    return f"{source_dataset}:{source_path}"


def table_path(root: str | Path, name: str) -> Path:
    """Return the path to a named catalog table under a processed data root."""
    return Path(root) / TABLES_DIR / name


def load_catalog(root: str | Path) -> pl.DataFrame:
    """Load the required sample catalog table."""
    return pl.read_parquet(table_path(root, CATALOG))


def load_annotations(root: str | Path) -> pl.DataFrame:
    """Load optional annotation rows, or an empty table with the expected schema.

    Annotations are modeled separately from samples so different label sources
    can coexist. For example, AMASS samples can have BABEL labels today, and a
    future MotionX import can add another ontology without changing the motion
    catalog schema.
    """

    path = table_path(root, ANNOTATIONS)
    if path.exists():
        return pl.read_parquet(path)
    else:
        return pl.DataFrame(
            schema={
                "sample_id": pl.String,
                "label_source": pl.String,
                "label_type": pl.String,
                "ontology": pl.String,
                "start_t": pl.Float64,
                "end_t": pl.Float64,
                "label": pl.String,
            }
        )


def load_splits(root: str | Path) -> pl.DataFrame:
    """Load optional split rows, or an empty table with the expected schema."""

    path = table_path(root, SPLITS)
    if path.exists():
        return pl.read_parquet(path)
    else:
        return pl.DataFrame(
            schema={
                "sample_id": pl.String,
                "split_set": pl.String,
                "split": pl.String,
            }
        )


def load_label_vocabulary(root: str | Path, label_set: str | None = None) -> pl.DataFrame:
    """Load optional label vocabularies, or an empty table with the expected schema.

    A vocabulary fixes which labels a benchmark scores and what integer each maps to. It is
    kept apart from ``annotations`` because the two answer different questions: annotations say
    what a sample is, a vocabulary says which answers are admissible and in what order. Labels
    observed in the data but absent from the vocabulary are out of scope for that benchmark,
    not errors.
    """

    path = table_path(root, VOCABULARY)
    if path.exists():
        table = pl.read_parquet(path)
        if label_set is not None:
            table = table.filter(pl.col("label_set") == label_set)
        return table

    return pl.DataFrame(
        schema={
            "label_source": pl.String,
            "label_set": pl.String,
            "ontology": pl.String,
            "label": pl.String,
            "label_index": pl.Int64,
        }
    )


def build_motion_view(root: str | Path, spec: MotionViewSpec) -> pl.DataFrame:
    """Materialize a sample table matching ``spec``.

    The returned frame is still catalog-shaped: every row points at one processed
    feature tensor through ``motion_path`` and retains metadata needed by the
    dataset class. This function performs only table filtering and joins; it does
    not touch tensor files.
    """

    catalog = load_catalog(root)
    annotations = load_annotations(root)
    splits = load_splits(root)

    df = catalog

    # Remove "broken" samples 
    if spec.exclude_broken and "broken" in df.columns:
        df = df.filter(~pl.col("broken"))

    if spec.source_datasets:
        df = df.filter(pl.col("source_dataset").is_in(spec.source_datasets))

    # Get all samples relative to a split set and a particular split
    if spec.split_set is not None:
        split_df = splits.filter(pl.col("split_set") == spec.split_set)
        if spec.split is not None:
            split_df = split_df.filter(pl.col("split") == spec.split)

        df = df.join(split_df.select("sample_id", "split_set", "split"), on="sample_id", how="inner")

    # Apply label sources if possible
    if spec.label_sources:
        labelled_ids = (
            annotations
            .filter(pl.col("label_source").is_in(spec.label_sources))
            .select("sample_id")
            .unique()
        )

        how = "inner" if spec.require_labels else "left"
        df = df.join(labelled_ids, on="sample_id", how=how)

    return df
