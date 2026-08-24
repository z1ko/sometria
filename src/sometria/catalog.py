"""The processed data layout: where things live, and which samples constitute a view.

This module owns ``data/processed/`` in both directions. Reads materialize sample
views for training and analysis; writes place tensors and upsert tables. No other
module builds a path into the processed root or writes a table -- callers name
*what* they want and never *where* it lives.

The root is deliberately split into separate concerns:

- ``tables/motion_catalog.parquet`` describes one row per motion sample.
- ``tables/annotations.parquet`` optionally describes labels over samples.
- ``tables/splits.parquet`` describes named dataset views such as BABEL's
  official split or a custom pretraining split.
- ``tables/label_vocabulary.parquet`` fixes which labels a benchmark scores.
- ``motions/`` holds one feature tensor per sample.
- ``stats/`` holds normalization artifacts, keyed by representation.

Filtering rules live here rather than in training code so they stay in one place
as more datasets and label sources are added: construct a ``MotionViewSpec`` and
call ``build_motion_view``.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import polars as pl

CATALOG = "motion_catalog.parquet"
ANNOTATIONS = "annotations.parquet"
SPLITS = "splits.parquet"
VOCABULARY = "label_vocabulary.parquet"

MOTIONS_DIR = "motions"
TABLES_DIR = "tables"
STATS_DIR = "stats"


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
    min_frames: int | None = None           # drop samples too short to fill a window


def stable_sample_id(source_dataset: str, source_path: str) -> str:
    """Return the stable key used to join catalogs, labels, and splits.

    Derived from the source path, never from a row number, so re-importing a
    corpus lands on the same ids and ``upsert_table`` replaces rather than
    duplicates.
    """

    return f"{source_dataset}:{source_path}"


def _table_path(root: str | Path, name: str) -> Path:
    """Return the path to a named catalog table under a processed data root."""

    return Path(root) / TABLES_DIR / name


def _safe_name(sample_id: str) -> str:
    """Return a compact deterministic filename stem for a sample id."""

    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]


def motion_path(root: str | Path, source_dataset: str, sample_id: str) -> Path:
    """Return the path for one sample's feature tensor, creating its directory.

    Sample ids contain separators and dataset-specific punctuation, so the stem is
    a hash rather than the id itself. That is the only reason the mapping is not
    reversible, and it is why nothing outside this module should construct it.
    """

    path = Path(root) / MOTIONS_DIR / source_dataset.lower() / f"{_safe_name(sample_id)}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalization_path(root: str | Path, representation: str, name: str) -> Path:
    """Return the path for a normalization stats artifact, creating its directory.

    Keyed by representation because statistics computed against one channel layout
    are meaningless under another.
    """

    path = Path(root) / STATS_DIR / representation / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def upsert_table(root: str | Path, name: str, df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Insert or replace rows in a named table using ``keys`` as identity.

    Every write into ``tables/`` goes through here. New rows win over old ones with
    the same key, so re-running an importer is idempotent rather than additive.
    """

    path = _table_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pl.read_parquet(path)
        df = pl.concat([old, df], how="diagonal_relaxed")
        df = df.unique(subset=keys, keep="last")

    df.write_parquet(path)
    return df


def load_catalog(root: str | Path) -> pl.DataFrame:
    """Load the required sample catalog table."""

    return pl.read_parquet(_table_path(root, CATALOG))


def load_annotations(root: str | Path) -> pl.DataFrame:
    """Load optional annotation rows, or an empty table with the expected schema.

    Annotations are modeled separately from samples so different label sources
    can coexist. For example, AMASS samples can have BABEL labels today, and a
    future MotionX import can add another ontology without changing the motion
    catalog schema.
    """

    path = _table_path(root, ANNOTATIONS)
    if path.exists():
        return pl.read_parquet(path)

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

    path = _table_path(root, SPLITS)
    if path.exists():
        return pl.read_parquet(path)

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

    path = _table_path(root, VOCABULARY)
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

    # Samples shorter than a training window would otherwise be zero-padded by the
    # collate, and padding scores as motionless -- so motion-aware masking keeps it,
    # spending the context budget on frames that are not there.
    if spec.min_frames is not None:
        df = df.filter(pl.col("n_frames") >= spec.min_frames)

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


def create_pretrain_split(
    *,
    output_root: str | Path,
    split_set: str = "pretrain_v1",
) -> pl.DataFrame:
    """Create the default self-supervised pretraining split.

    The current policy includes BABEL train samples plus AMASS samples with no
    BABEL official split membership. BABEL validation and test samples are kept
    out so they remain usable as held-out views.

    This lives beside ``build_motion_view`` rather than in ``babel``: persisting a
    view and materializing one are the same question asked in two directions, and
    the policy is really "hold out whatever an evaluation benchmark reserved" --
    which stops being BABEL-shaped as soon as a second label source exists.
    """

    catalog = load_catalog(output_root)
    splits = load_splits(output_root)

    babel_train_ids = splits.filter(
        (pl.col("split_set") == "babel_official")
        & (pl.col("split") == "train")
    ).select("sample_id")

    babel_any_ids = splits.filter(
        pl.col("split_set") == "babel_official"
    ).select("sample_id").unique()

    unlabelled_ids = catalog.join(babel_any_ids, on="sample_id", how="anti").select("sample_id")

    pretrain = (
        pl.concat([babel_train_ids, unlabelled_ids])
        .unique()
        .with_columns(
            pl.lit(split_set).alias("split_set"),
            pl.lit("train").alias("split"),
            pl.lit(None).cast(pl.String).alias("label_source"),
            pl.lit(None).cast(pl.Int64).alias("babel_sid"),
        )
    )

    return upsert_table(
        output_root, 
        SPLITS, 
        pretrain, 
        keys=["sample_id", "split_set"]
    )
