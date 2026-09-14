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
import re

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
    representation: str | None = None       # "opensim_sincos_log_vel_acc_tau_v2"
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


# One take reaches us through more than one path: the OpenSim conversion writes
# `DFaust67/50002/...csv`, the AMASS release ships `DFaust/50002/...npz`, and BABEL names the
# same sequence `DFaust67/DFaust_67/50002/..._poses.npz`. `sample_key` is the one place that
# reconciles them -- it names a *take*, independent of which corpus or format carries it, so
# labels, splits and the two arms of a feature comparison all join on the same string.

# BABEL dataset folder -> folder name used in our preprocessed tree
DATASET_ALIASES = {
    "MPIHDM05": "HDM05",
    "DFaust67": "DFaust",
    "Transitionsmocap": "Transitions",
    "MPImosh": "MoSh",
    "TCDhandMocap": "TCDHands",
    "MPILimits": "PosePrior",
    "SSMsynced": "SSM",
    "EyesJapanDataset": "Eyes_Japan_Dataset",
}


def _key(dataset: str, rest: str) -> str:
    rest = re.sub(r"_(poses|stageii)$", "", rest, flags=re.IGNORECASE)
    return f"{DATASET_ALIASES.get(dataset, dataset)}/{re.sub(r'[^a-z0-9/]', '', rest.lower())}"


def sample_key(path: str) -> str:
    """Normalize one of our catalog ``source_path`` values into the shared join key.

    Lossy on purpose -- case, spaces, dashes and underscores go -- because that is what it
    takes to reconcile two AMASS mirrors. Two genuinely distinct takes can therefore collapse
    onto one key; callers that join on it check for duplicates.
    """

    parts = Path(path).with_suffix("").parts  # <dataset>/<subject>/<seq>
    return _key(parts[0], "/".join(parts[1:]))


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

    # Samples under different representations have different feature shapes, and a view is
    # consumed as one stacked tensor. Mixing them fails deep inside normalization or the
    # collate; naming the representation fails here, by name.
    if spec.representation is not None:
        df = df.filter(pl.col("representation") == spec.representation)

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

    # Polars joins are parallel and do not preserve row order, so two calls with the same
    # spec return the same *set* of samples in a different sequence. Everything downstream
    # indexes positionally -- LabelledWindows draws a random crop for sample i, the
    # pretraining loader shuffles an index range, normalization accumulates in list order --
    # so without this, seeding the RNG cannot make a run reproducible: the seed picks the
    # same index, and the index points at a different motion. Measured on the BABEL train
    # view, row 0 differed between consecutive calls.
    return df.sort("sample_id")


def create_paired_split(
    *,
    output_root: str | Path,
    split_set: str,
    source_split_set: str,
    arms: tuple[str, ...],
    splits: tuple[str, ...] = (),
    exclude_broken: bool = True,
    min_frames: int | None = None,
) -> pl.DataFrame:
    """Copy a split onto several corpora, keeping only takes every one of them has.

    Two imports of the same corpus are never quite the same set: the OpenSim conversion
    kept 1 of LARa's 452 takes and 63 of DanceDB's 151, a handful of AMASS ``.npz`` are
    shape fits rather than takes, and the broken-torque filter fires on one arm only
    because the other has no torque. Training the arms on their own maximal sets then
    compares corpora as much as it compares features. This intersects them first.

    ``source_split_set`` supplies the membership policy and the train/val/test labels --
    typically ``pretrain_v1`` or ``babel_official``, which name samples in one arm only;
    every arm inherits the label of the take. ``exclude_broken`` and ``min_frames`` are
    applied *here*, not just at view time, because a take dropped from one arm's view for
    either reason has to leave the other arm too.
    """

    catalog = load_catalog(output_root).filter(pl.col("source_dataset").is_in(arms))

    if exclude_broken and "broken" in catalog.columns:
        catalog = catalog.filter(~pl.col("broken"))
    if min_frames is not None:
        catalog = catalog.filter(pl.col("n_frames") >= min_frames)

    takes = catalog.select(
        "sample_id",
        "source_dataset",
        pl.col("source_path").map_elements(sample_key, return_dtype=pl.String).alias("take"),
    )

    paired = (
        takes.group_by("take")
        .agg(pl.col("source_dataset").n_unique().alias("n_arms"))
        .filter(pl.col("n_arms") == len(arms))
        .select("take")
    )

    source = load_splits(output_root).filter(pl.col("split_set") == source_split_set)
    if splits:
        source = source.filter(pl.col("split").is_in(splits))

    # One split label per take, not per sample: the source split names one arm's ids, and a
    # lossy key can collapse two of them, so sort first and the pick stops being arbitrary.
    labels = (
        source.join(takes.select("sample_id", "take"), on="sample_id", how="inner")
        .sort("sample_id")
        .unique(subset=["take"], keep="first")
        .select("take", "split")
    )

    rows = (
        takes.join(paired, on="take", how="semi")
        .join(labels, on="take", how="inner")
        .select(
            "sample_id",
            pl.lit(split_set).alias("split_set"),
            "split",
            pl.lit(None).cast(pl.String).alias("label_source"),
            pl.lit(None).cast(pl.Int64).alias("babel_sid"),
        )
    )

    return upsert_table(output_root, SPLITS, rows, keys=["sample_id", "split_set"])


def create_pretrain_split(
    *,
    output_root: str | Path,
    split_set: str = "pretrain_v1",
    source_datasets: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Create the default self-supervised pretraining split.

    The current policy includes BABEL train samples plus samples with no BABEL
    official split membership. BABEL validation and test samples are kept out so
    they remain usable as held-out views.

    ``source_datasets`` restricts the split to named corpora. Leaving it empty means
    "every corpus in the catalog", which is right while every import is pretraining
    data and wrong the moment one is an evaluation corpus: CARE-PD is imported for a
    clinical probe, and pretraining on it would train on the probe's own subjects.
    Name the corpora rather than relying on the default once such a corpus exists.

    This lives beside ``build_motion_view`` rather than in ``babel``: persisting a
    view and materializing one are the same question asked in two directions, and
    the policy is really "hold out whatever an evaluation benchmark reserved" --
    which stops being BABEL-shaped as soon as a second label source exists.
    """

    catalog = load_catalog(output_root)
    splits = load_splits(output_root)

    if source_datasets:
        catalog = catalog.filter(pl.col("source_dataset").is_in(source_datasets))

    babel_train_ids = splits.filter(
        (pl.col("split_set") == "babel_official")
        & (pl.col("split") == "train")
    ).select("sample_id").join(catalog.select("sample_id"), on="sample_id", how="semi")

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
