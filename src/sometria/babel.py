"""The BABEL label source: its key mapping, its splits, its labels, its vocabulary.

BABEL annotates AMASS sequences with action labels. Three things have to happen for
those labels to reach our catalog, and they are deliberately separate:

- map BABEL's sequence identifiers onto our sample ids (``babel_key`` / ``sample_key``);
- import split membership (``import_babel_splits``);
- import the labels themselves (``import_babel_annotations``) and the benchmark
  vocabulary (``import_babel_action_vocabulary``).

Splits and labels are separate because a sample can belong to a split whether or not
its labels are available -- BABEL withholds ``act_cat`` for its entire test split.

When a second label source arrives, the key mapping is the part that genuinely varies;
everything else here is table writing that ``sometria.catalog`` already owns. Do not
introduce an importer protocol before then.
"""

import json
from pathlib import Path
import re

import polars as pl

from sometria.catalog import (
    ANNOTATIONS,
    SPLITS,
    VOCABULARY,
    load_catalog,
    upsert_table,
)

# BABEL identifies a sequence by `feat_p`, e.g. `MPIHDM05/MPI_HDM05/dg/HDM_dg_03-11_03_120_poses.npz`.
# Our sample paths look like `HDM05/dg/HDM_dg_03-11_03_120_stageii.csv`. So: drop BABEL's duplicated
# second component, rename the dataset folder, drop the `_poses` / `_stageii` suffix, and normalize
# case and separators (AMASS mirrors differ on spaces/underscores/dashes).

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


def babel_key(feat_p: str) -> str:
    """Normalize a BABEL ``feat_p`` into the shared join key."""

    parts = Path(feat_p).with_suffix("").parts  # <dataset>/<dataset>/<subject>/<seq>
    return _key(parts[0], "/".join(parts[2:]))


def sample_key(path: str) -> str:
    """Normalize one of our catalog ``source_path`` values into the shared join key."""

    parts = Path(path).with_suffix("").parts  # <dataset>/<subject>/<seq>
    return _key(parts[0], "/".join(parts[1:]))


def _catalog_keys(output_root: str | Path, *columns: str) -> pl.DataFrame:
    """Return catalog rows with a ``babel_match_key`` column added.

    The normalization is lossy on purpose -- it strips case, spaces, dashes and
    underscores to reconcile two AMASS mirrors -- so two distinct samples can collapse
    onto one key. That never raises: the join simply hands both of them the same BABEL
    sequence, and ``upsert_table``'s keep-last then hides which one won. Warn loudly,
    because the symptom otherwise surfaces months later as a wrong split.
    """

    keys = load_catalog(output_root).select(
        "sample_id",
        *columns,
        pl.col("source_path")
        .map_elements(sample_key, return_dtype=pl.String)
        .alias("babel_match_key"),
    )

    collisions = keys.filter(pl.col("babel_match_key").is_duplicated())
    if not collisions.is_empty():
        examples = collisions["sample_id"].to_list()[:4]
        print(
            f"warning: {len(collisions)} sample paths collide after normalization "
            f"and will share BABEL rows, e.g. {examples}"
        )

    return keys


def _babel_sequences(babel_root: str | Path):
    """Yield every annotated sequence across BABEL's three split files."""

    babel_root = Path(babel_root)
    for split in ("train", "val", "test"):
        data = json.loads((babel_root / f"{split}.json").read_text())
        for sequence in data.values():
            yield split, sequence


def import_babel_splits(
    *,
    output_root: str | Path,
    babel_root: str | Path,
    source_dataset: str = "AMASS",
    split_set: str = "babel_official",
) -> pl.DataFrame:
    """Import BABEL's official split membership for already-imported AMASS rows.

    This records split membership only; labels are intentionally separate so future
    label sources can use the same catalog.
    """

    rows = [
        {
            "babel_match_key": babel_key(sequence["feat_p"]),
            "split_set": split_set,
            "split": split,
            "label_source": "BABEL",
            "babel_sid": sequence["babel_sid"],
        }
        for split, sequence in _babel_sequences(babel_root)
    ]

    splits = (
        _catalog_keys(output_root, "source_path")
        .join(pl.DataFrame(rows), on="babel_match_key", how="inner")
        .select("sample_id", "split_set", "split", "label_source", "babel_sid")
    )

    return upsert_table(output_root, SPLITS, splits, keys=["sample_id", "split_set"])


# BABEL carries three parallel vocabularies per segment: the annotator's free text, its
# processed form, and zero or more entries from the action taxonomy. They are kept as separate
# rows under `ontology` rather than as columns, so a downstream probe can ask for exactly one
# vocabulary without having to know the others exist.
BABEL_ONTOLOGIES = ("raw", "proc", "act_cat")


def _babel_annotation_rows(sequence: dict) -> list[dict]:
    """Flatten one BABEL sequence into annotation rows, one per (segment, ontology, label).

    ``seq_ann`` labels describe the whole take and are emitted spanning ``0..dur``.
    ``frame_ann`` is absent for roughly 40% of sequences and carries its own spans. ``act_cat``
    is null for every label in BABEL's test split, so an ontology can legitimately contribute
    no rows at all.
    """

    rows = []
    spans = [("sequence", sequence.get("seq_ann"), 0.0, sequence.get("dur"))]
    spans.append(("frame", sequence.get("frame_ann"), None, None))

    for label_type, annotation, default_start, default_end in spans:
        if not annotation:
            continue
        for label in annotation["labels"]:
            start = default_start if label_type == "sequence" else label.get("start_t")
            end = default_end if label_type == "sequence" else label.get("end_t")
            values = {
                "raw": [label.get("raw_label")],
                "proc": [label.get("proc_label")],
                "act_cat": label.get("act_cat") or [],
            }
            for ontology in BABEL_ONTOLOGIES:
                for value in values[ontology]:
                    if value is None:
                        continue
                    rows.append(
                        {
                            "babel_match_key": babel_key(sequence["feat_p"]),
                            "label_type": label_type,
                            "ontology": ontology,
                            "start_t": None if start is None else float(start),
                            "end_t": None if end is None else float(end),
                            "label": value,
                        }
                    )
    return rows


def import_babel_annotations(
    *,
    output_root: str | Path,
    babel_root: str | Path,
    label_source: str = "BABEL",
) -> pl.DataFrame:
    """Import BABEL action labels for already-imported samples.

    Split membership lives in ``splits.parquet`` and is imported separately by
    ``import_babel_splits``; this writes the labels themselves into ``annotations.parquet``
    so a sample can belong to a split whether or not its labels are available. Sequences that
    do not match a catalog row are dropped, as are labels BABEL withholds.
    """

    rows = []
    for _, sequence in _babel_sequences(babel_root):
        rows.extend(_babel_annotation_rows(sequence))

    if not rows:
        raise ValueError(f"No BABEL annotations parsed from {babel_root}.")

    annotations = (
        pl.DataFrame(rows)
        .join(_catalog_keys(output_root), on="babel_match_key", how="inner")
        .with_columns(pl.lit(label_source).alias("label_source"))
        .select("sample_id", "label_source", "label_type", "ontology", "start_t", "end_t", "label")
        .unique()
    )

    return upsert_table(
        output_root,
        ANNOTATIONS,
        annotations,
        keys=["sample_id", "label_source", "label_type", "ontology", "start_t", "end_t", "label"],
    )


def import_babel_action_vocabulary(
    *,
    output_root: str | Path,
    babel_root: str | Path,
    label_source: str = "BABEL",
    sizes: tuple[int, ...] = (60, 120, 150),
    filename: str = "action_label_2_idx.json",
) -> pl.DataFrame:
    """Import BABEL's action-category vocabulary, and its prefixes, for downstream evaluation.

    ``action_label_2_idx.json`` fixes the 150 action categories the BABEL benchmarks score and
    the integer each maps to, ordered by frequency. It is a much smaller set than the labels
    actually present: the corpus carries a long tail of ``act_cat`` values with no index here,
    which downstream tasks treat as out of scope rather than as extra classes.

    The 60- and 120-way benchmarks are index *prefixes* of that one list, not separate files.
    They are materialized as their own ``label_set`` rows -- ``babel_action_60`` and
    ``babel_action_120`` -- so a benchmark is named rather than carried around as a cutoff.
    The three sets cover 70.3%, 75.3% and 76.0% of ``act_cat`` frame-annotation rows
    (72.7%, 78.8% and 79.7% once sequence-level rows are counted too, as the downstream
    loader does).
    """

    mapping = json.loads((Path(babel_root) / filename).read_text())

    indices = sorted(int(i) for i in mapping.values())
    if indices != list(range(len(mapping))):
        raise ValueError(
            f"{filename} indices are not a contiguous 0..{len(mapping) - 1} range; "
            "downstream code assumes they can index a classifier head directly."
        )

    full = pl.DataFrame(
        {
            "label_source": [label_source] * len(mapping),
            "ontology": ["act_cat"] * len(mapping),
            "label": list(mapping.keys()),
            "label_index": [int(i) for i in mapping.values()],
        }
    )

    unknown = [size for size in sizes if size > len(mapping)]
    if unknown:
        raise ValueError(f"{filename} holds {len(mapping)} labels, cannot cut it at {unknown}")

    vocabulary = pl.concat(
        [
            full.filter(pl.col("label_index") < size).with_columns(
                pl.lit(f"babel_action_{size}").alias("label_set")
            )
            for size in sizes
        ]
    ).select("label_source", "label_set", "ontology", "label", "label_index")

    return upsert_table(
        output_root,
        VOCABULARY,
        vocabulary,
        keys=["label_source", "label_set", "label"],
    )
