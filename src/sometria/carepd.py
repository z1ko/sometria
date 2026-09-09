"""The CARE-PD label source: MoCHA's gait-severity labels, its vocabulary, its splits.

The CARE-PD torque corpus reaches us the same way every other corpus does -- as OpenSim
CSV, imported by ``sometria.preprocess``. What is specific to it is the label side, and
that is what lives here:

- map MoCHA's sequence identifiers onto our sample ids (``sample_id_for``);
- import the labels themselves (``import_carepd_annotations``) and their vocabulary
  (``import_carepd_gait_vocabulary``);
- import subject-disjoint evaluation splits (``import_carepd_subject_splits``).

MoCHA labels only two of the nine CARE-PD subsets (3DGait and BMCLab, 871 of 8459
samples). The rest import as unlabelled motion, which is why the labels are a separate
table rather than a catalog column.

Splits are subject-disjoint rather than random: CARE-PD records several walks per
patient, so a random split puts the same subject on both sides and the probe scores
subject identity instead of gait severity.
"""

from pathlib import Path

import numpy as np
import polars as pl

from sometria.catalog import (
    ANNOTATIONS,
    SPLITS,
    VOCABULARY,
    load_catalog,
    stable_sample_id,
    upsert_table,
)

# MoCHA names a sequence `<subset>_canonical__<subject>__<take>`; the CSV it was built
# from is that same stem with a `.csv` suffix, sitting flat in the torque export.
SOURCE_DATASET = "CARE-PD"
LABEL_SOURCE = "CARE-PD"

# MDS-UPDRS item 3.10 (gait) scores 0-4. MoCHA carries 0-3; 4 (severe, cannot walk
# unassisted) is absent from the corpus because those patients produce no gait capture.
ONTOLOGY = "updrs_gait"
LABEL_SET = "carepd_updrs_gait"
GAIT_SCORES = (0, 1, 2, 3)


def sample_id_for(mocha_sample_id: str) -> str:
    """Map a MoCHA sequence identifier onto our catalog sample id."""

    return stable_sample_id(SOURCE_DATASET, f"{mocha_sample_id}.csv")


def _mocha_rows(mocha_root: str | Path) -> pl.DataFrame:
    """Read every MoCHA sequence into ``sample_id`` / ``subject`` / ``label`` rows.

    Only the scalar metadata is read; the ``x`` pose array is MoCHA's own feature
    encoding and would duplicate, at a different sample rate and channel layout, the
    tensors ``import_opensim_csv_dataset`` already wrote from the same source CSV.
    """

    files = sorted(Path(mocha_root).glob("sequences/*.npz"))
    if not files:
        raise FileNotFoundError(f"No MoCHA sequences under {mocha_root}/sequences.")

    rows = []
    for path in files:
        with np.load(path, allow_pickle=True) as payload:
            mocha_id = str(payload["sample_id"])
            rows.append(
                {
                    "sample_id": sample_id_for(mocha_id),
                    "mocha_sample_id": mocha_id,
                    "subject": f"{payload['dataset_id']}__{payload['subject_id']}",
                    "label": str(int(payload["label"])),
                }
            )

    return pl.DataFrame(rows)


def _matched(mocha: pl.DataFrame, output_root: str | Path, *columns: str) -> pl.DataFrame:
    """Join MoCHA rows onto imported catalog rows, warning about sequences with no CSV.

    A miss is not an error -- MoCHA and the torque export were produced separately, and
    the export drops takes that failed inverse dynamics -- but a silent miss would show
    up later as a probe quietly scoring fewer samples than its paper.
    """

    catalog = load_catalog(output_root).select("sample_id", *columns)
    matched = mocha.join(catalog, on="sample_id", how="inner")

    missing = len(mocha) - len(matched)
    if missing:
        examples = (
            mocha.join(catalog, on="sample_id", how="anti")["mocha_sample_id"].to_list()[:4]
        )
        print(
            f"warning: {missing} of {len(mocha)} MoCHA sequences have no imported CSV "
            f"and are dropped, e.g. {examples}"
        )

    return matched


def import_carepd_annotations(
    *,
    output_root: str | Path,
    mocha_root: str | Path,
    label_source: str = LABEL_SOURCE,
) -> pl.DataFrame:
    """Import MoCHA gait-severity labels for already-imported CARE-PD rows.

    One label per take, so each becomes a single ``sequence`` annotation spanning the
    whole sample. The span is read from the catalog's ``duration`` rather than from
    MoCHA's own frame count: the import resampled to 60 Hz, and the downstream loader
    slices windows in seconds against the resampled tensor.
    """

    annotations = (
        _matched(_mocha_rows(mocha_root), output_root, "duration")
        .with_columns(
            pl.lit(label_source).alias("label_source"),
            pl.lit("sequence").alias("label_type"),
            pl.lit(ONTOLOGY).alias("ontology"),
            pl.lit(0.0).alias("start_t"),
            pl.col("duration").cast(pl.Float64).alias("end_t"),
        )
        .select("sample_id", "label_source", "label_type", "ontology", "start_t", "end_t", "label")
    )

    return upsert_table(
        output_root,
        ANNOTATIONS,
        annotations,
        keys=["sample_id", "label_source", "label_type", "ontology", "start_t", "end_t", "label"],
    )


def import_carepd_gait_vocabulary(
    *,
    output_root: str | Path,
    label_source: str = LABEL_SOURCE,
    label_set: str = LABEL_SET,
) -> pl.DataFrame:
    """Fix the gait-severity classes a CARE-PD probe scores, and their head order.

    The scores are declared here rather than read off the data so an import that happens
    to contain no 3s still produces a four-way head, and so the label index is the score
    itself instead of a rank that shifts with the corpus.
    """

    vocabulary = pl.DataFrame(
        {
            "label_source": [label_source] * len(GAIT_SCORES),
            "label_set": [label_set] * len(GAIT_SCORES),
            "ontology": [ONTOLOGY] * len(GAIT_SCORES),
            "label": [str(score) for score in GAIT_SCORES],
            "label_index": list(GAIT_SCORES),
        }
    )

    return upsert_table(
        output_root, VOCABULARY, vocabulary, keys=["label_source", "label_set", "label"]
    )


def import_carepd_subject_splits(
    *,
    output_root: str | Path,
    mocha_root: str | Path,
    split_set: str = "carepd_subject",
    fractions: tuple[float, float] = (0.7, 0.15),
    seed: int = 0,
) -> pl.DataFrame:
    """Import subject-disjoint train/val/test splits over the labelled CARE-PD samples.

    CARE-PD publishes no official split, so this is ours and it is deterministic: the
    subjects are shuffled under ``seed`` and cut at ``fractions``, then every take of a
    subject follows that subject. Assigning takes instead of subjects would leak -- the
    same patient walks several times, and a probe can recognise the patient far more
    easily than the gait score.

    Subjects are namespaced by subset, because 3DGait and BMCLab number their subjects
    independently and would otherwise collide.
    """

    train_fraction, val_fraction = fractions
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError(f"fractions must leave a non-empty test split, got {fractions}")

    labelled = _matched(_mocha_rows(mocha_root), output_root)

    subjects = sorted(labelled["subject"].unique().to_list())
    order = np.random.default_rng(seed).permutation(len(subjects))
    n_train = int(len(subjects) * train_fraction)
    n_val = int(len(subjects) * val_fraction)

    assignment = {}
    for rank, index in enumerate(order):
        split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
        assignment[subjects[index]] = split

    splits = (
        labelled
        .with_columns(
            pl.col("subject").replace_strict(assignment).alias("split"),
            pl.lit(split_set).alias("split_set"),
            pl.lit(LABEL_SOURCE).alias("label_source"),
        )
        .select("sample_id", "split_set", "split", "label_source")
    )

    return upsert_table(output_root, SPLITS, splits, keys=["sample_id", "split_set"])
