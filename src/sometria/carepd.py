"""The CARE-PD label source: its UPDRS gait scores, its vocabulary, its official folds.

The CARE-PD torque corpus reaches us the same way every other corpus does -- as OpenSim
CSV, imported by ``sometria.preprocess``. What is specific to it is the label side:

- read UPDRS gait scores out of the released cohort pickles (``import_carepd_annotations``);
- fix the classes a probe scores (``import_carepd_gait_vocabulary``);
- import CARE-PD's own participant folds (``import_carepd_folds``).

Four of the nine cohorts carry ``UPDRS_GAIT``: 3DGait, BMCLab, PD-GaM and T-SDU-PD, for
2,952 of the 8,459 takes. The other five (DNE, E-LC, KUL-DT-T, T-LTC, T-SDU) import as
unlabelled motion here -- they carry medication, freezer and disease-status labels, which
are different ontologies and belong in their own importer when a probe needs them.

Splits are CARE-PD's, not ours. The release ships participant lists per cohort, and they
are participant lists rather than take lists for a reason: a patient contributes many
walks, so a take-level split scores patient identity instead of gait severity.

Reference: Adeli et al., "CARE-PD: A Multi-Site Anonymized Clinical Dataset for
Parkinson's Disease Gait Assessment", NeurIPS 2025.
https://huggingface.co/datasets/vida-adl/CARE-PD
"""

from pathlib import Path
import pickle

import polars as pl

from sometria.catalog import (
    ANNOTATIONS,
    SPLITS,
    VOCABULARY,
    load_catalog,
    stable_sample_id,
    upsert_table,
)

SOURCE_DATASET = "CARE-PD"
LABEL_SOURCE = "CARE-PD"

# MDS-UPDRS item 3.10 (gait). The scale runs 0-4; CARE-PD releases 0-3, because a patient
# scoring 4 cannot walk unassisted and so produces no gait capture.
ONTOLOGY = "updrs_gait"
LABEL_SET = "carepd_updrs_gait"
GAIT_SCORES = (0, 1, 2, 3)

# The cohorts whose released pickles carry a non-null UPDRS_GAIT.
UPDRS_COHORTS = ("3DGait", "BMCLab", "PD-GaM", "T-SDU-PD")

# CARE-PD's fixed train/eval split is not named uniformly across cohorts.
FIXED_FOLD_FILES = {
    "3DGait": "3DGait_fixed",
    "BMCLab": "BMCLab_fixed",
    "PD-GaM": "PD-GaM_authors_fixed",
    "T-SDU-PD": "T-SDU-PD_PD_fixed",
}


def sample_id_for(cohort: str, subject: str, walk: str) -> str:
    """Map one released ``(cohort, subject, walk)`` triple onto our catalog sample id.

    The torque export names a take ``<cohort>_canonical__<subject>__<walk>.csv``, flat, and
    ``import_opensim_csv_dataset`` keys on that relative path.
    """

    return stable_sample_id(SOURCE_DATASET, f"{cohort}_canonical__{subject}__{walk}.csv")


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _cohort_pickle(carepd_root: str | Path, cohort: str) -> Path:
    """Locate one cohort's canonicalized pickle under a CARE-PD release root."""

    root = Path(carepd_root)
    for candidate in (
        root / "Canonicalized_SMPL_pickles" / f"{cohort}_canonical.pkl",
        root / f"{cohort}_canonical.pkl",
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No {cohort}_canonical.pkl under {root}.")


def _updrs_rows(carepd_root: str | Path, cohorts: tuple[str, ...]) -> pl.DataFrame:
    """Read ``(sample_id, cohort, subject, label)`` for every take carrying a gait score.

    Only the scalar fields are kept. ``pose``/``trans`` are the release's SMPL encoding and
    would duplicate, at a different rate and channel layout, the tensors the OpenSim import
    already wrote from the same take.
    """

    rows = []
    for cohort in cohorts:
        cohort_data = _load_pickle(_cohort_pickle(carepd_root, cohort))
        for subject, walks in cohort_data.items():
            for walk, record in walks.items():
                score = record.get("UPDRS_GAIT")
                if score is None:
                    continue
                rows.append(
                    {
                        "sample_id": sample_id_for(cohort, subject, walk),
                        "cohort": cohort,
                        "subject": str(subject),
                        "label": str(int(score)),
                    }
                )

    if not rows:
        raise ValueError(f"No UPDRS_GAIT scores found under {carepd_root} for {cohorts}.")

    return pl.DataFrame(rows)


def _matched(rows: pl.DataFrame, output_root: str | Path, *columns: str) -> pl.DataFrame:
    """Join released rows onto imported catalog rows, warning about takes with no CSV.

    A miss is not an error -- the release and the torque export were produced separately,
    and the export drops takes that failed inverse dynamics -- but a silent miss would
    surface later as a probe quietly scoring fewer samples than the paper it cites.
    """

    catalog = load_catalog(output_root).select("sample_id", *columns)
    matched = rows.join(catalog, on="sample_id", how="inner")

    missing = len(rows) - len(matched)
    if missing:
        dropped = rows.join(catalog, on="sample_id", how="anti")
        print(
            f"warning: {missing} of {len(rows)} released takes have no imported CSV and are "
            f"dropped, by cohort: {dropped.group_by('cohort').len().sort('cohort').to_dicts()}"
        )

    return matched


def import_carepd_annotations(
    *,
    output_root: str | Path,
    carepd_root: str | Path,
    cohorts: tuple[str, ...] = UPDRS_COHORTS,
    label_source: str = LABEL_SOURCE,
) -> pl.DataFrame:
    """Import CARE-PD UPDRS gait scores for already-imported CARE-PD rows.

    One score per take, so each becomes a single ``sequence`` annotation spanning the whole
    sample. The span is read from the catalog's ``duration`` rather than from the release's
    own frame count: the import resampled to 60 Hz, and the downstream loader slices windows
    in seconds against the resampled tensor.
    """

    annotations = (
        _matched(_updrs_rows(carepd_root, cohorts), output_root, "duration")
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
    scores: tuple[int, ...] = GAIT_SCORES,
) -> pl.DataFrame:
    """Fix the gait-severity classes a CARE-PD probe scores, and their head order.

    The scores are declared rather than read off the data, so a cohort subset that happens to
    contain no 3s still produces a four-way head, and so the label index is the clinical score
    itself instead of a rank that shifts with which cohorts were imported.
    """

    vocabulary = pl.DataFrame(
        {
            "label_source": [label_source] * len(scores),
            "label_set": [label_set] * len(scores),
            "ontology": [ONTOLOGY] * len(scores),
            "label": [str(score) for score in scores],
            "label_index": list(range(len(scores))),
        }
    )

    return upsert_table(
        output_root, VOCABULARY, vocabulary, keys=["label_source", "label_set", "label"]
    )


def _fold_participants(carepd_root: str | Path, filename: str) -> dict[int, dict[str, list[str]]]:
    """Read one released fold file: ``{fold: {"train": [subject], "eval": [subject]}}``."""

    path = Path(carepd_root) / "folds" / "UPDRS_Datasets" / f"{filename}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No fold file {path}.")
    return _load_pickle(path)


def import_carepd_folds(
    *,
    output_root: str | Path,
    carepd_root: str | Path,
    cohorts: tuple[str, ...] = UPDRS_COHORTS,
    n_folds: int = 6,
    fixed_split_set: str = "carepd_fixed",
    fold_split_set: str = "carepd_6fold",
) -> pl.DataFrame:
    """Import CARE-PD's official participant splits: the fixed train/eval split and 6-fold CV.

    The released files list participant ids per cohort, so each cohort is resolved against its
    own takes and the four are then pooled into one split set -- fold 3 means "fold 3 of every
    cohort", which is what a multi-cohort probe evaluates. Splits keep the release's own
    ``train``/``eval`` names rather than being renamed to ``val``/``test``: CARE-PD publishes
    one held-out side, and calling it either would assert something the release does not.

    Leave-one-subject-out is deliberately not materialized. The release ships it as
    ``<cohort>_<n>fold_participants.pkl``, and it would be 110 single-subject split sets for a
    protocol nothing here runs yet.
    """
    # ponytail: fixed + 6-fold only. Add LOSO by looping the same helper over the
    # <cohort>_<n>fold_participants files when a probe actually reports it.

    labelled = _matched(_updrs_rows(carepd_root, cohorts), output_root)

    frames = []
    for cohort in cohorts:
        takes = labelled.filter(pl.col("cohort") == cohort).select("sample_id", "subject")

        sources = {fixed_split_set: _fold_participants(carepd_root, FIXED_FOLD_FILES[cohort])}
        for fold, members in _fold_participants(
            carepd_root, f"{cohort}_{n_folds}fold_participants"
        ).items():
            sources[f"{fold_split_set}_{fold}"] = {1: members}

        for split_set, folds in sources.items():
            for members in folds.values():
                assignment = {
                    subject: split
                    for split, subjects in members.items()
                    for subject in map(str, subjects)
                }
                unknown = set(assignment) - set(takes["subject"])
                if unknown:
                    print(
                        f"warning: {cohort} {split_set} names {len(unknown)} participants with no "
                        f"imported take, e.g. {sorted(unknown)[:4]}"
                    )
                frames.append(
                    takes
                    .filter(pl.col("subject").is_in(list(assignment)))
                    .with_columns(
                        pl.col("subject").replace_strict(assignment).alias("split"),
                        pl.lit(split_set).alias("split_set"),
                        pl.lit(LABEL_SOURCE).alias("label_source"),
                    )
                    .select("sample_id", "split_set", "split", "label_source")
                )

    return upsert_table(
        output_root, SPLITS, pl.concat(frames), keys=["sample_id", "split_set"]
    )
