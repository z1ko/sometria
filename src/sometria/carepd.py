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

SIX_FOLD = 6

# The paper's four evaluation protocols (S4.2), plus the two source files they are read from.
# "pooled" is ours -- see import_carepd_folds.
PROTOCOLS = ("fixed", "6fold", "loso", "cross", "lodo", "mida", "pooled")

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


def _loso_fold_file(carepd_root: str | Path, cohort: str) -> str:
    """Name of the cohort's leave-one-subject-out file.

    The release encodes the fold count in the filename and it equals the participant count,
    so it differs per cohort (43 / 23 / 30 / 14). Globbed rather than tabulated: a hardcoded
    map is a second place to be wrong when a cohort gains a participant.
    """

    folds = Path(carepd_root) / "folds" / "UPDRS_Datasets"
    candidates = [
        path.stem
        for path in folds.glob(f"{cohort}_*fold_participants.pkl")
        if not path.stem.startswith(f"{cohort}_{SIX_FOLD}fold")
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one leave-one-subject-out file for {cohort} in {folds}, "
            f"found {sorted(candidates)}"
        )
    return candidates[0]


def _rows(takes: pl.DataFrame, split_set: str, assignment: dict[str, str]) -> pl.DataFrame:
    """Label one cohort's takes train/eval from a participant assignment."""

    return (
        takes
        .filter(pl.col("subject").is_in(list(assignment)))
        .with_columns(
            pl.col("subject").replace_strict(assignment).alias("split"),
            pl.lit(split_set).alias("split_set"),
            pl.lit(LABEL_SOURCE).alias("label_source"),
        )
        .select("sample_id", "split_set", "split", "label_source")
    )


def _whole(takes: pl.DataFrame, split_set: str, split: str) -> pl.DataFrame:
    """Put every take of a selection on one side of a split."""

    return takes.select("sample_id").with_columns(
        pl.lit(split_set).alias("split_set"),
        pl.lit(split).alias("split"),
        pl.lit(LABEL_SOURCE).alias("label_source"),
    )


def import_carepd_folds(
    *,
    output_root: str | Path,
    carepd_root: str | Path,
    cohorts: tuple[str, ...] = UPDRS_COHORTS,
    protocols: tuple[str, ...] = PROTOCOLS,
) -> pl.DataFrame:
    """Materialize CARE-PD's published evaluation protocols as named split sets.

    The paper (S4.2) evaluates severity estimation four ways, and every one of them is
    participant-level: a patient contributes many walks, so a take-level split scores patient
    identity instead of gait severity. Only the first two need the released participant lists;
    the rest are cohort algebra over them.

    ==================  ===========================================  =====
    ``split_set``       train / eval                                 count
    ==================  ===========================================  =====
    ``..._fixed``       the release's fixed split, per cohort            4
    ``..._6fold_<k>``   6-fold participant CV, per cohort               24
    ``..._loso_<k>``    leave-one-subject-out, per cohort              110
    ``cross_<x>_to_<y>``  all of cohort x / all of cohort y             12
    ``lodo_<x>``        the other three cohorts / all of x               4
    ``mida_<x>_loso_<k>``  x's LOSO train + the others / x's subject    110
    ==================  ===========================================  =====

    Two of these deserve a note. LODO trains on no data from the target at all, so evaluating
    it on the whole cohort is the same in aggregate as pooling that cohort's LOSO eval folds --
    which is what keeps it directly comparable to MIDA (paper Fig. 4). And MIDA is LOSO with
    the other cohorts bolted onto the training side, so its fold count and held-out subjects
    are identical to plain LOSO by construction; that is the comparison it exists to make.

    ``carepd_fixed`` and ``carepd_6fold_<k>`` are ours, not the paper's: the four cohorts'
    splits unioned into one multi-site benchmark. Cheap to emit and a reasonable thing for a
    pretrained encoder to be scored on, but every cohort then appears on both sides, and
    cohort priors differ enough (T-SDU-PD is 44% score-2, PD-GaM 15%) that a probe can score
    by recognizing the capture site. Do not table those numbers against the paper's.

    Splits keep the release's own ``train``/``eval`` names rather than being renamed to
    ``val``/``test``: CARE-PD publishes one held-out side, and calling it either would assert
    something the release does not.
    """

    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocol(s) {sorted(unknown)}; known: {list(PROTOCOLS)}")

    labelled = _matched(_updrs_rows(carepd_root, cohorts), output_root)
    takes = {
        cohort: labelled.filter(pl.col("cohort") == cohort).select("sample_id", "subject")
        for cohort in cohorts
    }
    # Read once: LOSO is up to 43 folds per cohort and MIDA reuses the same lists.
    loso = {
        cohort: _fold_participants(carepd_root, _loso_fold_file(carepd_root, cohort))
        for cohort in cohorts
        if {"loso", "mida"} & set(protocols)
    }

    frames: list[pl.DataFrame] = []
    # Per-cohort frames kept aside so the pooled sets are a relabelling of exactly what the
    # per-cohort sets contain, rather than a second read of the same pickles.
    poolable: dict[str, list[pl.DataFrame]] = {}

    def per_cohort(prefix: str, folds: dict, cohort: str, pooled: str | None = None) -> None:
        for fold, members in folds.items():
            assignment = {
                subject: split
                for split, subjects in members.items()
                for subject in map(str, subjects)
            }
            unplaced = set(assignment) - set(takes[cohort]["subject"])
            if unplaced:
                print(
                    f"warning: {cohort} {prefix} names {len(unplaced)} participants with no "
                    f"imported take, e.g. {sorted(unplaced)[:4]}"
                )
            name = f"{prefix}_{fold}" if len(folds) > 1 else prefix
            rows = _rows(takes[cohort], name, assignment)
            frames.append(rows)
            if pooled is not None:
                key = f"{pooled}_{fold}" if len(folds) > 1 else pooled
                poolable.setdefault(key, []).append(rows)

    for cohort in cohorts:
        if "fixed" in protocols:
            per_cohort(
                f"carepd_{cohort}_fixed",
                _fold_participants(carepd_root, FIXED_FOLD_FILES[cohort]),
                cohort,
                pooled="carepd_fixed",
            )
        if "6fold" in protocols:
            per_cohort(
                f"carepd_{cohort}_6fold",
                _fold_participants(carepd_root, f"{cohort}_{SIX_FOLD}fold_participants"),
                cohort,
                pooled="carepd_6fold",
            )
        if "loso" in protocols:
            per_cohort(f"carepd_{cohort}_loso", loso[cohort], cohort)

        others = pl.concat([takes[other] for other in cohorts if other != cohort])

        if "cross" in protocols:
            for target in cohorts:
                if target == cohort:
                    continue
                name = f"carepd_cross_{cohort}_to_{target}"
                frames += [_whole(takes[cohort], name, "train"), _whole(takes[target], name, "eval")]

        if "lodo" in protocols:
            name = f"carepd_lodo_{cohort}"
            frames += [_whole(others, name, "train"), _whole(takes[cohort], name, "eval")]

        if "mida" in protocols:
            for fold, members in loso[cohort].items():
                name = f"carepd_mida_{cohort}_loso_{fold}"
                assignment = {
                    subject: split
                    for split, subjects in members.items()
                    for subject in map(str, subjects)
                }
                # The other cohorts join the training side wholesale; the held-out subject is
                # the same one plain LOSO holds out, which is what makes the pair comparable.
                frames += [_rows(takes[cohort], name, assignment), _whole(others, name, "train")]

    # Ours, not the paper's -- see the docstring.
    if "pooled" in protocols:
        for name, parts in poolable.items():
            frames.append(pl.concat(parts).with_columns(pl.lit(name).alias("split_set")))

    return upsert_table(output_root, SPLITS, pl.concat(frames), keys=["sample_id", "split_set"])
