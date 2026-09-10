"""Checks for the two pieces of CARE-PD ingest that can silently produce wrong training data:
subset parsing from a flat export, and the participant folds the release ships."""

import pickle

import polars as pl
import pytest

from sometria.carepd import import_carepd_annotations, import_carepd_folds, sample_id_for
from sometria.catalog import CATALOG, create_pretrain_split, upsert_table
from sometria.preprocess import _source_subset

COHORTS = ("3DGait", "BMCLab")


def test_source_subset_handles_both_layouts():
    # nested (AMASS, MotionX): leading directory
    assert _source_subset("CMU/01/01_01_stageii.csv") == "CMU"
    assert _source_subset("idea400/subset_0000/take_1.csv") == "idea400"
    # flat (CARE-PD): filename prefix, not the whole filename
    assert _source_subset("BMCLab_canonical__SUB01__SUB01_off_walk_13.csv") == "BMCLab_canonical"
    assert _source_subset("3DGait_canonical__0__vid0073_0055.csv") == "3DGait_canonical"
    # flat with no subset marker degrades to the old behaviour rather than crashing
    assert _source_subset("take.csv") == "take.csv"


def _subjects(cohort):
    # 3DGait numbers participants, BMCLab uses SUBnn -- both appear as bare strings in the
    # released fold files, so the importer must not assume either shape
    return [str(i) for i in range(6)] if cohort == "3DGait" else [f"SUB{i:02d}" for i in range(6)]


def _sequences():
    return [
        (cohort, subject, f"walk_{take}", (index + take) % 4)
        for cohort in COHORTS
        for index, subject in enumerate(_subjects(cohort))
        for take in range(3)
    ]


def _write_release(root, sequences):
    """Write a miniature CARE-PD release: cohort pickles plus fixed and 6-fold participants."""

    pickles = root / "Canonicalized_SMPL_pickles"
    folds = root / "folds" / "UPDRS_Datasets"
    pickles.mkdir(parents=True)
    folds.mkdir(parents=True)

    for cohort in COHORTS:
        cohort_data = {}
        for c, subject, walk, score in sequences:
            if c == cohort:
                cohort_data.setdefault(subject, {})[walk] = {
                    "pose": None,
                    "fps": 30,
                    "UPDRS_GAIT": score,
                    "medication": None,
                    "other": None,
                }
        (pickles / f"{cohort}_canonical.pkl").write_bytes(pickle.dumps(cohort_data))

        subjects = _subjects(cohort)
        fixed = {1: {"train": subjects[:4], "eval": subjects[4:]}}
        name = "3DGait_fixed" if cohort == "3DGait" else "BMCLab_fixed"
        (folds / f"{name}.pkl").write_bytes(pickle.dumps(fixed))

        six = {
            fold: {
                "train": [s for i, s in enumerate(subjects) if i % 6 != fold - 1],
                "eval": [s for i, s in enumerate(subjects) if i % 6 == fold - 1],
            }
            for fold in range(1, 7)
        }
        (folds / f"{cohort}_6fold_participants.pkl").write_bytes(pickle.dumps(six))


def _write_catalog(root, sequences, extra=()):
    rows = [
        {
            "sample_id": sample_id_for(cohort, subject, walk),
            "source_dataset": "CARE-PD",
            "source_path": f"{cohort}_canonical__{subject}__{walk}.csv",
            "duration": 4.0,
        }
        for cohort, subject, walk, _ in sequences
    ]
    rows.extend(extra)
    upsert_table(root, CATALOG, pl.DataFrame(rows), keys=["sample_id"])


@pytest.fixture
def release(tmp_path):
    sequences = _sequences()
    _write_release(tmp_path / "carepd", sequences)
    _write_catalog(tmp_path / "out", sequences)
    return tmp_path / "out", tmp_path / "carepd", sequences


def test_annotations_span_the_take_and_keep_the_clinical_score(release):
    out, carepd, sequences = release
    annotations = import_carepd_annotations(output_root=out, carepd_root=carepd, cohorts=COHORTS)

    assert len(annotations) == len(sequences) == 36
    assert set(annotations["label_type"]) == {"sequence"}
    assert set(annotations["ontology"]) == {"updrs_gait"}
    # span comes from the catalog's resampled duration, not the release's fps
    assert set(annotations["start_t"]) == {0.0} and set(annotations["end_t"]) == {4.0}

    expected = {sample_id_for(c, s, w): str(score) for c, s, w, score in sequences}
    got = dict(zip(annotations["sample_id"], annotations["label"]))
    assert got == expected


def test_takes_with_no_imported_csv_are_dropped_not_guessed(tmp_path, capsys):
    sequences = _sequences()
    _write_release(tmp_path / "carepd", sequences)
    # the torque export drops takes that failed inverse dynamics
    _write_catalog(tmp_path / "out", sequences[:-5])

    annotations = import_carepd_annotations(
        output_root=tmp_path / "out", carepd_root=tmp_path / "carepd", cohorts=COHORTS
    )
    assert len(annotations) == len(sequences) - 5
    assert "have no imported CSV" in capsys.readouterr().out


def test_official_folds_are_participant_disjoint(release):
    out, carepd, _ = release
    splits = import_carepd_folds(output_root=out, carepd_root=carepd, cohorts=COHORTS)

    catalog = pl.read_parquet(out / "tables" / CATALOG)
    resolved = splits.join(catalog.select("sample_id", "source_path"), on="sample_id").with_columns(
        pl.col("source_path").str.split("__").list.get(1).alias("subject"),
        pl.col("source_path").str.split("__").list.get(0).alias("cohort"),
    )

    assert set(splits["split"]) == {"train", "eval"}
    assert set(splits["split_set"]) == {"carepd_fixed"} | {f"carepd_6fold_{k}" for k in range(1, 7)}

    for split_set in resolved["split_set"].unique():
        one = resolved.filter(pl.col("split_set") == split_set)
        # every take of a participant lands on one side, in every fold
        spans = one.group_by("cohort", "subject").agg(pl.col("split").n_unique().alias("n"))
        assert spans["n"].max() == 1, f"{split_set} splits a participant across sides"
        assert len(one) == 36, f"{split_set} does not cover every take"


def test_six_folds_partition_the_participants(release):
    out, carepd, _ = release
    splits = import_carepd_folds(output_root=out, carepd_root=carepd, cohorts=COHORTS)

    catalog = pl.read_parquet(out / "tables" / CATALOG)
    resolved = splits.join(catalog.select("sample_id", "source_path"), on="sample_id")

    # each take is held out exactly once across the six folds
    held_out = (
        resolved
        .filter(pl.col("split_set").str.starts_with("carepd_6fold_") & (pl.col("split") == "eval"))
        .group_by("sample_id")
        .len()
    )
    assert len(held_out) == 36
    assert set(held_out["len"]) == {1}


def test_folds_pool_cohorts_under_one_split_set(release):
    out, carepd, _ = release
    splits = import_carepd_folds(output_root=out, carepd_root=carepd, cohorts=COHORTS)

    catalog = pl.read_parquet(out / "tables" / CATALOG)
    fixed = (
        splits.filter(pl.col("split_set") == "carepd_fixed")
        .join(catalog.select("sample_id", "source_path"), on="sample_id")
        .with_columns(pl.col("source_path").str.split("__").list.get(0).alias("cohort"))
    )
    # both cohorts contribute to the same split set rather than getting one each
    assert set(fixed["cohort"]) == {"3DGait_canonical", "BMCLab_canonical"}
    assert set(fixed.filter(pl.col("split") == "eval")["cohort"]) == {
        "3DGait_canonical",
        "BMCLab_canonical",
    }


def test_pretrain_split_can_exclude_an_eval_corpus(tmp_path):
    sequences = [("BMCLab", "SUB01", "walk_0", 1)]
    amass = {
        "sample_id": "AMASS:CMU/01/01_01_stageii.csv",
        "source_dataset": "AMASS",
        "source_path": "CMU/01/01_01_stageii.csv",
        "duration": 9.0,
    }
    _write_catalog(tmp_path, sequences, extra=[amass])

    unrestricted = create_pretrain_split(output_root=tmp_path, split_set="pretrain_all")
    assert len(unrestricted.filter(pl.col("split_set") == "pretrain_all")) == 2

    restricted = create_pretrain_split(
        output_root=tmp_path, split_set="pretrain_amass", source_datasets=("AMASS",)
    )
    ids = restricted.filter(pl.col("split_set") == "pretrain_amass")["sample_id"].to_list()
    assert ids == [amass["sample_id"]]
