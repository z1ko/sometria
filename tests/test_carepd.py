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


# Seven, not six: the leave-one-subject-out file is named for the participant count, so at
# six it would collide with <cohort>_6fold_participants and the importer could not tell the
# two protocols apart.
N_SUBJECTS = 7


def _subjects(cohort):
    # 3DGait numbers participants, BMCLab uses SUBnn -- both appear as bare strings in the
    # released fold files, so the importer must not assume either shape
    return (
        [str(i) for i in range(N_SUBJECTS)]
        if cohort == "3DGait"
        else [f"SUB{i:02d}" for i in range(N_SUBJECTS)]
    )


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

        loso = {
            fold: {"train": subjects[:fold - 1] + subjects[fold:], "eval": [subjects[fold - 1]]}
            for fold in range(1, len(subjects) + 1)
        }
        (folds / f"{cohort}_{len(subjects)}fold_participants.pkl").write_bytes(pickle.dumps(loso))


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

    assert len(annotations) == len(sequences) == 2 * N_SUBJECTS * 3
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

    for split_set in resolved["split_set"].unique():
        one = resolved.filter(pl.col("split_set") == split_set)
        # every take of a participant lands on one side, in every protocol and every fold
        spans = one.group_by("cohort", "subject").agg(pl.col("split").n_unique().alias("n"))
        assert spans["n"].max() == 1, f"{split_set} splits a participant across sides"


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
    assert len(held_out) == 2 * N_SUBJECTS * 3
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


def test_probe_datamodule_follows_the_label_set_off_babel(release, tmp_path):
    """The two settings that decide whether a CARE-PD probe sees any data at all.

    Both failed silently before: the wrong ontology filtered every sample out and produced
    an empty dataset rather than an error, and ``exclude_broken`` was unreachable from the
    config, so a benchmark whose broken filter is not label-neutral could not turn it off.
    """

    from omegaconf import OmegaConf

    from sometria.carepd import import_carepd_gait_vocabulary
    from sometria.downstream.dataset import LabelledMotionDataModule
    from sometria.downstream.labels import annotated_sample_ids

    out, carepd, sequences = release
    import_carepd_annotations(output_root=out, carepd_root=carepd, cohorts=COHORTS)
    import_carepd_gait_vocabulary(output_root=out)

    # the bug this guards: BABEL's ontology is the wrong question for this label set
    assert annotated_sample_ids(out, "act_cat") == set()
    assert len(annotated_sample_ids(out, "updrs_gait")) == len(sequences)

    config = OmegaConf.create(
        {
            "dataloader": {
                "root": str(out),
                "human": "config/human.yaml",
                "normalization": "unused-until-setup.pt",
                "batch_size": 4,
                "window_frames": 120,
                "label_set": "carepd_updrs_gait",
                "train": {
                    "split_set": "carepd_fixed",
                    "split": "train",
                    "source_datasets": ["CARE-PD"],
                    "label_sources": ["CARE-PD"],
                    "exclude_broken": False,
                },
                "val": {
                    "split_set": "carepd_fixed",
                    "split": "eval",
                    "source_datasets": ["CARE-PD"],
                    "label_sources": ["CARE-PD"],
                },
            }
        }
    )

    datamodule = LabelledMotionDataModule(config)

    assert datamodule.ontology == "updrs_gait"
    assert datamodule.require_in_vocabulary is False  # BABEL's all-negative rule by default
    assert datamodule.train_spec.exclude_broken is False
    # absent from the val block, so it keeps the corpus-wide default rather than
    # inheriting whatever the train block asked for
    assert datamodule.val_spec.exclude_broken is True
    assert datamodule.val_spec.split == "eval"


def _resolved(out, carepd):
    """Every split set, with cohort and subject attached."""

    splits = import_carepd_folds(output_root=out, carepd_root=carepd, cohorts=COHORTS)
    catalog = pl.read_parquet(out / "tables" / CATALOG)
    return splits.join(catalog.select("sample_id", "source_path"), on="sample_id").with_columns(
        pl.col("source_path").str.split("__").list.get(0).str.replace("_canonical", "").alias("cohort"),
        pl.col("source_path").str.split("__").list.get(1).alias("subject"),
    )


def test_leave_one_subject_out_holds_out_one_subject_per_fold(release):
    out, carepd, _ = release
    resolved = _resolved(out, carepd)

    for cohort in COHORTS:
        folds = sorted(
            name for name in resolved["split_set"].unique()
            if name.startswith(f"carepd_{cohort}_loso_")
        )
        assert len(folds) == N_SUBJECTS, cohort

        held_out = []
        for name in folds:
            one = resolved.filter(pl.col("split_set") == name)
            # a LOSO fold is one cohort only -- the other cohorts join in MIDA, not here
            assert set(one["cohort"]) == {cohort}, name
            evaluated = set(one.filter(pl.col("split") == "eval")["subject"])
            assert len(evaluated) == 1, f"{name} holds out {len(evaluated)} subjects"
            held_out.append(evaluated.pop())

        assert sorted(held_out) == sorted(_subjects(cohort)), cohort


def test_mida_is_loso_plus_the_other_cohorts_on_the_training_side(release):
    out, carepd, _ = release
    resolved = _resolved(out, carepd)

    for cohort in COHORTS:
        for fold in range(1, N_SUBJECTS + 1):
            loso = resolved.filter(pl.col("split_set") == f"carepd_{cohort}_loso_{fold}")
            mida = resolved.filter(pl.col("split_set") == f"carepd_mida_{cohort}_loso_{fold}")

            def side(frame, split):
                return set(frame.filter(pl.col("split") == split)["sample_id"])

            # identical held-out subject: that is what makes the pair comparable at all
            assert side(mida, "eval") == side(loso, "eval"), (cohort, fold)
            # and strictly more training data, all of it from the other cohorts
            assert side(loso, "train") < side(mida, "train"), (cohort, fold)
            added = mida.filter(
                pl.col("split") == "train"
            ).join(loso, on="sample_id", how="anti")
            assert set(added["cohort"]) == set(COHORTS) - {cohort}, (cohort, fold)


def test_lodo_and_cross_dataset_never_train_on_the_target_cohort(release):
    out, carepd, _ = release
    resolved = _resolved(out, carepd)
    takes = {cohort: len(resolved.filter(pl.col("cohort") == cohort)["sample_id"].unique())
             for cohort in COHORTS}

    for target in COHORTS:
        lodo = resolved.filter(pl.col("split_set") == f"carepd_lodo_{target}")
        assert set(lodo.filter(pl.col("split") == "train")["cohort"]) == set(COHORTS) - {target}
        # the whole cohort is evaluated: no target data was trained on, so nothing to hold back
        assert set(lodo.filter(pl.col("split") == "eval")["cohort"]) == {target}
        assert len(lodo.filter(pl.col("split") == "eval")) == takes[target]

        for source in COHORTS:
            if source == target:
                continue
            cross = resolved.filter(pl.col("split_set") == f"carepd_cross_{source}_to_{target}")
            assert set(cross.filter(pl.col("split") == "train")["cohort"]) == {source}
            assert set(cross.filter(pl.col("split") == "eval")["cohort"]) == {target}
