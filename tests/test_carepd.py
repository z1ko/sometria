"""Checks for the two pieces of CARE-PD ingest that can silently produce wrong training data:
subset parsing from a flat export, and subject-disjoint splits."""

import numpy as np
import polars as pl
import pytest

from sometria.carepd import import_carepd_subject_splits, sample_id_for
from sometria.catalog import CATALOG, create_pretrain_split, upsert_table
from sometria.preprocess import _source_subset


def test_source_subset_handles_both_layouts():
    # nested (AMASS, MotionX): leading directory
    assert _source_subset("CMU/01/01_01_stageii.csv") == "CMU"
    assert _source_subset("idea400/subset_0000/take_1.csv") == "idea400"
    # flat (CARE-PD): filename prefix, not the whole filename
    assert _source_subset("BMCLab_canonical__SUB01__SUB01_off_walk_13.csv") == "BMCLab_canonical"
    assert _source_subset("3DGait_canonical__0__vid0073_0055.csv") == "3DGait_canonical"
    # flat with no subset marker degrades to the old behaviour rather than crashing
    assert _source_subset("take.csv") == "take.csv"


def _write_mocha(root, sequences):
    (root / "sequences").mkdir(parents=True)
    for dataset_id, subject_id, take, label in sequences:
        sample_id = f"{dataset_id}__{subject_id}__{take}"
        np.savez(
            root / "sequences" / f"{sample_id}.npz",
            sample_id=sample_id,
            dataset_id=dataset_id,
            subject_id=subject_id,
            label=label,
        )


def _write_catalog(root, sequences, extra=()):
    rows = [
        {
            "sample_id": sample_id_for(f"{dataset_id}__{subject_id}__{take}"),
            "source_dataset": "CARE-PD",
            "source_path": f"{dataset_id}__{subject_id}__{take}.csv",
            "duration": 4.0,
        }
        for dataset_id, subject_id, take, _ in sequences
    ]
    rows.extend(extra)
    upsert_table(root, CATALOG, pl.DataFrame(rows), keys=["sample_id"])


@pytest.fixture
def corpus(tmp_path):
    # 8 subjects across two subsets, several takes each -- takes are what leaks if
    # the split assigns rows instead of subjects
    sequences = [
        (dataset_id, f"SUB{subject:02d}", f"walk_{take}", (subject + take) % 4)
        for dataset_id in ("BMCLab_canonical", "3DGait_canonical")
        for subject in range(4)
        for take in range(3)
    ]
    mocha = tmp_path / "mocha"
    _write_mocha(mocha, sequences)
    _write_catalog(tmp_path / "out", sequences)
    return tmp_path / "out", mocha


def test_subject_splits_are_disjoint(corpus):
    out, mocha = corpus
    splits = import_carepd_subject_splits(output_root=out, mocha_root=mocha)

    catalog = pl.read_parquet(out / "tables" / CATALOG)
    labelled = splits.join(catalog.select("sample_id", "source_path"), on="sample_id").with_columns(
        pl.col("source_path").str.split("__").list.slice(0, 2).list.join("__").alias("subject")
    )

    assert len(labelled) == 24
    per_subject = labelled.group_by("subject").agg(pl.col("split").n_unique().alias("n"))
    assert per_subject["n"].max() == 1, "a subject spans more than one split"
    assert set(labelled["split"].unique()) == {"train", "val", "test"}

    # every take of a subject follows that subject, so splits land on take boundaries of 3
    assert all(count % 3 == 0 for count in labelled.group_by("split").len()["len"])


def test_subject_splits_are_deterministic(corpus):
    out, mocha = corpus
    first = import_carepd_subject_splits(output_root=out, mocha_root=mocha, seed=7)
    second = import_carepd_subject_splits(output_root=out, mocha_root=mocha, seed=7)
    assert first.sort("sample_id").equals(second.sort("sample_id"))


def test_pretrain_split_can_exclude_an_eval_corpus(tmp_path):
    sequences = [("BMCLab_canonical", "SUB01", "walk_0", 1)]
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
