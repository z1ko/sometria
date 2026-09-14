"""Run with: python tests/test_paired_split.py"""

import sys
import tempfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sometria.catalog import SPLITS, CATALOG, create_paired_split, load_splits, upsert_table

ARMS = ("AMASS", "AMASS-SMPL")


def _catalog_row(arm, subset, take, broken=False, n_frames=1000):
    suffix = "csv" if arm == "AMASS" else "npz"
    return {
        "sample_id": f"{arm}:{subset}/s1/{take}.{suffix}",
        "source_dataset": arm,
        "source_path": f"{subset}/s1/{take}.{suffix}",
        "broken": broken,
        "n_frames": n_frames,
    }


def _root(rows, split_rows):
    """A throwaway processed root holding just the two tables the function reads."""

    root = Path(tempfile.mkdtemp())
    upsert_table(root, CATALOG, pl.DataFrame(rows), keys=["sample_id"])
    upsert_table(root, SPLITS, pl.DataFrame(split_rows), keys=["sample_id", "split_set"])
    return root


def _paired(rows, split_rows, **kwargs):
    root = _root(rows, split_rows)
    create_paired_split(
        output_root=root,
        split_set="paired",
        source_split_set="pretrain_v1",
        arms=ARMS,
        splits=("train",),
        **kwargs,
    )
    return load_splits(root).filter(pl.col("split_set") == "paired")


def _source(*takes, subset="CMU", split="train"):
    return [
        {"sample_id": f"AMASS:{subset}/s1/{take}.csv", "split_set": "pretrain_v1", "split": split}
        for take in takes
    ]


def test_a_take_only_one_arm_holds_is_in_neither():
    rows = [
        _catalog_row("AMASS", "CMU", "both"),
        _catalog_row("AMASS-SMPL", "CMU", "both"),
        _catalog_row("AMASS", "CMU", "opensim_only"),          # conversion kept it, npz missing
        _catalog_row("AMASS-SMPL", "CMU", "smpl_only"),        # conversion dropped it
    ]
    paired = _paired(rows, _source("both", "opensim_only", "smpl_only"))

    assert sorted(paired["sample_id"].to_list()) == [
        "AMASS-SMPL:CMU/s1/both.npz",
        "AMASS:CMU/s1/both.csv",
    ]
    assert paired["split"].unique().to_list() == ["train"]


def test_arms_are_matched_across_the_subset_renaming():
    # The OpenSim tree uses BABEL's folder names, the AMASS release its own.
    rows = [
        _catalog_row("AMASS", "EyesJapanDataset", "hello"),
        _catalog_row("AMASS-SMPL", "Eyes_Japan_Dataset", "hello"),
    ]
    assert len(_paired(rows, _source("hello", subset="EyesJapanDataset"))) == 2


def test_broken_or_short_in_one_arm_drops_the_take_from_both():
    """The torque filter can only ever fire on the OpenSim arm -- SMPL carries no torque."""

    rows = [
        _catalog_row("AMASS", "CMU", "torque", broken=True),
        _catalog_row("AMASS-SMPL", "CMU", "torque"),
        _catalog_row("AMASS", "CMU", "short", n_frames=120),
        _catalog_row("AMASS-SMPL", "CMU", "short", n_frames=1000),
        _catalog_row("AMASS", "CMU", "fine"),
        _catalog_row("AMASS-SMPL", "CMU", "fine"),
    ]
    paired = _paired(rows, _source("torque", "short", "fine"), min_frames=240)

    assert {i.split("/")[-1].split(".")[0] for i in paired["sample_id"]} == {"fine"}


def test_a_take_outside_the_source_split_stays_outside():
    rows = [
        _catalog_row("AMASS", "CMU", "held_out"),
        _catalog_row("AMASS-SMPL", "CMU", "held_out"),
        _catalog_row("AMASS", "CMU", "trained_on"),
        _catalog_row("AMASS-SMPL", "CMU", "trained_on"),
    ]
    paired = _paired(rows, _source("trained_on"))       # held_out has no pretrain_v1 row

    assert len(paired) == 2 and all("trained_on" in i for i in paired["sample_id"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
