"""Map BABEL train/val/test splits onto preprocessed AMASS samples.

BABEL identifies a sequence by `feat_p`, e.g. `MPIHDM05/MPI_HDM05/dg/HDM_dg_03-11_03_120_poses.npz`.
Our sample paths look like `HDM05/dg/HDM_dg_03-11_03_120_stageii.csv`. So: drop BABEL's duplicated
second component, rename the dataset folder, drop the `_poses` / `_stageii` suffix, and normalize case
and separators (AMASS mirrors differ on spaces/underscores/dashes).
"""

import json
import re
from pathlib import Path

import polars as pl

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
    parts = Path(feat_p).with_suffix("").parts  # <dataset>/<dataset>/<subject>/<seq>
    return _key(parts[0], "/".join(parts[2:]))


def sample_key(path: str) -> str:
    parts = Path(path).with_suffix("").parts  # <dataset>/<subject>/<seq>
    return _key(parts[0], "/".join(parts[1:]))


def load_babel_splits(splits_dir: str | Path) -> pl.DataFrame:
    """Read BABEL split jsons into a (key, split, babel_sid) frame."""
    splits_dir = Path(splits_dir)
    rows = []
    for split in ("train", "val", "test"):
        for seq in json.loads((splits_dir / f"{split}.json").read_text()).values():
            rows.append({"key": babel_key(seq["feat_p"]), "split": split, "babel_sid": seq["babel_sid"]})
    return pl.DataFrame(rows)


def assign_splits(samples: pl.DataFrame, splits_dir: str | Path) -> pl.DataFrame:
    """Add `split` and `babel_sid` columns to a samples frame; null where BABEL has no annotation."""
    keyed = samples.with_columns(
        pl.col("path").map_elements(sample_key, return_dtype=pl.String).alias("key")
    )
    dups = keyed.filter(pl.col("key").is_duplicated())["path"].to_list()
    if dups:
        print(f"warning: {len(dups)} sample paths collide after normalization, e.g. {dups[:4]}")

    return keyed.join(load_babel_splits(splits_dir), on="key", how="left").drop("key")


def enrich_samples(root: str | Path = "data/processed") -> pl.DataFrame:
    """Write `split` and `babel_sid` back into samples.parquet. Re-run after preprocess()."""
    root = Path(root)
    df = assign_splits(pl.read_parquet(root / "samples.parquet"), root / "splits/babel")
    df.write_parquet(root / "samples.parquet")
    return df


if __name__ == "__main__":
    assert babel_key("MPIHDM05/MPI_HDM05/dg/HDM_dg_03-11_03_120_poses.npz") == sample_key(
        "HDM05/dg/HDM_dg_03-11_03_120_stageii.csv"
    )
    assert babel_key("ACCAD/ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz") == sample_key(
        "ACCAD/Female1Gestures_c3d/D3_-_Conversation_Gestures_stageii.csv"
    )
    assert babel_key("MPImosh/MPI_mosh/50022/stretch_poses_poses.npz") == "MoSh/50022/stretchposes"

    df = enrich_samples()
    print(df.group_by("split").len().sort("len", descending=True))
    print(f"{df['split'].is_not_null().sum()}/{len(df)} samples covered by BABEL")
