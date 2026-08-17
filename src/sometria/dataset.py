
from typing import Any

import torch as t
import polars as pl
import glob

from pathlib import Path

from sometria.human import _load_sample, _resample_sample

# NOTE: Hardcoded paths, no need for more complexity
PATH_HUMAN_DEFINITION : Path = Path("config/human.yaml")
PATH_OUTPUT_ROOT: Path = Path("data/processed")

# Uniform rate for every sample. See _resample_sample for why 60 Hz.
TARGET_HZ: float = 60.0

def preprocess(
    *,
    input_root: str | Path,
    pattern: str,
    human: dict,
    save_path: str | Path,
    target_hz: float = TARGET_HZ,
) -> pl.DataFrame:

    search_path = f"{str(input_root)}/{pattern}"
    save_path = Path(save_path)

    files = sorted(Path(p) for p in glob.glob(search_path, recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {pattern}")

    # Create samples directory
    samples_dir = save_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # How many files to store
    written = 0

    sample_rows: list[dict] = []
    for i, path in enumerate(files, start=1):

        sample = _load_sample(path, human)
        sample = _resample_sample(sample, target_hz)

        # Normalize path
        sample["path"] = str(Path(sample["path"]).relative_to(input_root))

        sample_motion_path = samples_dir / f"sample_{i:04}.pt"
        t.save(
            {
                "motion": t.tensor(sample["motion"]),
                "time":   t.tensor(sample["time"])
            }, 
            sample_motion_path
        )

        # Store metadata of the sample
        sample_dict = { "sample": i, "sample_path": str(sample_motion_path.relative_to(save_path)) }
        sample_dict.update({
            k: sample[k] for k in ("path", "metadata", "hz", "original_hz", "n_frames", "duration")
        })
        sample_rows.append(sample_dict)

        written += 1
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} samples")

    samples_df = pl.DataFrame(sample_rows)
    samples_df.write_parquet(save_path / "samples.parquet")
    return samples_df


class MotionDataset(t.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str | Path,
        split: str | None = None,
        samples_df: pl.DataFrame | None = None,
    ) -> None:
        """`split` is one of BABEL's train/val/test, or "pretrain" for everything BABEL
        does not annotate plus its train split. Requires splits.enrich_samples() to have run."""
        super().__init__()

        self.root_folder = Path(root_folder)

        if samples_df is None:
            self.samples = pl.read_parquet(self.root_folder / "samples.parquet")
        else:
            self.samples = samples_df

        if split == "pretrain":
            self.samples = self.samples.filter(
                pl.col("split").is_null() | (pl.col("split") == "train")
            )
        elif split is not None:
            self.samples = self.samples.filter(pl.col("split") == split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.row(index, named=True)
        sample = t.load(
            self.root_folder / row["sample_path"],
            weights_only=True
        )

        return {
            "motion": sample["motion"],
            "time": sample["time"],
            "sample": row["sample"],
            "path": row["path"],
            "metadata": row["metadata"],
            "hz": row["hz"],
        }