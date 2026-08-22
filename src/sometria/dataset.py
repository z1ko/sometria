
import lightning as L
from omegaconf import DictConfig
import torch as t
import polars as pl

from pathlib import Path

from sometria.catalog import MotionViewSpec, build_motion_view


class RandomWindowCollate:
    """Collate variable-length motions into fixed-size training windows."""

    def __init__(self, window_frames: int) -> None:
        if window_frames <= 0:
            raise ValueError("window_frames must be positive.")
        self.window_frames = window_frames

    def __call__(self, batch: list[dict]) -> dict:
        windows = []
        valid = []
        sample_ids = []
        source_paths = []

        for item in batch:
            features = item["features"]
            n_frames = features.shape[0]
            sample_ids.append(item["sample_id"])
            source_paths.append(item["source_path"])

            if n_frames >= self.window_frames:
                max_start = n_frames - self.window_frames
                start = int(t.randint(max_start + 1, ()).item())
                window = features[start:start + self.window_frames]
                valid_window = t.ones(self.window_frames, dtype=t.bool)
            else:
                window = features.new_zeros((self.window_frames, *features.shape[1:]))
                window[:n_frames] = features
                valid_window = t.zeros(self.window_frames, dtype=t.bool)
                valid_window[:n_frames] = True

            windows.append(window)
            valid.append(valid_window)

        return {
            "features": t.stack(windows),
            "valid": t.stack(valid),
            "sample_id": sample_ids,
            "source_path": source_paths,
        }


class MotionDataset(t.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str | Path,
        samples: pl.DataFrame,
        normalization_path: str | Path,
    ) -> None:
        super().__init__()
        self.root_folder = Path(root_folder)
        self.samples = samples
        self.normalization_path = self._resolve_normalization_path(normalization_path)

        if not self.normalization_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {self.normalization_path}. "
                "Create them during preprocessing and pass the path here."
            )

        stats = t.load(self.normalization_path, weights_only=True)
        self.mean = stats["mean"].float()
        self.std = stats["std"].float()
        self.normalization_mask = stats.get(
            "normalization_mask",
            t.ones_like(self.mean, dtype=t.bool),
        ).bool()
        if self.normalization_mask.shape != self.mean.shape:
            raise ValueError(
                f"normalization_mask has shape {tuple(self.normalization_mask.shape)}, "
                f"but mean has shape {tuple(self.mean.shape)}."
            )

    def _resolve_normalization_path(self, normalization_path: str | Path) -> Path:
        path = Path(normalization_path)
        if path.is_absolute():
            return path
        return self.root_folder / path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.row(index, named=True)
        sample = t.load(
            self.root_folder / row["motion_path"],
            weights_only=True
        )

        features = sample["features"].float()
        if features.shape[1:] != self.mean.shape[1:]:
            raise ValueError(
                f"{row['motion_path']} has feature shape {tuple(features.shape[1:])}, "
                f"but normalization stats expect {tuple(self.mean.shape[1:])}."
            )

        normalized_features = t.where(
            self.normalization_mask,
            (features - self.mean) / self.std,
            features,
        )

        return {
            "features": normalized_features,
            "time": sample["time"],
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
            "source_subset": row["source_subset"],
            "source_path": row["source_path"],
            "representation": row["representation"],
            "metadata": row["metadata"],
            "hz": row["hz"],
            "duration": row["duration"],
        }

class MotionDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.root_folder = Path(config.dataloader.root)
        self.batch_size = config.dataloader.batch_size
        self.num_workers = config.dataloader.get("num_workers", 4)
        self.window_frames = config.dataloader.get("window_frames", 256)

        normalization = config.dataloader.get("normalization")
        if normalization is None:
            raise ValueError("config.dataloader.normalization is required.")
        self.normalization_path = Path(normalization)

        self.train_spec = MotionViewSpec(
            split_set=config.dataloader.train.split_set,
            split=config.dataloader.train.split,
            source_datasets=tuple(config.dataloader.train.get("source_datasets", [])),
        )

        self.val_spec = MotionViewSpec(
            split_set=config.dataloader.val.split_set,
            split=config.dataloader.val.split,
            source_datasets=tuple(config.dataloader.val.get("source_datasets", [])),
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            train_samples = build_motion_view(self.root_folder, self.train_spec)
            val_samples   = build_motion_view(self.root_folder, self.val_spec)

            self.train_dataset = MotionDataset(
                self.root_folder,
                train_samples,
                normalization_path=self.normalization_path,
            )
            self.val_dataset = MotionDataset(
                self.root_folder,
                val_samples,
                normalization_path=self.normalization_path,
            )
        

    def _loader(self, dataset: MotionDataset, shuffle: bool, drop_last: bool):
        return t.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=RandomWindowCollate(self.window_frames),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, drop_last=False, shuffle=False)
