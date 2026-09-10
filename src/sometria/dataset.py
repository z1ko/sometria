
import lightning as L
from omegaconf import DictConfig
import torch as t
import polars as pl

from pathlib import Path

from sometria.catalog import MotionViewSpec, build_motion_view
from sometria.representation import Representation


class WindowCollate:
    """Crop each motion in a batch to one fixed-size window.

    ``random_offset`` is what separates training from evaluation: a random crop each
    epoch is augmentation while training, but while validating it moves the metric for
    reasons unrelated to the model, and ``ModelCheckpoint(monitor="val/loss")`` then
    selects on crop luck. Evaluation takes the centre of the motion, every epoch.

    A motion shorter than a window is an error here, not something to zero-pad. Padding
    scores as motionless, so motion-aware masking keeps it and spends the context budget
    on frames that are not there -- which is why ``MotionViewSpec.min_frames`` drops those
    samples on every split. This raise is that decision's backstop, so a view built
    without ``min_frames`` fails by name rather than by training on padding.
    """

    def __init__(self, window_frames: int, random_offset: bool = True) -> None:
        if window_frames <= 0:
            raise ValueError("window_frames must be positive.")
        self.window_frames = window_frames
        self.random_offset = random_offset

    def __call__(self, batch: list[dict]) -> dict:
        windows = []
        sample_ids = []
        source_paths = []

        for item in batch:
            features = item["features"]
            n_frames = features.shape[0]
            if n_frames < self.window_frames:
                raise ValueError(
                    f"{item['sample_id']} has {n_frames} frames, fewer than the "
                    f"{self.window_frames}-frame window. Build the view with "
                    f"MotionViewSpec(min_frames={self.window_frames})."
                )

            max_start = n_frames - self.window_frames
            start = (
                int(t.randint(max_start + 1, ()).item())
                if self.random_offset
                else max_start // 2
            )

            windows.append(features[start:start + self.window_frames])
            sample_ids.append(item["sample_id"])
            source_paths.append(item["source_path"])

        return {
            "features": t.stack(windows),
            "sample_id": sample_ids,
            "source_path": source_paths,
        }


class MotionDataset(t.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str | Path,
        samples: pl.DataFrame,
        normalization_path: str | Path,
        representation: Representation,
    ) -> None:
        super().__init__()
        self.root_folder = Path(root_folder)
        self.samples = samples
        self.representation = representation
        self.normalization_path = self._resolve_normalization_path(normalization_path)

        if not self.normalization_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {self.normalization_path}. "
                "Create them during preprocessing and pass the path here."
            )

        payload = t.load(self.normalization_path, weights_only=True)
        # Only mean/std are kept: which slots they apply to is the representation's,
        # not the stats file's, so the stored normalization_mask is ignored.
        self.stats = {"mean": payload["mean"].float(), "std": payload["std"].float()}

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
        expected = self.stats["mean"].shape[1:]
        if features.shape[1:] != expected:
            raise ValueError(
                f"{row['motion_path']} has feature shape {tuple(features.shape[1:])}, "
                f"but normalization stats expect {tuple(expected)}."
            )

        return {
            "features": self.representation.to_model(features, self.stats),
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
        self.duration_weighted = config.dataloader.get("duration_weighted", True)

        normalization = config.dataloader.get("normalization")
        if normalization is None:
            raise ValueError("config.dataloader.normalization is required.")
        self.normalization_path = Path(normalization)

        human = config.dataloader.get("human")
        if human is None:
            raise ValueError("config.dataloader.human is required.")
        self.representation = Representation.from_config(human)

        # Short samples are dropped, not zero-padded: padding scores as motionless, so
        # motion-aware masking keeps it and spends the context budget on frames that are
        # not there. The spec is where that decision is enforced, and it has to reach both
        # splits -- WindowCollate raises on anything shorter that reaches it.
        self.train_spec = MotionViewSpec(
            split_set=config.dataloader.train.split_set,
            split=config.dataloader.train.split,
            source_datasets=tuple(config.dataloader.train.get("source_datasets", [])),
            exclude_broken=config.dataloader.train.get("exclude_broken", True),
            min_frames=self.window_frames,
        )

        self.val_spec = MotionViewSpec(
            split_set=config.dataloader.val.split_set,
            split=config.dataloader.val.split,
            source_datasets=tuple(config.dataloader.val.get("source_datasets", [])),
            min_frames=self.window_frames,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            train_samples = build_motion_view(self.root_folder, self.train_spec)
            val_samples   = build_motion_view(self.root_folder, self.val_spec)

            self.train_dataset = MotionDataset(
                self.root_folder,
                train_samples,
                normalization_path=self.normalization_path,
                representation=self.representation,
            )
            self.val_dataset = MotionDataset(
                self.root_folder,
                val_samples,
                normalization_path=self.normalization_path,
                representation=self.representation,
            )
        

    def _loader(
        self,
        dataset: MotionDataset,
        shuffle: bool,
        drop_last: bool,
        sampler=None,
        random_offset: bool = True,
    ):
        return t.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=WindowCollate(self.window_frames, random_offset=random_offset),
        )

    def _duration_weighted_sampler(self, dataset: MotionDataset):
        """Draw each motion in proportion to its length, one random window per draw.

        Sampling motions uniformly gives a 20 s take and a 4 s take the same one window
        per epoch, so the short one's frames are seen five times as often. Weighting by
        frame count equalizes exposure per frame instead of per file. Draws are with
        replacement, so a long motion contributes several windows per epoch and lands on
        different offsets each time.
        """

        frames = dataset.samples["n_frames"].to_list()
        return t.utils.data.WeightedRandomSampler(
            weights=frames,
            # one epoch = enough windows to cover the corpus once, not one per file
            num_samples=max(1, sum(frames) // self.window_frames),
            replacement=True,
        )

    def train_dataloader(self):
        sampler = self._duration_weighted_sampler(self.train_dataset) if self.duration_weighted else None
        return self._loader(self.train_dataset, drop_last=True, shuffle=True, sampler=sampler)

    def val_dataloader(self):
        return self._loader(
            self.val_dataset, drop_last=False, shuffle=False, random_offset=False
        )
