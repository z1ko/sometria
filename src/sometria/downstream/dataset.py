"""Labelled windows: the same crops as pretraining, with a multi-hot target attached.

Two things differ from the pretraining loader. Windows carry labels, and validation
enumerates *tiles* -- every non-overlapping window of every sample, in order -- rather
than drawing one random crop per sample. Tiles mean every annotated frame is scored
exactly once and the number is repeatable between epochs and between runs.

Items come out already fixed-size, so there is no collate here: short samples are
excluded by ``MotionViewSpec.min_frames`` on both splits, exactly as in pretraining, and
a window is never padded.
"""

from pathlib import Path

import lightning as L
import numpy as np
from omegaconf import DictConfig
import polars as pl
import torch as t

from sometria.catalog import MotionViewSpec, build_motion_view
from sometria.dataset import MotionDataset
from sometria.downstream.labels import (
    LABEL_MIN_COVERAGE,
    annotated_sample_ids,
    load_label_segments,
    load_label_vocabulary_index,
    window_multi_hot,
    window_patch_labels,
)
from sometria.representation import Representation


def _tiles(n_frames: int, window_frames: int) -> list[int]:
    """Window starts covering ``n_frames`` end to end, or nothing if it is too short.

    The last tile is flush with the end of the motion rather than dropped, so no frame
    goes unscored -- ``n % window_frames`` is 13.4% of ``babel_official/val``'s frames,
    and always the end of a take. It overlaps its predecessor instead, which double-counts
    at worst; dropping would systematically never score how a motion finishes.
    """

    if n_frames < window_frames:
        return []
    starts = list(range(0, n_frames - window_frames + 1, window_frames))
    if starts[-1] + window_frames < n_frames:
        starts.append(n_frames - window_frames)
    return starts


class LabelledWindows(t.utils.data.Dataset):
    """Fixed-length windows over a labelled motion view.

    ``tiles=False`` yields one randomly placed window per sample per epoch (training).
    ``tiles=True`` yields every non-overlapping window of every sample (evaluation).
    """

    def __init__(
        self,
        motions: MotionDataset,
        segments: dict[str, np.ndarray],
        *,
        num_labels: int,
        window_frames: int,
        min_coverage: float = LABEL_MIN_COVERAGE,
        tiles: bool = False,
        label_patches: int | None = None,
    ) -> None:
        super().__init__()
        self.motions = motions
        self.segments = segments
        self.num_labels = num_labels
        self.window_frames = window_frames
        self.min_coverage = min_coverage
        self.label_patches = label_patches

        frames = motions.samples["n_frames"].to_list()
        if tiles:
            self.index = [(i, start) for i, n in enumerate(frames) for start in _tiles(n, window_frames)]
        else:
            self.index = [(i, None) for i, n in enumerate(frames) if n >= window_frames]

        # ponytail: one-entry cache. Tiles of a sample are contiguous in the index and
        # evaluation does not shuffle, so this turns one load per tile into one per
        # sample. A real LRU only pays off if the order stops being sequential.
        self._cached: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        sample_index, start = self.index[index]
        if self._cached is None or self._cached[0] != sample_index:
            self._cached = (sample_index, self.motions[sample_index])
        item = self._cached[1]

        features = item["features"]
        if start is None:
            start = int(t.randint(features.shape[0] - self.window_frames + 1, ()).item())

        start_t = float(item["time"][start])
        end_t = start_t + self.window_frames / float(item["hz"])
        # from_numpy is a view, not a copy -- see load_label_segments for why the
        # dict cannot hold tensors in the first place.
        rows = self.segments.get(item["sample_id"])
        segments = t.zeros(0, 3) if rows is None else t.from_numpy(rows)

        # ``label_patches`` is the whole difference between segmentation and
        # classification: same windows, same coverage rule, a target resolved in time
        # rather than collapsed over it.
        if self.label_patches is None:
            labels = window_multi_hot(
                segments, start_t, end_t, self.num_labels, self.min_coverage
            )
        else:
            labels = window_patch_labels(
                segments, start_t, end_t, self.label_patches, self.num_labels, self.min_coverage
            )

        return {
            "features": features[start : start + self.window_frames],
            "labels": labels,
            "sample_id": item["sample_id"],
        }


class LabelledMotionDataModule(L.LightningDataModule):
    """The labelled counterpart of :class:`~sometria.dataset.MotionDataModule`."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        loader = config.dataloader
        self.root_folder = Path(loader.root)
        self.batch_size = loader.batch_size
        self.num_workers = loader.get("num_workers", 4)
        self.window_frames = loader.get("window_frames", 240)
        self.label_set = loader.label_set
        self.min_coverage = loader.get("label_min_coverage", LABEL_MIN_COVERAGE)
        # Which ontology counts as "annotated" follows the label set, and is not
        # act_cat for every benchmark: CARE-PD's scores are `updrs_gait`. Hardcoding
        # the BABEL default here filtered every CARE-PD sample out of the view and
        # produced an empty dataset rather than an error.
        self.ontology = load_label_vocabulary_index(self.root_folder, self.label_set)[0]["ontology"][0]
        # Set for segmentation, absent for classification. It belongs to the loader
        # rather than the model because it changes the shape of the target.
        self.label_patches = loader.get("label_patches")
        # Which BABEL annotation levels count. None means both, which is right for
        # window classification; segmentation wants ("frame",) because a sequence label
        # spans 0..dur and so is constant across every window it touches. Narrowing it
        # also drops the samples that have nothing left, rather than leaving them in as
        # all-negative windows.
        label_types = loader.get("label_types")
        self.label_types = None if label_types is None else tuple(label_types)
        # Whether a sample with no in-vocabulary label is a negative or is out of scope.
        # False is right for BABEL: `transition` is 19% of rows and has no index, so a
        # window covering only it trains as all-negative, and dropping it would bias the
        # benchmark toward segments that happen to carry a scoreable label. True is right
        # for a single-label benchmark that trims its vocabulary -- CARE-PD's score-3
        # takes are 1% of the corpus and, left in, are not "none of the above", they are
        # a class the head was told to stop predicting.
        self.require_in_vocabulary = loader.get("require_in_vocabulary", False)

        normalization = loader.get("normalization")
        if normalization is None:
            raise ValueError("config.dataloader.normalization is required.")
        self.normalization_path = Path(normalization)

        human = loader.get("human")
        if human is None:
            raise ValueError("config.dataloader.human is required.")
        self.representation = Representation.from_config(human)

        self.train_spec = self._spec(loader.train)
        self.val_spec = self._spec(loader.val)

    def _spec(self, split: DictConfig) -> MotionViewSpec:
        # exclude_broken defaults on, as everywhere else, but has to be reachable per
        # benchmark: TAU_RATE_MAX was calibrated as five robust sigmas above AMASS's
        # median torque rate, and on a 25 Hz clinical capture it cuts near that cohort's
        # own median. Dropping is then a coin flip rather than a quality signal, and
        # because cohorts are not balanced across severity it moves the class
        # distribution -- a filtered CARE-PD run is not scoring CARE-PD's benchmark.
        return MotionViewSpec(
            split_set=split.split_set,
            split=split.split,
            source_datasets=tuple(split.get("source_datasets", [])),
            label_sources=tuple(split.get("label_sources", ["BABEL"])),
            require_labels=True,
            exclude_broken=split.get("exclude_broken", True),
            min_frames=self.window_frames,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return

        self.train_dataset = self._windows(self.train_spec, tiles=False)
        self.val_dataset = self._windows(self.val_spec, tiles=True)

    def _windows(self, spec: MotionViewSpec, *, tiles: bool) -> LabelledWindows:
        samples = build_motion_view(self.root_folder, spec).filter(
            pl.col("sample_id").is_in(
                annotated_sample_ids(
                    self.root_folder, self.ontology, label_types=self.label_types
                )
            )
        )
        segments, num_labels = load_label_segments(
            self.root_folder,
            label_set=self.label_set,
            sample_ids=samples["sample_id"].to_list(),
            **({} if self.label_types is None else {"label_types": self.label_types}),
        )
        self.num_labels = num_labels

        if self.require_in_vocabulary:
            samples = samples.filter(pl.col("sample_id").is_in(list(segments)))
            if samples.is_empty():
                raise ValueError(
                    f"no sample in {spec.split_set}/{spec.split} carries a label in "
                    f"{self.label_set!r}; check the vocabulary and label_sources"
                )

        return LabelledWindows(
            MotionDataset(
                self.root_folder,
                samples,
                normalization_path=self.normalization_path,
                representation=self.representation,
            ),
            segments,
            num_labels=num_labels,
            window_frames=self.window_frames,
            min_coverage=self.min_coverage,
            tiles=tiles,
            label_patches=self.label_patches,
        )

    def _loader(self, dataset: LabelledWindows, shuffle: bool):
        return t.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)


class LabelledWindowsEx(t.utils.data.Dataset):
    def __init__(
        self,
        motions: MotionDataset,
        segments: dict[str, np.ndarray],
        *,
        num_labels: int,
        window_frames: int,
        min_coverage: float = LABEL_MIN_COVERAGE,
        tiles: bool = False,
    ) -> None:
        super().__init__()

        self.motions = motions
        self.segments = segments
        self.num_labels = num_labels
        self.window_frames = window_frames
        self.min_coverage = min_coverage
        self.tiles = tiles

        frames = motions.samples["n_frames"].to_list()
        if tiles:
            self.index = [
                (i, start) 
                for i, n in enumerate(frames) 
                for start in _tiles(n, window_frames)
            ]
        else:
            self.index = [(i, None) for i, n in enumerate(frames) if n >= window_frames]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        sample_index, start = self.index[index]
        
        # Load sample directly without non-thread-safe cached state
        item = self.motions[sample_index]
        features = item["features"]

        if start is None:
            # Pick a random frame offset for training
            max_start = features.shape[0] - self.window_frames
            start = int(t.randint(0, max_start + 1, ()).item())

        start_t = float(item["time"][start])
        end_t = start_t + self.window_frames / float(item["hz"])

        rows = self.segments.get(item["sample_id"])
        segments = t.zeros(0, 3) if rows is None else t.from_numpy(rows)

        labels = window_multi_hot(
            segments, start_t, end_t, self.num_labels, self.min_coverage
        )

        return {
            "features": features[start : start + self.window_frames],
            "labels": labels,
            "sample_id": item["sample_id"],
        }