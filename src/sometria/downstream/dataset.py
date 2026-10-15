"""Labelled windows: the same crops as pretraining, with a multi-hot target attached.

Two things differ from the pretraining loader. Windows carry labels, and validation
enumerates *tiles* -- every non-overlapping window of every sample, in order -- rather
than drawing one random crop per sample. Tiles mean every annotated frame is scored
exactly once and the number is repeatable between epochs and between runs.

Items come out already fixed-size, so there is no collate here: short samples are
excluded by ``MotionViewSpec.min_frames`` on both splits, exactly as in pretraining, and
a window is never padded.

:class:`LabelledSpanChunks` is the second sampler, and it is the exception to both of
those sentences. It reproduces BABEL's published protocol -- chunks cut inside an
annotated span, short spans looped to fill the window -- so that a number can be quoted
next to their table. Which one a run uses is ``dataloader.chunk_spans``. They are not
comparable to each other and are never mixed in one results table.
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
    load_label_spans,
    load_label_vocabulary_index,
    window_multi_hot,
    window_patch_labels,
)
from sometria.representation import Representation, build_representation


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
            # NaN, not a default, where the corpus has none (amass-smpl): torque per kg of
            # an invented mass is a number with no meaning. Read by forecast_dynamics only.
            "mass": float((item.get("metadata") or {}).get("subject_mass_kg") or "nan"),
        }


def _span_chunks(
    start_frame: int, end_frame: int, window_frames: int
) -> list[tuple[int, int]]:
    """``(start, real_frames)`` for every chunk of one span, BABEL's way.

    Fixed stride inside the span, and the tail chunk is kept rather than shifted back to
    fit -- ``_tiles`` does the opposite, because there a window belongs to a take and may
    overlap its neighbour freely, while here a chunk belongs to a *span* and must not
    reach into the one after it. A span shorter than the window yields exactly one chunk,
    which :func:`_tile_to_window` then fills. An empty span yields nothing, matching their
    "is segment smaller than 1 frame?" guard.
    """

    return [
        (start, min(start + window_frames, end_frame) - start)
        for start in range(start_frame, end_frame, window_frames)
    ]


def _tile_to_window(features: t.Tensor, window_frames: int) -> t.Tensor:
    """Fill a window by looping the chunk's real frames, as BABEL's preprocessing does.

    Their chunker zero-pads (``create_dataset.py``) and their ``pre_normalization`` then
    overwrites every trailing zero with a cyclic repeat of the real frames, so the model
    never sees the padding -- it sees a short motion played on loop, with a discontinuity
    at each wrap whose period tracks the span's duration. Reproducing the second step and
    not the first is the faithful choice: the zeros are an intermediate their released
    features do not contain.

    Cheaper here than there, in one respect worth stating: our features carry ``vel``,
    ``acc`` and ``tau`` computed per frame at ingest, so looping copies each frame's own
    derivatives. Their pipeline stores positions and the wrap is a genuine teleport.
    """

    if features.shape[0] >= window_frames:
        return features[:window_frames]
    repeats = -(-window_frames // features.shape[0])
    return features.repeat((repeats,) + (1,) * (features.ndim - 1))[:window_frames]


class LabelledSpanChunks(t.utils.data.Dataset):
    """BABEL's official sampling: one row per chunk of one annotated span.

    The contrast with :class:`LabelledWindows` is what a row *is*. There, a window is cut
    from the take on a fixed grid and its target is whatever categories cover enough of
    it -- at a 1.34 s median span a 4 s window spans several, so the target is genuinely
    multi-label and no chunk boundary is ever an annotation boundary. Here a chunk is cut
    *inside* one span and inherits that span's categories, so its target is homogeneous by
    construction and the benchmark is the one BABEL published.

    Three consequences, all theirs:

    - A span shorter than the window is looped to fill it (:func:`_tile_to_window`), and
      on ``babel_official/val`` the median chunk is its span repeated ~3.2 times. Loop
      period therefore correlates with span duration, and span duration correlates with
      category -- which is why the results page reports the ``>= window_frames`` subset
      beside the headline.
    - Samples shorter than the window are admitted, not dropped, so the view feeding this
      must be built without ``MotionViewSpec.min_frames``. That is 21% of labelled
      samples, and it is the one place in this repo where a window is not all real motion.
    - A span with no in-vocabulary category contributes no rows at all. Under
      :class:`LabelledWindows` it would be an all-negative window, which is right when a
      window is a slice of time and wrong when a row *is* a labelled span.

    Targets are materialized once into a single array rather than per item. One array
    pickles into a worker as bytes; a tensor per chunk would cost a file descriptor each,
    which is the failure ``load_label_segments`` documents.
    """

    def __init__(
        self,
        motions: MotionDataset,
        spans: dict[str, np.ndarray],
        *,
        num_labels: int,
        window_frames: int,
    ) -> None:
        super().__init__()
        self.motions = motions
        self.num_labels = num_labels
        self.window_frames = window_frames

        index: list[tuple[int, int, int]] = []
        targets: list[np.ndarray] = []
        samples = motions.samples
        for i, (sample_id, n_frames, hz) in enumerate(
            zip(samples["sample_id"], samples["n_frames"], samples["hz"])
        ):
            rows = spans.get(sample_id)
            if rows is None:
                continue

            # A span is a (start, end) pair; its categories are the rows sharing it, which
            # is the `yk` their writer computes and then drops.
            grouped: dict[tuple[float, float], list[int]] = {}
            for start_t, end_t, label in rows:
                grouped.setdefault((float(start_t), float(end_t)), []).append(int(label))

            for (start_t, end_t), labels in grouped.items():
                target = np.zeros(num_labels, dtype=np.float32)
                target[labels] = 1.0
                # int() rather than round(), as in their `int(30.0*seg['start_t'])`.
                start_frame = max(0, int(start_t * hz))
                end_frame = min(int(n_frames), int(end_t * hz))
                for start, real in _span_chunks(start_frame, end_frame, window_frames):
                    index.append((i, start, real))
                    targets.append(target)

        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 3)
        self.targets = (
            np.stack(targets) if targets else np.zeros((0, num_labels), dtype=np.float32)
        )

        # ponytail: same one-entry cache as LabelledWindows, and it pays off for the same
        # reason -- a sample's chunks are contiguous in the index and evaluation does not
        # shuffle. Training does, and then it simply never hits.
        self._cached: tuple[int, dict] | None = None

    @property
    def unlooped_rows(self) -> t.Tensor:
        """Mask over rows whose span filled the window without being looped.

        20% of ``babel_official/val``'s chunks, and the only ones a model cannot read by
        timing the cyclic wrap: loop period *is* span duration, and span duration tracks
        the category. Scored as a second column beside the headline -- if a cell ordering
        survives here it is about motion, and if it does not, it was about periodicity.

        Absolute scores on this subset are not comparable to the headline's. Long spans
        are a different class mix (``walk`` and ``stand``, not ``jump``), so only the
        ordering across cells transfers.
        """

        return t.from_numpy(self.index[:, 2] >= self.window_frames)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        sample_index, start, real = (int(v) for v in self.index[index])
        if self._cached is None or self._cached[0] != sample_index:
            self._cached = (sample_index, self.motions[sample_index])
        item = self._cached[1]

        return {
            "features": _tile_to_window(
                item["features"][start : start + real], self.window_frames
            ),
            "labels": t.from_numpy(self.targets[index]),
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
        # Swaps the whole sampling scheme for BABEL's published one: chunks cut inside
        # annotated spans instead of windows tiled over the take. See LabelledSpanChunks
        # for what changes and why min_frames has to come off when it is set.
        self.chunk_spans = loader.get("chunk_spans", False)

        normalization = loader.get("normalization")
        if normalization is None:
            raise ValueError("config.dataloader.normalization is required.")
        self.normalization_path = Path(normalization)

        human = loader.get("human")
        if human is None:
            raise ValueError("config.dataloader.human is required.")
        self.representation = build_representation(human)

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
            representation=self.representation.name,
            split_set=split.split_set,
            split=split.split,
            source_datasets=tuple(split.get("source_datasets", [])),
            label_sources=tuple(split.get("label_sources", ["BABEL"])),
            require_labels=True,
            exclude_broken=split.get("exclude_broken", True),
            # BABEL's chunker has no length filter -- a one-frame span becomes a full
            # sample -- so the view keeps takes shorter than the window and the chunk
            # dataset loops them up to size.
            min_frames=1 if self.chunk_spans else self.window_frames,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return

        # Chunks are a deterministic enumeration, so training and validation build the
        # same thing and `train_passes` above 1 would only repeat identical rows.
        self.train_dataset = (
            self._chunks(self.train_spec) if self.chunk_spans
            else self._windows(self.train_spec, tiles=False)
        )
        self.val_dataset = (
            self._chunks(self.val_spec) if self.chunk_spans
            else self._windows(self.val_spec, tiles=True)
        )

    def _chunks(self, spec: MotionViewSpec) -> LabelledSpanChunks:
        samples = build_motion_view(self.root_folder, spec)
        spans, num_labels = load_label_spans(
            self.root_folder,
            label_set=self.label_set,
            sample_ids=samples["sample_id"].to_list(),
        )
        self.num_labels = num_labels

        # No annotated_sample_ids filter here: a sample contributes rows exactly when it
        # has an in-vocabulary span, and one with none contributes nothing rather than an
        # all-negative window. Dropping it from the view too just avoids loading it.
        samples = samples.filter(pl.col("sample_id").is_in(list(spans)))
        if samples.is_empty():
            raise ValueError(
                f"no sample in {spec.split_set}/{spec.split} carries a span labelled in "
                f"{self.label_set!r}; check the vocabulary and label_sources"
            )

        return LabelledSpanChunks(
            MotionDataset(
                self.root_folder,
                samples,
                normalization_path=self.normalization_path,
                representation=self.representation,
            ),
            spans,
            num_labels=num_labels,
            window_frames=self.window_frames,
        )

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

    def _loader(self, dataset: t.utils.data.Dataset, shuffle: bool):
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

def _demo() -> None:
    """Check the chunking and the loop against BABEL's arithmetic, by hand."""

    # A 9 s span at 60 Hz is 540 frames: two full chunks and a 60-frame tail that is kept
    # where `_tiles` would have shifted it back to 300.
    assert _span_chunks(0, 540, 240) == [(0, 240), (240, 240), (480, 60)]
    # Offsets are relative to the span, not to the take.
    assert _span_chunks(100, 640, 240) == [(100, 240), (340, 240), (580, 60)]
    # A 1.34 s span -- the median on babel_official/val -- is one chunk, mostly loop.
    assert _span_chunks(0, 80, 240) == [(0, 80)]
    # Their degeneracy guard: a span with no frames in it produces no sample.
    assert _span_chunks(10, 10, 240) == []

    # `n_chunks = ceil(len / window)` is what their chunker computes; ours had better agree.
    for length in (1, 79, 240, 241, 540, 1000):
        assert len(_span_chunks(0, length, 240)) == -(-length // 240), length

    features = t.arange(80, dtype=t.float32)[:, None, None].expand(80, 43, 5)
    window = _tile_to_window(features, 240)
    assert window.shape == (240, 43, 5)
    # Frame 80 is frame 0 again: the wrap their cyclic pad produces, discontinuity and all.
    assert window[:80].equal(features) and window[80:160].equal(features)
    assert float(window[79, 0, 0]) == 79.0 and float(window[80, 0, 0]) == 0.0

    # A tail that does not divide the window is truncated, not padded further.
    assert _tile_to_window(features[:70], 240).shape[0] == 240
    assert float(_tile_to_window(features[:70], 240)[239, 0, 0]) == 239 % 70
    # A full-length chunk is untouched.
    assert _tile_to_window(features.repeat(3, 1, 1), 240).equal(features.repeat(3, 1, 1)[:240])
    print("ok (span chunking and cyclic fill)")


if __name__ == "__main__":
    _demo()
