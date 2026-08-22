
"""Preprocessing pipeline for turning raw motion files into Sometria datasets.

This module owns the work that happens before training:

- load raw OpenSim CSV files using the human model definition;
- resample motion to the project-wide target rate;
- measure sample quality on the raw kinematic/dynamic channels;
- convert raw channels into model-ready features;
- write processed tensors under ``motions/``;
- update catalog/split tables under ``tables/``;
- compute and save normalization statistics under ``stats/``.

Dataset-specific importers belong here when they translate an external dataset
format into the shared processed layout. Generic table filtering belongs in
``sometria.catalog`` and runtime normalization application belongs in
``sometria.dataset``.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from scipy.signal import resample_poly
from fractions import Fraction

import torch as t
import numpy as np
import polars as pl
import glob
import tqdm
import yaml

from sometria.catalog import CATALOG, SPLITS
from sometria.splits import babel_key, sample_key

# NOTE: Hardcoded paths, no need for more complexity
PATH_HUMAN_DEFINITION : Path = Path("config/human.yaml")
PATH_OUTPUT_ROOT: Path = Path("data/processed")

# Uniform rate for every sample. See _resample_sample for why 60 Hz.
TARGET_HZ: float = 60.0

# Load the definition of the human model
def _load_human_definition(path: str | Path) -> dict:
    """Load the YAML description of the OpenSim human model columns."""

    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f) or {}


# All columns relative to the motion
def _columns_kinematics(human: dict) -> list:
    """Return CSV columns containing position, velocity, acceleration, and torque."""

    return [
        f"{dof}{suffix}" for dof in human["dofs"] for suffix in ("", "_vel", "_acc", "_tau")
    ]


# All additional metadata columns
def _columns_metadata(human: dict) -> list:
    """Return CSV columns copied as per-sample metadata."""

    return human["metadata"]


# Resolve a dof name (or list of names) from the config into indices along the dof axis
def _dof_indices(human: dict, key: str) -> list[int]:
    """Resolve DOF names from the human config into integer axis indices."""

    index = {dof: i for i, dof in enumerate(human["dofs"])}
    names = human.get(key) or []
    if isinstance(names, str):
        names = [names]
    unknown = [n for n in names if n not in index]
    if unknown:
        raise KeyError(f"{key}: not in dofs: {unknown}")
    return [index[n] for n in names]


# Estimate the capture hz from the time column
def _estimate_hz(time: np.ndarray) -> float:
    """Estimate the sample rate from a monotonically increasing time column."""

    time = np.asarray(time, dtype=np.float64)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Could not estimate Hz: no positive time deltas.")
    return float(1.0 / np.median(dt))


# Build human model features: (T, dofs, 4) -> (T, kept dofs, 5)
# Every kept dof is an angle, so all of them get the same [sin, cos, vel, acc, tau] slots and the
# dof axis stays a clean per-joint token space. The root translations are excluded.
def build_features(motion: np.ndarray, human: dict) -> np.ndarray:
    """Convert raw OpenSim channels into the current model representation.

    Input shape is ``(time, dofs, 4)`` with channels ``position, velocity,
    acceleration, torque``. The output shape is ``(time, kept_dofs, 5)`` with
    channels ``sin(position), cos(position), velocity, acceleration, torque``.
    Root translations are expected to be listed in ``excluded_dofs`` because
    they are distances, not angles.
    """

    kept = np.delete(motion, _dof_indices(human, "excluded_dofs"), axis=1)

    out = np.zeros(kept.shape[:2] + (5,), dtype=kept.dtype)
    out[:, :, 0] = np.sin(kept[:, :, 0])
    out[:, :, 1] = np.cos(kept[:, :, 0])
    out[:, :, 2:] = kept[:, :, 1:]
    return out


# Names of the dofs that survive `build_features`, in output order
def feature_dofs(human: dict) -> list[str]:
    """Return DOF names in the same order as the feature tensor's DOF axis."""

    excluded = set(_dof_indices(human, "excluded_dofs"))
    translations = set(_dof_indices(human, "root_position_dofs"))
    kept = [(i, dof) for i, dof in enumerate(human["dofs"]) if i not in excluded]

    still_there = [dof for i, dof in kept if i in translations]
    if still_there:
        raise ValueError(
            f"root_position_dofs are distances, not angles, so build_features cannot encode them "
            f"as sin/cos: {still_there}. Either add them to excluded_dofs, or give them their own "
            f"branch in build_features."
        )
    return [dof for _, dof in kept]


# Which feature slots get the signed log. The position slots are sin/cos, already bounded in
# [-1, 1], so only the derivative channels need compressing.
def feature_log_mask(human: dict) -> np.ndarray:
    """Return a mask for channels that may benefit from signed-log scaling."""

    mask = np.ones((len(feature_dofs(human)), 5), dtype=bool)
    mask[:, :2] = False
    return mask

# Bring a sample onto a uniform rate. 60 Hz leaves 2x headroom over the ~15 Hz the data actually
# carries: the marker trajectories were low-pass filtered upstream before being differentiated, so
# above 30 Hz there is only fitting noise (pos and vel measure exactly 0.000% there, acc and tau
# 0.002% and 0.0004% median).
def _resample_sample(sample: dict, target_hz: float) -> dict:
    """Resample one loaded sample to ``target_hz`` while preserving its schema."""

    ratio = Fraction(target_hz / round(sample["hz"])).limit_denominator(50)
    if ratio == 1:
        return sample | {"original_hz": sample["hz"], "hz": float(target_hz)}

    motion = sample["motion"]
    frames, dofs, channels = motion.shape
    motion = resample_poly(
        motion.reshape(frames, dofs * channels), ratio.numerator, ratio.denominator, axis=0
    ).reshape(-1, dofs, channels)

    n_frames = len(motion)
    time = sample["time"][0] + np.arange(n_frames) / target_hz
    return sample | {
        "motion": motion,
        "time": time,
        "n_frames": n_frames,
        "original_hz": sample["hz"],
        "duration": n_frames / target_hz,
        "hz": float(target_hz),
    }


# Load a single sample from a csv
def _load_sample(path: str | Path, human: dict) -> dict:
    """Load one OpenSim CSV into raw arrays and per-sample metadata.

    The returned ``motion`` array has shape ``(time, dofs, channels)`` and still
    contains the raw channels. Feature conversion is deliberately separate so
    quality checks can inspect raw torques.
    """

    num_channels = len(human["channels"])
    num_dofs = len(human["dofs"])

    lf = pl.scan_csv(path)
    
    motion   = lf.select(_columns_kinematics(human)).collect().to_numpy().reshape(-1, num_dofs, num_channels)
    metadata = lf.select(_columns_metadata(human)).first().collect().row(0, named=True)
    time     = lf.select("time").collect().to_numpy().reshape(-1)

    n_frames = len(time)
    hz = _estimate_hz(time)
    duration = n_frames / hz

    return {
        "path": str(path),
        "motion": motion,
        "metadata": metadata,
        "n_frames": n_frames,
        "duration": duration,
        "hz": hz,
        "time": time
    }


# 5 robust sigmas above the median rate.
TAU_RATE_MAX: float = 3.0e5
TAU_INDEX: int = 3


def measure_quality(motion: np.ndarray, hz: float, tau_rate_max: float = TAU_RATE_MAX) -> dict:
    """Measure basic quality metrics for one raw in-memory motion sample.

    ``broken`` currently means either non-finite values or a torque discontinuity
    whose frame-to-frame rate exceeds ``tau_rate_max``.
    """

    tau = motion[:, :, TAU_INDEX]
    jump = np.abs(np.diff(tau, axis=0)).max() if len(tau) > 1 else 0.0
    tau_rate = float(jump * hz)
    nonfinite = not bool(np.isfinite(motion).all())
    return {
        "tau_rate": tau_rate,
        "tau_absmax": float(np.abs(tau).max()),
        "nonfinite": nonfinite,
        "broken": tau_rate > tau_rate_max or nonfinite,
    }

MOTIONS_DIR = "motions"
TABLES_DIR = "tables"
STATS_DIR = "stats"

@dataclass(frozen=True)
class ImportConfig:
    """Configuration for importing one raw dataset into the processed layout."""

    source_dataset: str     # "AMASS", "MotionX", ...
    representation: str     # "opensim_49dof_tau", "smplx", ...
    input_root: Path
    output_root: Path
    pattern: str
    target_hz: float = TARGET_HZ

def stable_sample_id(source_dataset: str, source_path: str) -> str:
    """Return the stable key used to join catalogs, labels, and splits."""

    return f"{source_dataset}:{source_path}"

def _safe_name(sample_id: str) -> str:
    """Return a compact deterministic filename stem for a sample id."""

    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]

def _table_path(output_root: Path, name: str) -> Path:
    """Return and create the path for a processed table artifact."""

    path = output_root / TABLES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _normalization_path(output_root: Path, representation: str, name: str) -> Path:
    """Return and create the path for a normalization stats artifact."""

    path = output_root / STATS_DIR / representation / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _upsert_table(path: Path, df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Insert or replace rows in a parquet table using ``keys`` as identity."""

    if path.exists():
        old = pl.read_parquet(path)
        df = pl.concat([old, df], how="diagonal_relaxed")
        df = df.unique(subset=keys, keep="last")

    df.write_parquet(path)
    return df

def import_opensim_csv_dataset(*, config: ImportConfig, human: dict) -> pl.DataFrame:
    """Import OpenSim-style CSV files into processed feature tensors and catalog rows.

    Each matched CSV becomes one ``.pt`` file under ``motions/<source_dataset>/``.
    The tensor payload stores model-ready ``features`` and ``time``. The catalog
    stores provenance, shape information, timing, metadata, and quality metrics.
    Existing catalog rows with the same ``sample_id`` are replaced.
    """

    search_path = str(config.input_root / config.pattern)

    files = sorted(Path(p) for p in glob.glob(search_path, recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {config.pattern}")

    motion_dir = config.output_root / MOTIONS_DIR / config.source_dataset.lower()
    motion_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(tqdm.tqdm(files), start=1):
        sample = _load_sample(path, human)
        sample = _resample_sample(sample, config.target_hz)

        source_path = str(path.relative_to(config.input_root))
        sample_id = stable_sample_id(config.source_dataset, source_path)

        quality = measure_quality(sample["motion"], sample["hz"])
        features = build_features(sample["motion"], human)

        # Save model-ready feature tensor. Quality is still measured on raw motion above.
        motion_path = motion_dir / f"{_safe_name(sample_id)}.pt"
        t.save(
            {
                "features": t.tensor(features, dtype=t.float32),
                "time":   t.tensor(sample["time"]),
            },
            motion_path,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": config.source_dataset,
                "source_subset": source_path.split("/")[0],
                "source_path": source_path,
                "motion_path": str(motion_path.relative_to(config.output_root)),
                "representation": config.representation,
                "n_dofs": features.shape[1],
                "n_features": features.shape[2],
                "metadata": sample["metadata"],
                "hz": sample["hz"],
                "original_hz": sample["original_hz"],
                "n_frames": sample["n_frames"],
                "duration": sample["duration"],
                **quality,
            }
        )

    catalog = pl.DataFrame(rows)
    return _upsert_table(
        _table_path(config.output_root, CATALOG),
        catalog,
        keys=["sample_id"],
    )


def import_babel_splits(
    *,
    output_root: str | Path,
    babel_root: str | Path,
    source_dataset: str = "AMASS",
    split_set: str = "babel_official"
) -> pl.DataFrame:
    """Import BABEL's official split membership for already-imported AMASS rows.

    BABEL paths are normalized through ``sometria.splits`` and matched against
    catalog ``source_path`` values. This records split membership only; labels
    are intentionally separate so future label sources can use the same catalog.
    """

    output_root = Path(output_root)
    babel_root = Path(babel_root)

    catalog = pl.read_parquet(_table_path(output_root, CATALOG))
    catalog_keys = catalog.select(
        "sample_id",
        "source_path",
        pl.col("source_path")
        .map_elements(sample_key, return_dtype=pl.String)
        .alias("babel_match_key"),
    )

    rows = []
    for split in ("train", "val", "test"):
        data = json.loads((babel_root / f"{split}.json").read_text())
        for seq in data.values():
            rows.append(
                {
                    "babel_match_key": babel_key(seq["feat_p"]),
                    "split_set": split_set,
                    "split": split,
                    "label_source": "BABEL",
                    "babel_sid": seq["babel_sid"],
                }
            )

    babel = pl.DataFrame(rows)
    splits = (
        catalog_keys
        .join(babel, on="babel_match_key", how="inner")
        .select("sample_id", "split_set", "split", "label_source", "babel_sid")
    )

    return _upsert_table(
        _table_path(output_root, SPLITS),
        splits,
        keys=["sample_id", "split_set"],
    )


def create_pretrain_split(
    *,
    output_root: str | Path,
    split_set: str = "pretrain_v1",
) -> pl.DataFrame:
    """Create the default self-supervised pretraining split.

    The current policy includes BABEL train samples plus AMASS samples with no
    BABEL official split membership. BABEL validation and test samples are kept
    out so they remain usable as held-out views.
    """

    output_root = Path(output_root)

    catalog = pl.read_parquet(_table_path(output_root, CATALOG))
    splits = pl.read_parquet(_table_path(output_root, SPLITS))

    babel_train_ids = splits.filter(
        (pl.col("split_set") == "babel_official")
        & (pl.col("split") == "train")
    ).select("sample_id")

    babel_any_ids = splits.filter(
        pl.col("split_set") == "babel_official"
    ).select("sample_id").unique()

    unlabelled_ids = catalog.join(babel_any_ids, on="sample_id", how="anti").select("sample_id")

    pretrain = (
        pl.concat([babel_train_ids, unlabelled_ids])
        .unique()
        .with_columns(
            pl.lit(split_set).alias("split_set"),
            pl.lit("train").alias("split"),
            pl.lit(None).cast(pl.String).alias("label_source"),
            pl.lit(None).cast(pl.Int64).alias("babel_sid"),
        )
    )

    return _upsert_table(
        _table_path(output_root, SPLITS),
        pretrain,
        keys=["sample_id", "split_set"],
    )


def compute_feature_normalization(
    *,
    output_root: str | Path,
    samples: pl.DataFrame,
    eps: float = 1e-6,
) -> dict:
    """Compute per-DOF, per-channel feature normalization statistics.

    Statistics are accumulated over all frames from the provided sample table and
    returned with shapes broadcastable over sample tensors: ``mean`` and ``std``
    are ``(1, dofs, features)``. Use a train-only view here to avoid leaking
    validation or test distributions into training.
    """

    output_root = Path(output_root)
    if samples.is_empty():
        raise ValueError("Cannot compute normalization stats from an empty sample table.")

    total = None
    total_sq = None
    n_frames = 0
    n_samples = 0
    feature_shape = None

    for row in tqdm.tqdm(samples.iter_rows(named=True), total=len(samples)):
        payload = t.load(output_root / row["motion_path"], weights_only=True)
        if "features" not in payload:
            raise KeyError(
                f"{row['motion_path']} does not contain 'features'. "
                "Rerun preprocessing before computing normalization."
            )

        features = payload["features"].to(dtype=t.float64)
        if features.ndim != 3:
            raise ValueError(
                f"{row['motion_path']} has feature shape {tuple(features.shape)}; expected (T, dofs, features)."
            )
        if not t.isfinite(features).all():
            raise ValueError(f"{row['motion_path']} contains non-finite features.")

        current_shape = tuple(features.shape[1:])
        if feature_shape is None:
            feature_shape = current_shape
            total = t.zeros(feature_shape, dtype=t.float64)
            total_sq = t.zeros(feature_shape, dtype=t.float64)
        elif current_shape != feature_shape:
            raise ValueError(
                f"{row['motion_path']} has feature shape {current_shape}, "
                f"but previous samples had {feature_shape}."
            )

        total += features.sum(dim=0)
        total_sq += features.square().sum(dim=0)
        n_frames += features.shape[0]
        n_samples += 1

    assert total is not None
    assert total_sq is not None
    assert feature_shape is not None

    mean = total / n_frames
    variance = (total_sq / n_frames) - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(eps)

    return {
        "mean": mean.unsqueeze(0).to(dtype=t.float32),
        "std": std.unsqueeze(0).to(dtype=t.float32),
        "n_frames": n_frames,
        "n_samples": n_samples,
        "feature_shape": feature_shape,
        "eps": eps,
    }


def save_feature_normalization(
    *,
    output_root: str | Path,
    samples: pl.DataFrame,
    name: str,
    representation: str,
    split_set: str | None = None,
    split: str | None = None,
    eps: float = 1e-6,
) -> Path:
    """Compute and save normalization stats for a named representation/view."""

    output_root = Path(output_root)
    stats = compute_feature_normalization(
        output_root=output_root,
        samples=samples,
        eps=eps,
    )
    stats |= {
        "name": name,
        "representation": representation,
        "split_set": split_set,
        "split": split,
    }

    path = _normalization_path(output_root, representation, name)
    t.save(stats, path)
    return path
