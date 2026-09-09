"""Motion ingest: turning one raw motion file into one stored sample.

Per file: load the OpenSim CSV using the human model definition, resample onto the
project-wide target rate, measure quality on the raw kinematic and dynamic channels,
encode through the representation, and write a tensor plus a catalog row.

There is exactly one ingest path, because every corpus reaches us as OpenSim CSV --
the conversion happens upstream. ``import_opensim_csv_dataset`` is specific about
*format*, not about *corpus*: it globs ``*.csv`` and reads ``<dof>_vel`` / ``_acc`` /
``_tau`` columns.

Neighbouring concerns live elsewhere. What the feature channels *mean* belongs to
``sometria.representation``; where things are stored and which samples constitute a
view belong to ``sometria.catalog``; corpus-level statistics belong to
``sometria.normalization``; a label corpus belongs in its own module, as
``sometria.babel``.
"""

from dataclasses import dataclass
from fractions import Fraction
import glob
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import resample_poly
import torch as t
import tqdm

from sometria.catalog import CATALOG, motion_path, stable_sample_id, upsert_table
from sometria.representation import Representation

# NOTE: Hardcoded paths, no need for more complexity
PATH_HUMAN_DEFINITION: Path = Path("config/human.yaml")
PATH_OUTPUT_ROOT: Path = Path("data/processed")

# Uniform rate for every sample. See _resample_sample for why 60 Hz.
TARGET_HZ: float = 60.0


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


# Estimate the capture hz from the time column
def _estimate_hz(time: np.ndarray) -> float:
    """Estimate the sample rate from a monotonically increasing time column."""

    time = np.asarray(time, dtype=np.float64)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Could not estimate Hz: no positive time deltas.")
    return float(1.0 / np.median(dt))


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


def _source_subset(source_path: str) -> str:
    """Return the corpus subset a relative source path belongs to.

    AMASS and MotionX are laid out as ``<subset>/<...>/<take>.csv``, so the leading
    directory names the subset. CARE-PD ships flat, encoding the same thing as a
    ``<subset>__<subject>__<take>.csv`` filename prefix.
    """

    head = source_path.split("/")[0]
    # ponytail: two layouts, one line. Add a hook on ImportConfig if a third appears.
    return head if "/" in source_path else head.split("__")[0]


@dataclass(frozen=True)
class ImportConfig:
    """Configuration for importing one raw dataset into the processed layout."""

    source_dataset: str     # "AMASS", "MotionX", ...
    input_root: Path
    output_root: Path
    pattern: str
    target_hz: float = TARGET_HZ


def import_opensim_csv_dataset(
    *, config: ImportConfig, human: dict, representation: Representation
) -> pl.DataFrame:
    """Import OpenSim-style CSV files into processed feature tensors and catalog rows.

    Each matched CSV becomes one ``.pt`` file under ``motions/<source_dataset>/``.
    The tensor payload stores model-ready ``features`` and ``time``. The catalog
    stores provenance, shape information, timing, metadata, and quality metrics,
    with ``representation`` recording which encoding produced the tensors.
    Existing catalog rows with the same ``sample_id`` are replaced.
    """

    search_path = str(config.input_root / config.pattern)

    files = sorted(Path(p) for p in glob.glob(search_path, recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {config.pattern}")

    rows = []
    for path in tqdm.tqdm(files):
        sample = _load_sample(path, human)
        sample = _resample_sample(sample, config.target_hz)

        source_path = str(path.relative_to(config.input_root))
        sample_id = stable_sample_id(config.source_dataset, source_path)

        quality = measure_quality(sample["motion"], sample["hz"])
        features = representation.encode(sample["motion"])

        # Save model-ready feature tensor. Quality is still measured on raw motion above.
        tensor_path = motion_path(config.output_root, config.source_dataset, sample_id)
        t.save(
            {
                "features": t.tensor(features, dtype=t.float32),
                "time":   t.tensor(sample["time"]),
            },
            tensor_path,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": config.source_dataset,
                "source_subset": _source_subset(source_path),
                "source_path": source_path,
                "motion_path": str(tensor_path.relative_to(config.output_root)),
                "representation": representation.name,
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

    return upsert_table(config.output_root, CATALOG, pl.DataFrame(rows), keys=["sample_id"])
