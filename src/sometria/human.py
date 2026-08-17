
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.signal import resample_poly

# Load the definition of the human model
def _load_human_definition(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f) or {}

# All columns relative to the motion
def _columns_kinematics(human: dict) -> list:
    return [
        f"{dof}{suffix}" for dof in human["dofs"] for suffix in ("", "_vel", "_acc", "_tau")
    ]

# All additional metadata columns
def _columns_metadata(human: dict) -> list:
    return human["metadata"]

# Resolve a dof name (or list of names) from the config into indices along the dof axis
def _dof_indices(human: dict, key: str) -> list[int]:
    index = {dof: i for i, dof in enumerate(human["dofs"])}
    names = human.get(key) or []
    if isinstance(names, str):
        names = [names]
    unknown = [n for n in names if n not in index]
    if unknown:
        raise KeyError(f"{key}: not in dofs: {unknown}")
    return [index[n] for n in names]

# Estimat the capture hz from the time column
def _estimate_hz(time: np.ndarray) -> float:
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
    kept = np.delete(motion, _dof_indices(human, "excluded_dofs"), axis=1)

    out = np.zeros(kept.shape[:2] + (5,), dtype=kept.dtype)
    out[:, :, 0] = np.sin(kept[:, :, 0])
    out[:, :, 1] = np.cos(kept[:, :, 0])
    out[:, :, 2:] = kept[:, :, 1:]
    return out

# Names of the dofs that survive `build_features`, in output order
def feature_dofs(human: dict) -> list[str]:
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
    mask = np.ones((len(feature_dofs(human)), 5), dtype=bool)
    mask[:, :2] = False
    return mask


# Bring a sample onto a uniform rate. 60 Hz leaves 2x headroom over the ~15 Hz the data actually
# carries: the marker trajectories were low-pass filtered upstream before being differentiated, so
# above 30 Hz there is only fitting noise (pos and vel measure exactly 0.000% there, acc and tau
# 0.002% and 0.0004% median).
# resample_poly applies the anti-alias FIR filter, and that part is not optional: naive decimation
# would fold BMLmovi's 30-40 Hz fitting artifact back down into 15-30 Hz, which is signal band.
def _resample_sample(sample: dict, target_hz: float) -> dict:
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