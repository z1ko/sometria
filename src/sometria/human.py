
from pathlib import Path

import numpy as np
import polars as pl
import yaml

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

# Estimat the capture hz from the time column
def _estimate_hz(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Could not estimate Hz: no positive time deltas.")
    return float(1.0 / np.median(dt))

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

# Apply torque normalization to a sample based on subject mass
def _normalize_sample_torque(sample: dict) -> dict:
    mass = sample["metadata"]["subject_mass_kg"]
    motion = sample["motion"]
    motion[:, :, -1] = motion[:, :, -1] / mass
    return sample

