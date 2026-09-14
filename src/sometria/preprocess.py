"""Motion ingest: turning one raw motion file into one stored sample.

Per file: load the OpenSim CSV using the human model definition, resample onto the
project-wide target rate, measure quality on the raw kinematic and dynamic channels,
encode through the representation, and write a tensor plus a catalog row.

Ingest paths are named by *format*, not by corpus. ``import_opensim_csv_dataset`` globs
``*.csv`` and reads ``<dof>_vel`` / ``_acc`` / ``_tau`` columns, which is how every
corpus reaches us once the OpenSim conversion upstream has run.
``import_smpl_dataset`` reads AMASS-style ``.npz`` files instead -- the same takes
*before* that conversion, kept so the conversion itself can be measured. Both share one
loop in ``import_dataset``; only the loader differs.

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
import zipfile

import numpy as np
import polars as pl
from scipy.signal import resample_poly
import torch as t
import tqdm

from sometria.catalog import CATALOG, motion_path, stable_sample_id, upsert_table
from sometria.representation import Representation, SmplRepresentation

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


# SMPL and SMPL-X agree on the first 22 joints (pelvis, then the body chain), and diverge
# after them: SMPL-X continues into hands, jaw and eyes. Reading only the body keeps one
# joint layout across AMASS mirrors, and the joints past it have no OpenSim counterpart.
SMPL_BODY_JOINTS: int = 22


# Load a single sample from an AMASS-style SMPL npz
def _load_smpl_sample(path: str | Path) -> dict:
    """Load one AMASS ``.npz`` into raw axis-angle joint rotations and metadata.

    The returned ``motion`` array has shape ``(time, joints, 3)``: one axis-angle
    rotation per body joint, the raw content of the file. Everything else the file
    carries -- body shape, marker latents, global translation -- is deliberately dropped,
    because the OpenSim tensors this is compared against do not carry it either.
    """

    with np.load(path, allow_pickle=True) as data:
        if "poses" not in data.files:
            raise ValueError("no 'poses' array (AMASS ships shape-only files too).")

        poses = np.asarray(data["poses"], dtype=np.float64)
        # AMASS renamed this key between releases and both mirrors are in the wild.
        hz = float(next(data[k] for k in ("mocap_frame_rate", "mocap_framerate") if k in data.files))

    if poses.shape[1] < SMPL_BODY_JOINTS * 3:
        raise ValueError(f"{poses.shape[1]} pose values, fewer than the body's 66.")
    if len(poses) < 2:
        raise ValueError(f"{len(poses)} frames, too few to difference.")

    motion = poses[:, : SMPL_BODY_JOINTS * 3].reshape(-1, SMPL_BODY_JOINTS, 3)
    n_frames = len(motion)

    return {
        "path": str(path),
        "motion": motion,
        # ponytail: same struct as the CSV import writes, so one catalog holds both. A SMPL
        # file has no subject height or mass -- that comes from the OpenSim scaling step.
        "metadata": {"subject_height_m": None, "subject_mass_kg": None},
        "n_frames": n_frames,
        "duration": n_frames / hz,
        "hz": hz,
        "time": np.arange(n_frames) / hz,
    }


# What "this file is not a take" looks like from below. A corrupt or truncated archive raises
# BadZipFile, which inherits straight from Exception and so slips past ValueError -- one such
# file in a corpus of 19k used to end the whole import at whatever percent it had reached.
# Deliberately not `except Exception`: a typo in a loader should still be loud.
UNREADABLE = (ValueError, OSError, EOFError, zipfile.BadZipFile)


# 5 robust sigmas above the median rate.
TAU_RATE_MAX: float = 3.0e5
TAU_INDEX: int = 3


def measure_quality(motion: np.ndarray, hz: float, tau_rate_max: float = TAU_RATE_MAX) -> dict:
    """Measure basic quality metrics for one raw in-memory motion sample.

    ``broken`` currently means either non-finite values or a torque discontinuity
    whose frame-to-frame rate exceeds ``tau_rate_max``.
    """

    nonfinite = not bool(np.isfinite(motion).all())
    if motion.shape[-1] <= TAU_INDEX:
        # No torque channel to check (SMPL carries kinematics only), so finiteness is the
        # whole test. The tau columns stay in the catalog, as nulls, so one schema serves
        # both imports.
        return {
            "tau_rate": None,
            "tau_absmax": None,
            "nonfinite": nonfinite,
            "broken": nonfinite,
        }

    tau = motion[:, :, TAU_INDEX]
    jump = np.abs(np.diff(tau, axis=0)).max() if len(tau) > 1 else 0.0
    tau_rate = float(jump * hz)
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
    """Import OpenSim-style CSV files. See :func:`import_dataset`."""

    return import_dataset(
        config=config,
        representation=representation,
        load=lambda path: _load_sample(path, human),
    )


def import_smpl_dataset(
    *, config: ImportConfig, representation: SmplRepresentation
) -> pl.DataFrame:
    """Import AMASS-style SMPL ``.npz`` files. See :func:`import_dataset`.

    The point of comparison for the OpenSim import: same corpus, same catalog, same
    windows, but the joint rotations as the mocap fit produced them, without the
    anatomical DOFs and the inverse-dynamics torques the OpenSim conversion adds. Give it
    its own ``source_dataset`` (``"AMASS-SMPL"``) so the two never land in one view --
    they have different feature shapes, and a view is trained on as a single tensor.
    """

    return import_dataset(config=config, representation=representation, load=_load_smpl_sample)


def import_dataset(
    *, config: ImportConfig, representation: Representation, load
) -> pl.DataFrame:
    """Import raw motion files into processed feature tensors and catalog rows.

    ``load`` turns one path into the raw sample dict the rest of the pipeline expects;
    it is the only thing that differs between corpora formats. A file it cannot read is
    skipped and counted rather than raised, because a mirror of 19k files reliably holds a
    few that are not takes at all -- shape-only fits, truncated downloads -- and losing an
    hour of import to the last of them is not a useful way to find out.

    Each matched file becomes one ``.pt`` file under ``motions/<source_dataset>/``.
    The tensor payload stores model-ready ``features`` and ``time``. The catalog
    stores provenance, shape information, timing, metadata, and quality metrics,
    with ``representation`` recording which encoding produced the tensors.
    Existing catalog rows with the same ``sample_id`` are replaced.
    """

    search_path = str(config.input_root / config.pattern)

    files = sorted(Path(p) for p in glob.glob(search_path, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {config.pattern}")

    rows = []
    skipped = []
    for path in tqdm.tqdm(files):
        try:
            sample = load(path)
        except UNREADABLE as error:
            skipped.append(f"{path}: {type(error).__name__}: {error}")
            continue

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

    if skipped:
        print(f"skipped {len(skipped)} unreadable files, e.g. {skipped[:3]}")

    return upsert_table(config.output_root, CATALOG, pl.DataFrame(rows), keys=["sample_id"])
