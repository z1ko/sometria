"""Per-DOF, per-channel normalization statistics over a sample view.

Corpus-level work, unlike ``sometria.preprocess`` which is per-file: ingest is re-run
when the raw data changes, statistics are re-run when the split changes. Always compute
from a train-only view so validation and test distributions never leak into training.

Statistics are saved beside the representation that produced them and passed back into
``Representation.to_model`` at load time. The representation never holds them: the
channel layout and the training-split statistics change on different clocks.
"""

from dataclasses import asdict
from pathlib import Path

import polars as pl
import torch as t
import tqdm

from sometria.catalog import MotionViewSpec, normalization_path
from sometria.representation import Representation


def compute_feature_normalization(
    *,
    output_root: str | Path,
    samples: pl.DataFrame,
    representation: Representation,
    eps: float = 1e-6,
) -> dict:
    """Compute per-DOF, per-channel feature normalization statistics.

    Statistics are accumulated over all frames from the provided sample table and
    returned with shapes broadcastable over sample tensors: ``mean`` and ``std``
    are ``(1, dofs, features)``. Which slots actually get normalized at runtime is
    the representation's business; the mask is copied into the payload only so
    stats files written before ``sometria.representation`` existed stay readable.
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

    mask = t.as_tensor(representation._mask, dtype=t.bool)
    if tuple(mask.shape) != feature_shape:
        raise ValueError(
            f"{representation.name} expects feature shape {tuple(mask.shape)}, "
            f"but the stored tensors have shape {feature_shape}."
        )

    mean = total / n_frames
    variance = (total_sq / n_frames) - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(eps)

    return {
        "mean": mean.unsqueeze(0).to(dtype=t.float32),
        "std": std.unsqueeze(0).to(dtype=t.float32),
        "normalization_mask": mask.unsqueeze(0),
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
    representation: Representation,
    spec: MotionViewSpec,
    eps: float = 1e-6,
) -> Path:
    """Compute and save normalization stats for a named representation/view."""

    stats = compute_feature_normalization(
        output_root=output_root,
        samples=samples,
        representation=representation,
        eps=eps,
    )

    stats |= {
        "name": name,
        "representation": representation.name,
        "view": asdict(spec),
    }

    path = normalization_path(output_root, representation.name, name)
    t.save(stats, path)
    return path
