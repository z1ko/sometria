"""Flag samples whose inverse-dynamics torques are physically impossible.

The `tau` channel blows up where the ID solve hits singular configurations (yoga in MOYO,
marker issues in CMU), producing step discontinuities of up to 1e10 N.m/s and peaks of 2.7e8 N.m.
Kinematics (pos/vel/acc) are unaffected — only `tau` is corrupt.

The metric is max |dtau/dt| in N.m/s, which is rate-normalized so 60Hz and 120Hz samples compare.
Its log10 is roughly normal with median 1.5e4 and robust sigma 0.26 dex; the default threshold is
5 sigma above the median. Nothing separates cleanly, so the metric is stored alongside the flag —
re-threshold with `flag_broken()` instead of rescanning.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch as t

# 5 robust sigmas above the median rate. See module docstring.
TAU_RATE_MAX = 3.0e5

TAU = 3  # channel index of torque in (T, dof, channel)

def scan_quality(root: str | Path = "data/processed") -> pl.DataFrame:
    """Read every sample tensor and measure torque sanity. Takes a couple of minutes."""
    root = Path(root)
    samples = pl.read_parquet(root / "samples.parquet")

    rows = []
    for i, r in enumerate(samples.iter_rows(named=True), start=1):
        motion = t.load(root / r["sample_path"], weights_only=True)["motion"].numpy()
        tau = motion[:, :, TAU]
        jump = np.abs(np.diff(tau, axis=0)).max() if len(tau) > 1 else 0.0
        rows.append(
            {
                "sample": r["sample"],
                "tau_rate": float(jump * r["hz"]),
                "tau_absmax": float(np.abs(tau).max()),
                "nonfinite": not bool(np.isfinite(motion).all()),
            }
        )
        if i % 4000 == 0:
            print(f"Scanned {i}/{len(samples)} samples")

    return pl.DataFrame(rows)


def flag_broken(samples: pl.DataFrame, tau_rate_max: float = TAU_RATE_MAX) -> pl.DataFrame:
    return samples.with_columns(
        ((pl.col("tau_rate") > tau_rate_max) | pl.col("nonfinite")).alias("broken")
    )


def enrich_quality(root: str | Path = "data/processed") -> pl.DataFrame:
    """Write `tau_rate`, `tau_absmax`, `nonfinite` and `broken` into samples.parquet."""
    root = Path(root)
    samples = pl.read_parquet(root / "samples.parquet").drop(
        ["tau_rate", "tau_absmax", "nonfinite", "broken"], strict=False
    )
    df = flag_broken(samples.join(scan_quality(root), on="sample"))
    df.write_parquet(root / "samples.parquet")
    return df


if __name__ == "__main__":
    df = enrich_quality()
    print(df.group_by("split", "broken").len().sort("split", "broken"))
    print(f"broken: {df['broken'].sum()}/{len(df)} samples")
