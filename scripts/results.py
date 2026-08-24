"""Read an ablation off a directory of runs.

    python scripts/results.py runs/mae
    python scripts/results.py runs/mae_probes val/macro_map

One line per run: the best value the metric reached, and which epoch it happened on.
Direction follows the same rule ``train.py`` gives ``ModelCheckpoint`` -- a metric whose
name ends in ``map`` is maximized, everything else is minimized -- so what this reports
is the number the checkpoint was selected on.

Runs are the immediate subdirectories of ``root``; the latest ``version_*`` of each wins,
because a rerun of the same ablation is a correction, not a second sample.
"""

import sys
from pathlib import Path

import polars as pl


def best(run: Path, metric: str) -> tuple[float, int] | None:
    """``(value, epoch)`` for the best epoch of ``metric``, or ``None`` if never logged."""

    versions = sorted(run.glob("lightning_logs/version_*/metrics.csv"))
    if not versions:
        return None

    frame = pl.read_csv(versions[-1], infer_schema_length=None)
    if metric not in frame.columns or "epoch" not in frame.columns:
        return None

    rows = frame.select("epoch", metric).drop_nulls()
    if rows.is_empty():
        return None

    row = rows.sort(metric, descending=metric.endswith("map")).row(0)
    return float(row[1]), int(row[0])


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    metric = sys.argv[2] if len(sys.argv) > 2 else "val/loss"

    if not root.is_dir():
        raise SystemExit(f"no such run directory: {root}")

    found = {run.name: best(run, metric) for run in sorted(root.iterdir()) if run.is_dir()}
    found = {name: value for name, value in found.items() if value is not None}
    if not found:
        raise SystemExit(f"no run under {root} logged {metric!r}")

    width = max(len(name) for name in found)
    print(f"{'run':<{width}}  {metric:>12}  epoch")
    for name, (value, epoch) in sorted(
        found.items(), key=lambda kv: kv[1][0], reverse=metric.endswith("map")
    ):
        print(f"{name:<{width}}  {value:>12.4f}  {epoch:>5}")


if __name__ == "__main__":
    main()
