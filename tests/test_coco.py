"""Run with: python tests/test_coco.py"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sometria.preprocess import ImportConfig, import_coco_dataset
from sometria.representation import CocoRepresentation, build_representation

HUMAN = yaml.safe_load((ROOT / "config/coco.yaml").read_text())
REP = build_representation(ROOT / "config/coco.yaml")


def _keypoints(frames=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(frames, len(HUMAN["joints"]), 3)) * 0.3 + [0.0, 0.0, 1.0]


def test_config_builds_the_coco_layout():
    assert isinstance(REP, CocoRepresentation)
    assert len(REP.dofs) == 18 and REP.indices("pos0", "vel0", "acc0") == (0, 3, 6)
    assert not REP._mask[:, :3].any() and REP._mask[:, 3:].all()


def test_encode_ignores_global_placement_and_heading():
    motion = _keypoints()
    yaw = np.linspace(0, 4, len(motion))[:, None]           # heading that changes every frame
    c, s = np.cos(yaw), np.sin(yaw)
    moved = np.stack([c * motion[..., 0] - s * motion[..., 1],
                      s * motion[..., 0] + c * motion[..., 1], motion[..., 2]], -1)
    moved = moved + np.linspace(0, 5, len(motion))[:, None, None]

    features = REP.encode(motion)
    assert features.shape == (len(motion), 18, 9)
    assert np.allclose(features, REP.encode(moved), atol=1e-9)

    pos = REP.decode(features)
    left, right = pos[:, HUMAN["joints"].index("left_hip")], pos[:, HUMAN["joints"].index("right_hip")]
    assert np.allclose(left + right, 0, atol=1e-9)           # hip midpoint at the origin
    assert np.allclose((left - right)[:, 1], 0, atol=1e-9)   # hip axis along +x
    assert ((left - right)[:, 0] >= 0).all()


def test_import_reads_csv_and_skips_bad_files():
    raw, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    motion = _keypoints(frames=600)
    columns = {"frame": np.arange(600), "time": np.arange(600) / 120.0}
    for j, joint in enumerate(HUMAN["joints"]):
        for a, axis in enumerate("xyz"):
            columns[f"{joint}_{axis}"] = motion[:, j, a]
    pl.DataFrame(columns).write_csv(raw / "good_stageii.csv")
    pl.DataFrame({"frame": [0, 1], "time": [0.0, 0.1]}).write_csv(raw / "no_joints.csv")

    catalog = import_coco_dataset(
        config=ImportConfig(source_dataset="AMASS-COCO", input_root=raw, output_root=out, pattern="*.csv"),
        human=HUMAN,
        representation=REP,
    )
    row = catalog.row(0, named=True)
    assert catalog["source_path"].to_list() == ["good_stageii.csv"]
    assert (row["n_dofs"], row["n_features"], row["n_frames"]) == (18, 9, 300)
    assert abs(row["original_hz"] - 120.0) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
