"""Run with: python tests/test_smpl.py"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sometria.preprocess import SMPL_BODY_JOINTS, _load_smpl_sample, measure_quality
from sometria.representation import SmplRepresentation, build_representation

REP = build_representation(ROOT / "config/smpl.yaml")


def _rotations(frames=16, joints=SMPL_BODY_JOINTS, seed=0):
    """Random axis-angle joint rotations, well inside the +-pi shell where log is unique."""

    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(frames, joints, 3))


def test_config_builds_the_smpl_layout():
    assert isinstance(REP, SmplRepresentation)
    assert len(REP.dofs) == SMPL_BODY_JOINTS - 1     # the pelvis carries global orientation
    assert REP.dofs[0] == "left_hip"
    assert REP.indices("rot0", "vel0", "acc0") == (0, 6, 12)


def test_pose_channels_are_left_uncompressed_and_unnormalized():
    # Rotation matrix entries are already bounded; only the derivatives get signed_log.
    assert not REP._mask[:, :6].any()
    assert REP._mask[:, 6:].all()


def test_encode_decode_recovers_the_rotations():
    motion = _rotations()
    features = REP.encode(motion)
    assert features.shape == (len(motion), len(REP.dofs), 18)
    assert np.abs(features[:, :, :6]).max() <= 1.0                  # rot6d stays bounded

    recovered = REP.decode(features)
    assert np.allclose(recovered, motion[:, 1:], atol=1e-8)         # pelvis is gone for good


def test_encode_is_continuous_across_the_axis_angle_wrap():
    # Axis-angle jumps by 2pi across the pi shell; the 6D form must not.
    axis = np.array([0.0, 0.0, 1.0])
    motion = np.zeros((2, SMPL_BODY_JOINTS, 3))
    motion[0, 1] = axis * (np.pi - 1e-6)
    motion[1, 1] = axis * (-np.pi + 1e-6)

    features = REP.encode(motion)
    assert np.abs(np.diff(features[:, :, :6], axis=0)).max() < 1e-5


def test_quality_reports_no_torque_metrics_for_smpl():
    quality = measure_quality(_rotations(), hz=60.0)
    assert quality["tau_rate"] is None and not quality["broken"]

    broken = _rotations()
    broken[3, 2, 0] = np.nan
    assert measure_quality(broken, hz=60.0)["broken"]


def test_loader_rejects_files_that_are_not_takes(tmp=None):
    path = Path(__file__).with_name("_smpl_fixture.npz")
    try:
        np.savez(path, betas=np.zeros(16), mocap_frame_rate=np.float64(120.0))
        try:
            _load_smpl_sample(path)
        except ValueError:
            pass
        else:
            raise AssertionError("a file without 'poses' should raise")

        np.savez(path, poses=np.zeros((5, 165)), mocap_frame_rate=np.float64(120.0))
        sample = _load_smpl_sample(path)
        assert sample["motion"].shape == (5, SMPL_BODY_JOINTS, 3)
        assert sample["hz"] == 120.0 and sample["duration"] == 5 / 120.0
    finally:
        path.unlink(missing_ok=True)


def test_a_corrupt_file_does_not_end_the_import():
    """One truncated .npz in 19k used to end the run at whatever percent it had reached."""

    import tempfile

    from sometria.preprocess import ImportConfig, import_smpl_dataset

    raw, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    (raw / "truncated.npz").write_bytes(b"PK\x03\x04 not really a zip")
    (raw / "empty.npz").write_bytes(b"")
    np.savez(raw / "good.npz", poses=np.zeros((600, 165)), mocap_frame_rate=np.float64(120.0))

    catalog = import_smpl_dataset(
        config=ImportConfig(
            source_dataset="AMASS-SMPL", input_root=raw, output_root=out, pattern="*.npz"
        ),
        representation=REP,
    )

    assert catalog["source_path"].to_list() == ["good.npz"]
    assert catalog["n_dofs"][0] == len(REP.dofs) and catalog["n_features"][0] == 18


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
