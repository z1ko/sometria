"""Run with: python tests/test_representation.py"""

import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.representation import Representation, signed_exp, signed_log

REP = Representation.from_config(Path(__file__).resolve().parents[1] / "config/human.yaml")


def _raw_motion(frames=32, dofs=None, seed=0):
    """Random raw OpenSim channels: angles in (-pi, pi], heavy-tailed derivatives."""

    rng = np.random.default_rng(seed)
    dofs = dofs if dofs is not None else len(REP.dofs) + len(REP._excluded)
    motion = rng.normal(scale=200.0, size=(frames, dofs, 4))
    motion[:, :, 0] = rng.uniform(-np.pi, np.pi, size=(frames, dofs))
    return motion


def test_signed_log_roundtrip():
    x = np.array([-1e4, -517.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 517.0, 1e4])
    assert np.allclose(signed_exp(signed_log(x)), x, rtol=1e-9)
    assert np.all(np.sign(signed_log(x)) == np.sign(x))          # sign preserved
    assert np.all(np.diff(signed_log(x)) > 0)                    # monotone
    assert abs(signed_log(np.array([1e-6]))[0] - 1e-6) < 1e-12   # near-identity at 0


def test_encode_applies_log_only_to_derivative_channels():
    motion = _raw_motion()
    features = REP.encode(motion)
    assert features.shape == (len(motion), len(REP.dofs), len(REP.channels))

    kept = np.delete(motion, REP._excluded, axis=1)

    # sin/cos untouched, and still bounded
    assert np.allclose(features[:, :, 0], np.sin(kept[:, :, 0]))
    assert np.allclose(features[:, :, 1], np.cos(kept[:, :, 0]))

    # vel/acc/tau compressed and exactly invertible back to the raw channels
    assert np.allclose(signed_exp(features[:, :, 2:]), kept[:, :, 1:])
    assert np.abs(features[:, :, 2:]).max() < np.abs(kept[:, :, 1:]).max()

    # the mask is what decides, so a mask change cannot silently skip the transform
    assert REP._mask[:, :2].sum() == 0
    assert REP._mask[:, 2:].all()


def test_log_compresses_the_tail_that_dominated_the_loss():
    # one 84-sigma torque spike in an otherwise ordinary signal
    raw = np.concatenate([np.random.default_rng(1).normal(size=999), [84.0]])
    before = raw.std()
    after = signed_log(raw).std()
    assert after < before / 2


def test_decode_inverts_encode_on_the_kept_dofs():
    motion = _raw_motion()
    kept = np.delete(motion, REP._excluded, axis=1)
    back = REP.decode(REP.encode(motion))

    assert back.shape == kept.shape
    assert np.allclose(back[:, :, 1:], kept[:, :, 1:], rtol=1e-9)  # vel/acc/tau exact
    assert np.allclose(back[:, :, 0], kept[:, :, 0], atol=1e-12)   # angle, already wrapped

    # angles outside (-pi, pi] come back wrapped, not equal: sin/cos cannot say which turn
    wrapped = REP.decode(REP.encode(motion + np.array([2 * np.pi, 0, 0, 0])))
    assert np.allclose(wrapped[:, :, 0], kept[:, :, 0], atol=1e-9)


def test_to_model_and_to_physical_are_inverses_and_skip_sin_cos():
    features = t.tensor(REP.encode(_raw_motion(frames=128)), dtype=t.float64)
    stats = {
        "mean": features.mean(0, keepdim=True),
        "std": features.std(0, keepdim=True).clamp_min(1e-6),
    }

    model_input = REP.to_model(features, stats)
    assert t.equal(model_input[:, :, :2], features[:, :, :2])       # sin/cos pass through
    assert model_input[:, :, 2:].mean(0).abs().max() < 1e-8      # centered where it applies

    assert t.allclose(REP.to_physical(model_input, stats), features, atol=1e-10)


def test_indices_names_the_channels_the_loss_uses():
    assert REP.indices("vel", "acc", "tau") == (2, 3, 4)
    assert REP.indices("sin") == (0,)
    try:
        REP.indices("torque")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown channel name should raise")


def test_dofs_drop_the_excluded_ones_and_keep_feature_order():
    assert len(REP.dofs) == 43
    assert len(set(REP.dofs)) == len(REP.dofs)
    assert REP.dofs[0] == "hip_flexion_r"               # the six pelvis dofs are excluded
    assert not set(REP.dofs) & {"pelvis_tilt", "pelvis_tx", "pelvis_tz"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
