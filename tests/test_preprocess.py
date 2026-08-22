"""Run with: python tests/test_preprocess.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.preprocess import (
    _load_human_definition,
    build_features,
    feature_dofs,
    feature_log_mask,
    signed_exp,
    signed_log,
)

HUMAN = _load_human_definition(Path(__file__).resolve().parents[1] / "config/human.yaml")


def test_signed_log_roundtrip():
    x = np.array([-1e4, -517.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 517.0, 1e4])
    assert np.allclose(signed_exp(signed_log(x)), x, rtol=1e-9)
    assert np.all(np.sign(signed_log(x)) == np.sign(x))          # sign preserved
    assert np.all(np.diff(signed_log(x)) > 0)                    # monotone
    assert abs(signed_log(np.array([1e-6]))[0] - 1e-6) < 1e-12   # near-identity at 0


def test_build_features_applies_log_only_to_derivative_channels():
    rng = np.random.default_rng(0)
    frames, dofs = 32, len(HUMAN["dofs"])
    motion = rng.normal(scale=200.0, size=(frames, dofs, 4))
    motion[:, :, 0] = rng.uniform(-np.pi, np.pi, size=(frames, dofs))

    features = build_features(motion, HUMAN)
    assert features.shape == (frames, len(feature_dofs(HUMAN)), 5)

    kept = np.delete(motion, [i for i, d in enumerate(HUMAN["dofs"]) if d not in feature_dofs(HUMAN)], axis=1)

    # sin/cos untouched, and still bounded
    assert np.allclose(features[:, :, 0], np.sin(kept[:, :, 0]))
    assert np.allclose(features[:, :, 1], np.cos(kept[:, :, 0]))

    # vel/acc/tau compressed and exactly invertible back to the raw channels
    assert np.allclose(signed_exp(features[:, :, 2:]), kept[:, :, 1:])
    assert np.abs(features[:, :, 2:]).max() < np.abs(kept[:, :, 1:]).max()

    # the mask is what decides, so a mask change cannot silently skip the transform
    assert feature_log_mask(HUMAN)[:, :2].sum() == 0
    assert feature_log_mask(HUMAN)[:, 2:].all()


def test_log_compresses_the_tail_that_dominated_the_loss():
    # one 84-sigma torque spike in an otherwise ordinary signal
    raw = np.concatenate([np.random.default_rng(1).normal(size=999), [84.0]])
    before = raw.std()
    after = signed_log(raw).std()
    assert after < before / 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
