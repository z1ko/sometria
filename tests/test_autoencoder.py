"""Run with: python tests/test_autoencoder.py"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.models.autoencoder import MotionConvAutoencoder, MotionMaskedAutoencoder


def _features(batch=2, frames=32, dofs=43, channels=5, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, dofs, channels, generator=g)


def test_conv_autoencoder_reconstructs_the_input_shape():
    x = _features()
    model = MotionConvAutoencoder(hidden_channels=32, depth=1)
    y = model(x)
    assert y.shape == x.shape


def test_random_masked_autoencoder_predicts_target_patches():
    x = _features()
    valid = t.ones(x.shape[:2], dtype=t.bool)
    model = MotionMaskedAutoencoder(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        mask_strategy="random",
    )

    prediction, target, target_valid, mask = model(
        x, valid, generator=t.Generator().manual_seed(0)
    )

    assert prediction.shape == target.shape
    assert prediction.shape == (x.shape[0], mask.targets.shape[1], 8 * x.shape[-1])
    assert target_valid.shape == mask.targets.shape
    assert model.reconstruction_loss(prediction, target, target_valid).isfinite()


def test_mamp_masked_autoencoder_predicts_target_patches():
    x = _features()
    valid = t.ones(x.shape[:2], dtype=t.bool)
    model = MotionMaskedAutoencoder(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        mask_strategy="mamp",
        tau=0.25,
    )

    prediction, target, target_valid, mask = model(
        x, valid, generator=t.Generator().manual_seed(0)
    )

    assert prediction.shape == target.shape
    assert prediction.shape == (x.shape[0], mask.targets.shape[1], 8 * x.shape[-1])
    assert target_valid.shape == mask.targets.shape
    assert model.reconstruction_loss(prediction, target, target_valid).isfinite()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
