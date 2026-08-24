"""Run with: python tests/test_masked.py"""

import sys
import tempfile
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.models.masked import MaskedMotionAutoencoder

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4)
D = SPEC.num_dofs
C = SPEC.num_features
L = SPEC.grid_shape[0] * D


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


def _model(**kwargs):
    return MaskedMotionAutoencoder(SPEC, decoder_depth=1, **kwargs)


def test_mae_configuration_predicts_its_target_patches():
    """tau <= 0: uniform masking, the MAE baseline."""

    model = _model(tau=0.0)
    x = _features()
    prediction, target, target_valid, mask = model(x, generator=t.Generator().manual_seed(0))

    assert prediction.shape == target.shape == (2, mask.targets.shape[1], SPEC.token_dim)
    assert target_valid.shape == mask.targets.shape
    assert model.reconstruction_loss(prediction, target, target_valid).isfinite()


def test_mamp_configuration_predicts_its_target_patches():
    """tau > 0: motion-aware masking, loss on the velocity channel."""

    model = _model(tau=0.25, loss_channels=(2,))
    x = _features()
    prediction, target, target_valid, mask = model(x, generator=t.Generator().manual_seed(0))

    assert prediction.shape == target.shape == (2, mask.targets.shape[1], SPEC.token_dim)
    assert model.reconstruction_loss(prediction, target, target_valid).isfinite()


def test_the_mask_splits_the_grid_exactly_once():
    model = _model()
    _, _, _, mask = model(_features(), generator=t.Generator().manual_seed(0))

    assert mask.context.shape[1] == round(L * (1.0 - model.mask_ratio))
    assert mask.context.shape[1] + mask.targets.shape[1] == L
    assert set(mask.context[0].tolist()) & set(mask.targets[0].tolist()) == set()


def test_the_encoder_never_sees_a_target_token():
    """Encoding must depend on context values only, or the objective is trivial."""

    model = _model().eval()
    x = _features(batch=1)
    g = t.Generator().manual_seed(0)
    with t.no_grad():
        _, _, _, mask = model(x, generator=t.Generator().manual_seed(0))
        before = model.backbone.embed_tokens(x, index=mask.context)

        scrambled = x.clone()
        scrambled[:] = t.randn(x.shape, generator=g)
        # put the context frames back; only target-only patches differ now
        keep = t.zeros(L, dtype=t.bool)
        keep[mask.context[0]] = True
        grid = keep.reshape(-1, D)
        for patch in range(grid.shape[0]):
            for dof in range(D):
                if grid[patch, dof]:
                    lo = patch * SPEC.patch_size
                    scrambled[0, lo : lo + SPEC.patch_size, dof] = x[0, lo : lo + SPEC.patch_size, dof]

        after = model.backbone.embed_tokens(scrambled, index=mask.context)

    assert t.allclose(before, after, atol=1e-5)


def test_loss_scores_only_the_channels_it_was_given():
    model = _model(tau=0.0, loss_channels=(2,))
    prediction = t.zeros(1, 4, SPEC.token_dim)
    target = t.zeros(1, 4, SPEC.patch_size, C)
    target[..., 3] = 100.0                                  # acc: not a loss channel
    valid = t.ones(1, 4, dtype=t.bool)

    assert model.reconstruction_loss(prediction, target.flatten(-2), valid) == 0.0

    target[..., 2] = 2.0                                    # vel: is one
    loss = model.reconstruction_loss(prediction, target.flatten(-2), valid)
    assert abs(loss.item() - 4.0) < 1e-6


def test_invalid_target_tokens_are_left_out_of_the_loss():
    model = _model(tau=0.0)
    x = _features(batch=1)
    valid = t.ones(1, 240, dtype=t.bool)
    valid[0, 120:] = False

    prediction, target, target_valid, mask = model(x, valid, generator=t.Generator().manual_seed(0))

    # padding is forced into targets, and then out of the loss
    assert not target_valid.all()
    assert model.reconstruction_loss(prediction, target, target_valid).isfinite()


def test_a_checkpoint_reloads_without_being_told_the_architecture():
    """The gotcha this design exists for: the spec travels, the module does not."""

    model = _model(tau=0.25)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "objective.ckpt"
        t.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.0.0",
                "loops": {},
            },
            path,
        )
        reloaded = MaskedMotionAutoencoder.load_from_checkpoint(path, map_location="cpu")

    assert isinstance(reloaded.backbone, MotionTransformerEncoder)
    assert reloaded.backbone.spec == SPEC
    # plain fields, not the dataclass: Lightning refuses to log a frozen dataclass, and
    # torch.load's weights_only default refuses to unpickle one
    assert isinstance(reloaded.hparams.backbone, dict)
    assert reloaded.hparams.tau == 0.25
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert t.equal(a, b), name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
