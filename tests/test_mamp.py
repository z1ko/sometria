"""Run with: python tests/test_mamp.py"""

import sys
import tempfile
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.masking import MaskSpec, extract_motion
from sometria.models.mamp import MaskedMotionPredictor
from sometria.representation import channel_names

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4)
D = SPEC.num_dofs
C = SPEC.num_features
L = SPEC.grid_shape[0] * D


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


def _model(mask_ratio=0.90, tau=0.80, score_channels=(2,), **kwargs):
    mask = MaskSpec(mask_ratio=mask_ratio, tau=tau, score_channels=score_channels)
    return MaskedMotionPredictor(SPEC, mask, decoder_depth=1, **kwargs)


def _width(model):
    return SPEC.patch_size * len(model.loss_channels)


def test_the_prediction_is_as_wide_as_the_channels_the_loss_scores():
    model = _model()
    x = _features()
    prediction, target, window = model(x, generator=t.Generator().manual_seed(0))

    assert model.prediction.out_features == _width(model)
    assert prediction.shape == target.shape == (2, window.mask.targets.shape[1], _width(model))
    assert model.reconstruction_loss(prediction, target).isfinite()


def test_the_motion_target_is_the_difference_of_the_input_not_a_channel():
    """MAMP's extract_motion, taken over the window so patch boundaries are real."""

    model = _model(motion_stride=1, loss_channels=(0, 1))
    x = _features(batch=1)
    _, target, window = model(x, generator=t.Generator().manual_seed(0))

    expected = extract_motion(x, 1)[..., [0, 1]]
    for k, flat in enumerate(window.mask.targets[0].tolist()[:20]):
        patch, dof = divmod(flat, D)
        lo = patch * SPEC.patch_size
        want = expected[0, lo : lo + SPEC.patch_size, dof].flatten()
        assert t.allclose(target[0, k], want, atol=1e-6), (patch, dof)

    # the last frame of the window has no successor and is left at zero, as in the reference
    assert (extract_motion(x, 1)[0, -1] == 0).all()


def test_loss_scores_only_the_channels_it_was_given():
    """The head is as wide as the target, so an unscored channel never reaches the loss."""

    model = _model(loss_channels=(2,), norm_targets=False)
    assert model.prediction.out_features == SPEC.patch_size

    x = _features(batch=1)
    _, target, window = model(x, generator=t.Generator().manual_seed(0))
    patch, dof = divmod(window.mask.targets[0, 0].item(), D)
    lo = patch * SPEC.patch_size
    want = extract_motion(x, 1)[0, lo : lo + SPEC.patch_size, dof, 2]
    assert t.allclose(target[0, 0], want, atol=1e-6)

    prediction = t.zeros(1, 4, SPEC.patch_size)
    flat = t.full((1, 4, SPEC.patch_size), 2.0)
    assert abs(model.reconstruction_loss(prediction, flat).item() - 4.0) < 1e-6


def test_an_empty_loss_channel_set_is_rejected():
    try:
        _model(loss_channels=())
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a loss with no channel to score")


def test_the_mask_splits_the_grid_exactly_once():
    model = _model()
    _, _, window = model(_features(), generator=t.Generator().manual_seed(0))
    mask = window.mask

    assert mask.context.shape[1] == round(L * (1.0 - model.mask.mask_ratio))
    assert mask.context.shape[1] + mask.targets.shape[1] == L
    assert set(mask.context[0].tolist()) & set(mask.targets[0].tolist()) == set()


def test_the_encoder_never_sees_a_target_token():
    """Encoding must depend on context values only, or the objective is trivial."""

    model = _model().eval()
    x = _features(batch=1)
    g = t.Generator().manual_seed(0)
    with t.no_grad():
        _, _, window = model(x, generator=t.Generator().manual_seed(0))
        mask = window.mask
        before = model.backbone.embed_tokens(x, index=mask.context)

        scrambled = t.randn(x.shape, generator=g)
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
        reloaded = MaskedMotionPredictor.load_from_checkpoint(path, map_location="cpu")

    assert isinstance(reloaded.backbone, MotionTransformerEncoder)
    assert reloaded.backbone.spec == SPEC
    # plain fields, not the dataclass: Lightning refuses to log a frozen dataclass, and
    # torch.load's weights_only default refuses to unpickle one
    assert isinstance(reloaded.hparams.spec, dict)
    assert isinstance(reloaded.hparams.mask, dict)
    assert reloaded.mask == model.mask
    assert reloaded.hparams.mask["tau"] == 0.25
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert t.equal(a, b), name


def test_the_channel_losses_add_back_up_to_the_loss():
    """A breakdown, not a reweighting: every channel holds the same number of slots."""

    model = _model()
    x = _features()
    prediction, target, _ = model(x, generator=t.Generator().manual_seed(0))

    split = model.channel_losses(prediction, target)
    assert tuple(split) == tuple(f"mse/{name}" for name in channel_names(model.loss_channels))
    assert abs(t.stack(list(split.values())).mean().item()
               - model.reconstruction_loss(prediction, target).item()) < 1e-5


def test_a_channel_loss_is_that_channel_and_no_other():
    """Channel c is every c-th slot of a token, because a token flattens (patch, C)."""

    model = _model(norm_targets=False)
    width = len(model.channel_names)
    target = t.zeros(1, 4, SPEC.patch_size * width)
    prediction = t.zeros_like(target)
    prediction[..., 1::width] = 2.0     # the whole error sits in channel 1

    split = list(model.channel_losses(prediction, target).values())
    assert split[1].item() == 4.0
    assert all(v.item() == 0.0 for i, v in enumerate(split) if i != 1)

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
