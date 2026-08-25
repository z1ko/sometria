"""Run with: python tests/test_mae.py"""

import sys
import tempfile
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.masking import MaskSpec
from sometria.models.mae import MaskedAutoencoder
from sometria.representation import channel_names

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4)
D = SPEC.num_dofs
C = SPEC.num_features
L = SPEC.grid_shape[0] * D


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


def _model(mask_ratio=0.90, **kwargs):
    mask = MaskSpec(mask_ratio=mask_ratio, tau=0.0, score_channels=())
    return MaskedAutoencoder(SPEC, mask, decoder_depth=1, **kwargs)


def test_the_prediction_is_a_whole_patch_of_every_channel():
    """MAE reconstructs the input, so the head is as wide as a token: patch x channels."""

    model = _model()
    x = _features()
    prediction, target, window = model(x, generator=t.Generator().manual_seed(0))

    assert model.prediction.out_features == SPEC.token_dim == SPEC.patch_size * C
    assert prediction.shape == target.shape == (2, window.mask.targets.shape[1], SPEC.token_dim)
    assert model.reconstruction_loss(prediction, target).isfinite()


def test_the_target_is_the_input_the_encoder_did_not_see():
    """Every target token is that patch of the input, verbatim -- no differencing."""

    model = _model()
    x = _features(batch=1)
    _, target, window = model(x, generator=t.Generator().manual_seed(0))

    for k, flat in enumerate(window.mask.targets[0].tolist()[:20]):
        patch, dof = divmod(flat, D)
        lo = patch * SPEC.patch_size
        assert t.allclose(target[0, k], x[0, lo : lo + SPEC.patch_size, dof].flatten(), atol=1e-6)


def test_the_mask_splits_the_grid_exactly_once():
    model = _model()
    _, _, window = model(_features(), generator=t.Generator().manual_seed(0))
    mask = window.mask

    assert mask.context.shape[1] == round(L * (1.0 - model.mask.mask_ratio))
    assert mask.context.shape[1] + mask.targets.shape[1] == L
    assert set(mask.context[0].tolist()) & set(mask.targets[0].tolist()) == set()


def test_target_normalization_standardizes_each_token_on_its_own():
    """norm_skes_loss: the loss asks for the shape of a patch, not its magnitude."""

    model = _model(norm_targets=True)
    prediction = t.zeros(1, 2, SPEC.patch_size)
    target = t.randn(1, 2, SPEC.patch_size) * 100.0 + 50.0

    # a constant prediction against a standardized target scores its mean square, which
    # is (n-1)/n because torch's var is unbiased -- the reference standardizes the same way
    expected = (SPEC.patch_size - 1) / SPEC.patch_size
    assert abs(model.reconstruction_loss(prediction, target).item() - expected) < 1e-3

    # and scaling one token by 1000 does not change the loss it contributes
    scaled = target.clone()
    scaled[:, 0] *= 1000.0
    assert abs(
        model.reconstruction_loss(prediction, scaled).item()
        - model.reconstruction_loss(prediction, target).item()
    ) < 1e-3

    off = _model(norm_targets=False)
    assert off.reconstruction_loss(prediction, scaled) > 1e4


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

    model = _model(mask_ratio=0.75)
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
        reloaded = MaskedAutoencoder.load_from_checkpoint(path, map_location="cpu")

    assert isinstance(reloaded.backbone, MotionTransformerEncoder)
    assert reloaded.backbone.spec == SPEC
    # plain fields, not the dataclass: Lightning refuses to log a frozen dataclass, and
    # torch.load's weights_only default refuses to unpickle one
    assert isinstance(reloaded.hparams.spec, dict)
    assert isinstance(reloaded.hparams.mask, dict)
    assert reloaded.mask == model.mask
    assert reloaded.hparams.mask["mask_ratio"] == 0.75
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert t.equal(a, b), name


def test_the_channel_losses_add_back_up_to_the_loss():
    """A breakdown, not a reweighting: every channel holds the same number of slots."""

    model = _model()
    x = _features()
    prediction, target, _ = model(x, generator=t.Generator().manual_seed(0))

    split = model.channel_losses(prediction, target)
    assert tuple(split) == tuple(f"mse/{name}" for name in channel_names())
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

def test_a_narrowed_loss_scores_those_channels_of_the_right_patch():
    """The channels are a stride inside a token, not a slice off its end."""

    model = _model(loss_channels=(0, 1))
    assert model.prediction.out_features == SPEC.patch_size * 2
    assert model.channel_names == ("sin", "cos")

    x = _features(batch=1)
    _, target, window = model(x, generator=t.Generator().manual_seed(0))
    assert target.shape[-1] == SPEC.patch_size * 2

    for k, flat in enumerate(window.mask.targets[0].tolist()[:20]):
        patch, dof = divmod(flat, D)
        lo = patch * SPEC.patch_size
        want = x[0, lo : lo + SPEC.patch_size, dof][:, [0, 1]].flatten()
        assert t.allclose(target[0, k], want, atol=1e-6), (patch, dof)


def test_an_out_of_range_or_empty_loss_channel_set_is_rejected():
    for bad in ((), (0, C)):
        try:
            _model(loss_channels=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for loss_channels={bad}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
