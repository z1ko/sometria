"""Run with: python tests/test_window.py

Not one nn.Module is built in this file. A masked window is index arithmetic over a
feature tensor, and the point of giving it a module of its own is that it can be
computed, checked and plotted without a backbone.
"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec
from sometria.masking import MaskSpec
from sometria.models.window import mask_window, masked_token_mse, standardize_tokens

SPEC = EncoderSpec(num_dofs=6, num_features=5, patch_size=8, window_frames=80, d_model=32, num_heads=4)
TIME_PATCHES, DOFS = SPEC.grid_shape          # 10, 6
TOKENS = TIME_PATCHES * DOFS                  # 60
MASK = MaskSpec(mask_ratio=0.5, tau=0.80, score_channels=(2,))


def _features(batch=2, frames=80, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, DOFS, SPEC.num_features, generator=g)


def _window(features=None, mask=MASK, **kwargs):
    return mask_window(
        features if features is not None else _features(),
        SPEC,
        mask,
        generator=t.Generator().manual_seed(0),
        **kwargs,
    )


def test_context_and_targets_partition_the_token_grid():
    w = _window()

    assert w.num_time_patches == TIME_PATCHES
    assert w.mask.context.shape == (2, TOKENS // 2)
    assert w.mask.targets.shape == (2, TOKENS // 2)
    for b in range(2):
        both = t.cat([w.mask.context[b], w.mask.targets[b]]).sort().values
        assert t.equal(both, t.arange(TOKENS))


def test_a_shorter_window_gets_fewer_time_patches():
    w = _window(_features(frames=40))

    assert w.num_time_patches == 5
    assert w.patches.shape[1] == 5 * DOFS


def test_values_is_the_flattened_view_of_patches():
    w = _window()

    assert w.patches.shape == (2, TOKENS, SPEC.patch_size, SPEC.num_features)
    assert w.values.shape == (2, TOKENS, SPEC.patch_size * SPEC.num_features)
    assert t.equal(w.values, w.patches.flatten(start_dim=-2))


def test_uniform_masking_needs_no_score_channel():
    """tau <= 0 is the ablation baseline, not a second code path."""

    w = _window(mask=MaskSpec(mask_ratio=0.5, tau=0.0, score_channels=()))

    assert w.mask.context.shape[1] == TOKENS // 2


def test_target_valid_is_all_true_when_no_frame_mask_is_given():
    w = _window()

    assert w.target_valid.shape == w.mask.targets.shape
    assert w.target_valid.dtype == t.bool
    assert w.target_valid.all()


def test_padding_is_forced_into_the_targets_and_marked_invalid():
    """A padded window must never spend its context budget on frames that are not there."""

    valid = t.ones(2, 80, dtype=t.bool)
    valid[:, 56:] = False                     # 7 of 10 time patches real -> 42 real tokens
    real = 7 * DOFS
    w = _window(valid=valid)

    assert w.target_valid.shape == w.mask.targets.shape
    # the context budget is 30 tokens and 42 are real, so no padding needs to be kept
    assert (w.mask.context < real).all()
    # every padded token landed in the targets, and every one of them is marked invalid
    assert w.target_valid.sum().item() == 2 * (TOKENS // 2 - (TOKENS - real))


def test_a_mis_shaped_window_is_named_rather_than_failing_in_a_matmul():
    wrong_dofs = t.randn(2, 80, DOFS + 1, SPEC.num_features)
    try:
        _window(wrong_dofs)
    except ValueError as error:
        assert "DOFs" in str(error)
    else:
        raise AssertionError("expected a ValueError naming the DOF count")


def test_masked_token_mse_scores_valid_targets_only():
    prediction = t.zeros(1, 4, 3)
    target = t.zeros(1, 4, 3)
    target[0, 0] = 2.0                        # squared error 4 per value, valid
    target[0, 3] = 100.0                      # invalid: must not reach the loss
    valid = t.tensor([[True, True, True, False]])

    loss = masked_token_mse(prediction, target, valid)

    assert t.isclose(loss, t.tensor(4.0 / 3.0))


def test_masked_token_mse_survives_a_window_with_no_valid_target():
    prediction = t.zeros(1, 2, 3, requires_grad=True)
    loss = masked_token_mse(prediction, t.ones(1, 2, 3), t.zeros(1, 2, dtype=t.bool))

    assert loss.isfinite() and loss.item() == 0.0
    loss.backward()                           # a zeroed loss still has to be differentiable


def test_masked_token_mse_survives_an_empty_target_set():
    """round(L * (1 - mask_ratio)) == L is a legal MaskSpec, and leaves nothing to score."""

    w = _window(mask=MaskSpec(mask_ratio=0.005, tau=0.0, score_channels=()))
    assert w.mask.targets.shape[1] == 0

    prediction = t.zeros(2, 0, 3, requires_grad=True)
    loss = masked_token_mse(prediction, t.zeros(2, 0, 3), w.target_valid)

    assert loss.isfinite() and loss.item() == 0.0
    loss.backward()


def test_standardize_tokens_normalizes_each_token_over_its_own_values():
    """MAMP's norm_skes_loss: the loss asks for the shape of a patch, not its magnitude."""

    target = t.randn(2, 5, 16) * t.tensor([1.0, 10.0, 100.0, 0.1, 1.0]).reshape(1, 5, 1)
    normalized = standardize_tokens(target)

    assert t.allclose(normalized.mean(dim=-1), t.zeros(2, 5), atol=1e-5)
    assert t.allclose(normalized.var(dim=-1), t.ones(2, 5), atol=1e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
