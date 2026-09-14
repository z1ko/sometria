"""Prediction and target must line up, token for token, on the same masked positions.

JEPA's two halves come from different modules reading different inputs -- the student from
the context tokens, the teacher from the whole grid -- and are rejoined only by a shared
index set. Knock that index set out of step with itself and nothing complains: the shapes
still match, the loss is still finite, and the model trains happily against targets
belonging to other tokens. These are the checks that fail when that happens.

The second test doubles as the guard on the 7.2-point decision from S-JEPA's Table 6, that
the teacher reads the *whole* grid and the mask is applied to its output. It reconstructs
the target from a full-grid teacher pass, so masking the teacher's input instead would
fail it.
"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.models.baseline import gather_tokens, random_mask
from sometria.models.jepa2 import JEPA, standardize

DIM, NUM_DOFS, FRAMES, PATCH = 64, 4, 48, 8
NUM_TOKENS = (FRAMES // PATCH) * NUM_DOFS      # 6 time patches x 4 DOFs = 24
MASK_RATIO = 0.90
CHANNELS = (0, 1, 2, 3)


def _jepa() -> JEPA:
    return JEPA(
        num_dofs=NUM_DOFS,
        num_frames=FRAMES,
        num_frames_in_patch=PATCH,
        enc_depth=2,
        dec_depth=1,
        dim=DIM,
        num_heads=4,
        mask_ratio=MASK_RATIO,
        channels_input=CHANNELS,
    ).eval()


def _features(batch: int = 2) -> t.Tensor:
    return t.randn(batch, FRAMES, NUM_DOFS, 5, generator=t.Generator().manual_seed(0))


def test_prediction_and_target_cover_exactly_the_masked_tokens():
    """Both halves are (B, N_masked, dim) -- embeddings, not patch values."""

    model = _jepa()
    with t.no_grad():
        prediction, target = model(_features())

    n_masked = NUM_TOKENS - int(NUM_TOKENS * (1.0 - MASK_RATIO))
    assert prediction.shape == (2, n_masked, DIM)
    assert target.shape == (2, n_masked, DIM)


def test_target_is_the_whole_grid_teacher_read_at_the_masked_indices():
    """The teacher sees every token; the mask selects from its *output*, standardized."""

    model, x = _jepa(), _features()

    t.manual_seed(0)
    with t.no_grad():
        _, target = model(x)

    # Same seed reproduces the same draw, so this is the index set `forward` used.
    t.manual_seed(0)
    _, mask_idx = random_mask(model.num_tokens, model.mask_ratio, x.size(0), x.device)
    with t.no_grad():
        whole_grid = model.encoder_teacher(x[..., list(CHANNELS)], None, None)
    expected = standardize(gather_tokens(whole_grid, mask_idx))

    assert t.allclose(target, expected, atol=1e-6)


def test_prediction_is_the_predictor_read_at_those_same_indices():
    """The student's half is gathered at the *same* positions the target came from."""

    model, x = _jepa(), _features()

    t.manual_seed(0)
    with t.no_grad():
        prediction, _ = model(x)

    t.manual_seed(0)
    keep_idx, mask_idx = random_mask(model.num_tokens, model.mask_ratio, x.size(0), x.device)
    with t.no_grad():
        context = model.encoder_student(x[..., list(CHANNELS)], keep_idx, None)
        expected = gather_tokens(model.predictor(context, keep_idx), mask_idx)

    assert t.allclose(prediction, expected, atol=1e-6)


def test_collapse_metrics_are_absent_rather_than_infinite_for_a_single_window():
    """A one-window batch cannot measure spread across the batch, so it must not pretend to.

    ``loss_over_null`` divides by a variance taken across the batch axis. For one window
    that variance is exactly zero, the clamp turns the ratio into something of order 1e8,
    and because a validation loader keeps its short trailing batch, a single such batch
    would dominate the epoch mean of the metric ``config/pretrain_jepa.yaml`` selects
    checkpoints by.
    """

    model = _jepa()
    with t.no_grad():
        loss, metrics = model.step(_features(batch=1))

    assert t.isfinite(loss)
    assert metrics == {}

    with t.no_grad():
        _, metrics = model.step(_features(batch=2))

    assert set(metrics) == {"embed_std", "loss_over_null"}
    assert all(t.isfinite(v) for v in metrics.values())


def test_the_ema_endpoints_must_be_ordered():
    """A reversed pair ramps the teacher down, which trains happily and learns nothing."""

    import pytest

    with pytest.raises(ValueError, match="ema_start <= ema_stop"):
        JEPA(num_dofs=NUM_DOFS, num_frames=FRAMES, num_frames_in_patch=PATCH, enc_depth=1,
             dec_depth=1, dim=DIM, num_heads=4, ema_start=1.0, ema_stop=0.996)
