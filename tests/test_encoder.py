"""Run with: python tests/test_encoder.py"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.architecture.pos_embed import PositionalEncoding

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4, num_dofs=43, window_frames=240)
D = SPEC.num_dofs
C = SPEC.num_features


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


def test_positional_encoding_slices_to_the_window_it_is_asked_for():
    pos = PositionalEncoding(30, D, 32)
    assert pos.get_flat(30).shape == (1, 30 * D, 32)
    assert pos.get_flat(10).shape == (1, 10 * D, 32)
    # a short window takes the prefix, so patch 0 keeps meaning "start of window"
    assert t.equal(pos.get_flat(10), pos.get_flat(30)[:, : 10 * D])


def test_positional_encoding_refuses_a_window_longer_than_it_holds():
    pos = PositionalEncoding(30, D, 32)
    try:
        pos.get_flat(31)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for more time patches than exist")


def test_embed_tokens_returns_one_vector_per_grid_cell():
    encoder = MotionTransformerEncoder(SPEC)
    tokens = encoder.embed_tokens(_features())
    assert tokens.shape == (2, 30 * D, SPEC.d_model)
    assert encoder.grid_shape == (30, D)


def test_a_shorter_window_still_embeds():
    """The downstream path may use fewer frames than pretraining did."""

    encoder = MotionTransformerEncoder(SPEC)
    tokens = encoder.embed_tokens(_features(frames=80))
    assert tokens.shape == (2, 10 * D, SPEC.d_model)


def test_embed_tokens_with_an_index_encodes_only_that_subset():
    encoder = MotionTransformerEncoder(SPEC)
    x = _features()
    index = t.stack([t.randperm(30 * D)[:100] for _ in range(x.shape[0])])
    assert encoder.embed_tokens(x, index=index).shape == (2, 100, SPEC.d_model)


def test_poolings_have_the_widths_the_spec_advertises():
    encoder = MotionTransformerEncoder(SPEC)
    x = _features()
    assert encoder.embed(x, pool="window").shape == (2, SPEC.pooled_dim("window"))
    assert encoder.embed(x, pool="dof").shape == (2, SPEC.pooled_dim("dof"))
    assert SPEC.pooled_dim("dof") == D * SPEC.d_model


def test_dof_pooling_keeps_the_joint_axis_separate():
    """Slice ``d`` of the pooled vector is DOF ``d`` averaged over time, nothing else."""

    encoder = MotionTransformerEncoder(SPEC).eval()
    x = _features(batch=1)
    with t.no_grad():
        pooled = encoder.embed(x, pool="dof").reshape(D, SPEC.d_model)
        grid = encoder.embed_tokens(x).reshape(30, D, SPEC.d_model)

    assert t.allclose(pooled, grid.mean(dim=0), atol=1e-5)


def test_spec_rejects_a_window_that_does_not_tile():
    try:
        EncoderSpec(window_frames=241, patch_size=8)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a window that is not a whole number of patches")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
