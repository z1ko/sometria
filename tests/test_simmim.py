"""The mask has to be *inside* the encoder, and it has to keep its position.

SimMIM's only structural claim is that held-out tokens go through the encoder rather than
being dropped and rebuilt in a decoder. Get that wrong and nothing crashes: the shapes
still work, the loss still falls, and what you have trained is an MAE with an unusually
weak decoder. These tests pin the claim down.

The second one also guards `encode()`, which every probe calls. It reproduces the parent
encoder's output exactly, which it must: `MaskedInputEncoder` reorders when positional
embeddings are added -- after the mask substitution rather than before -- and that
reordering has to be a no-op whenever nothing is masked.
"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.models.baseline import Encoder
from sometria.models.simmim import MaskedInputEncoder, SimMIM

DIM, NUM_DOFS, FRAMES, PATCH, HEADS, DEPTH = 64, 4, 48, 8, 4, 2
NUM_TOKENS = (FRAMES // PATCH) * NUM_DOFS      # 6 time patches x 4 DOFs = 24
CHANNELS_IN = (0, 1, 2, 3)
CHANNELS_OUT = (0, 1, 2, 3, 4)
ENCODER_ARGS = (NUM_DOFS, PATCH, DIM, DEPTH, HEADS, CHANNELS_IN, 4, FRAMES // PATCH)


def _simmim() -> SimMIM:
    return SimMIM(
        num_dofs=NUM_DOFS, num_frames=FRAMES, num_frames_in_patch=PATCH, enc_depth=DEPTH,
        dim=DIM, num_heads=HEADS, channels_input=CHANNELS_IN, channels_output=CHANNELS_OUT,
    ).eval()


def _features(batch: int = 2) -> t.Tensor:
    return t.randn(batch, FRAMES, NUM_DOFS, 5, generator=t.Generator().manual_seed(0))


def test_the_encoder_reads_every_token_and_the_head_predicts_patch_values():
    """MAE's encoder sees the visible tokens; SimMIM's sees all of them. That is the model."""

    model = _simmim()
    with t.no_grad():
        encoded = model.encoder(_features()[..., list(CHANNELS_IN)], None, None,
                                t.arange(20).expand(2, -1))

    assert encoded.shape == (2, NUM_TOKENS, DIM)
    # Not `dim`: the head lands in patch space, which is what separates it from JEPA's.
    assert model.head.out_features == PATCH * len(CHANNELS_OUT)
    assert t.isfinite(model(_features()))


def test_masking_nothing_reproduces_the_plain_encoder():
    """`encode()` is the probe's entry point, so the reordered positional add must be inert."""

    plain, subclass = Encoder(*ENCODER_ARGS).eval(), MaskedInputEncoder(*ENCODER_ARGS).eval()
    subclass.load_state_dict(plain.state_dict(), strict=False)

    values = _features()[..., list(CHANNELS_IN)]
    with t.no_grad():
        assert t.allclose(plain(values, None, None), subclass(values, None, None), atol=1e-5)


def test_a_hidden_token_is_replaced_but_still_knows_where_it_is():
    """Substitute *before* position is added, never after.

    Doing it after would overwrite the positional embedding too, leaving every hidden token
    identical. The encoder could then not tell one masked position from another, and the
    reconstruction would have nothing to condition on. This test fails in exactly that case.
    """

    encoder = MaskedInputEncoder(*ENCODER_ARGS).eval()
    values = _features(1)[..., list(CHANNELS_IN)]
    hidden = t.tensor([[0, 1, 2]])

    with t.no_grad():
        free = encoder(values, None, None)
        masked = encoder(values, None, None, hidden)

    # The hidden positions moved...
    assert not t.allclose(free[0, :3], masked[0, :3], atol=1e-4)
    # ...but they did not all collapse onto one vector.
    assert not t.allclose(masked[0, 0], masked[0, 1], atol=1e-5)


def test_the_mask_token_is_trained():
    """It is a learned vector, so a gradient has to reach it or it stays at its init."""

    model = _simmim()
    model(_features()).backward()

    assert model.encoder.mask_token.grad is not None
    assert float(model.encoder.mask_token.grad.norm()) > 0
