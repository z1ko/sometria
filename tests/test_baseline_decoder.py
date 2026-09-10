"""The decoder-width knob: it must weaken the decoder without breaking anything that
predates it, and it must survive mixed precision.

`dec_dim` exists to test whether the decoder is strong enough to absorb the reconstruction
task on its own -- see `scripts/decoder_capacity.sh`. Both risks it carries are silent:
an old checkpoint that no longer loads, and a dtype mismatch that only appears under AMP.
"""

import sys
from pathlib import Path

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.models.baseline import MAE

DIM, ENC_DEPTH = 64, 2


def _mae(dec_dim=None, dec_depth=1):
    return MAE(
        num_dofs=4,
        num_frames=48,
        num_frames_in_patch=8,
        enc_depth=ENC_DEPTH,
        dec_depth=dec_depth,
        dim=DIM,
        dec_dim=dec_dim,
        num_heads=4,
        channels_input=(0, 1, 2, 3),
        channels_output=(0, 1, 2, 3, 4),
    )


def _features(batch=2):
    return t.randn(batch, 48, 4, 5, generator=t.Generator().manual_seed(0))


def test_the_default_adds_no_parameters_so_old_checkpoints_still_load():
    """`dec_dim=None` must be byte-for-byte the architecture that ran before it existed.

    The projection is an `nn.Identity` rather than a `nn.Linear(dim, dim)` for exactly
    this reason: Identity contributes no state_dict entries, so every checkpoint written
    before this argument existed still loads key for key instead of raising on an
    unexpected `decoder.embed.weight`.
    """

    keys = _mae(dec_dim=None).state_dict().keys()
    assert not [k for k in keys if k.startswith("decoder.embed")]
    # and the widths really are the encoder's
    model = _mae(dec_dim=None)
    assert model.decoder.mask_token.shape[-1] == DIM
    assert model.decoder.pos_s.shape[-1] == DIM


@pytest.mark.parametrize("dec_dim", [16, 32])
def test_a_narrow_decoder_narrows_every_decoder_parameter(dec_dim):
    model = _mae(dec_dim=dec_dim)
    assert model.decoder.embed.in_features == DIM
    assert model.decoder.embed.out_features == dec_dim
    assert model.decoder.mask_token.shape[-1] == dec_dim
    assert model.decoder.pos_s.shape[-1] == dec_dim
    assert model.decoder.pos_t.shape[-1] == dec_dim
    assert model.decoder.proj.in_features == dec_dim
    # the encoder is untouched -- this knob is about what the decoder can absorb
    assert model.encoder.pos_s.shape[-1] == DIM

    decoder = sum(p.numel() for p in model.decoder.parameters())
    wide = sum(p.numel() for p in _mae(dec_dim=None).decoder.parameters())
    assert decoder < wide


@pytest.mark.parametrize("dec_dim", [None, 16])
def test_the_forward_survives_mixed_precision(dec_dim):
    """The regression this file mostly exists for.

    A transformer block ends in a LayerNorm, which autocast keeps in fp32, so an Identity
    `embed` hands the decoder fp32 and it matches the fp32 mask token. A `Linear` embed is
    on autocast's bf16 list and hands back bf16, and the scatter that places encoded
    tokens into the mask canvas then dies with "Expected self.dtype to be equal to
    src.dtype". It only shows up under AMP, and training runs under AMP.
    """

    model = _mae(dec_dim=dec_dim)
    with t.autocast("cpu", dtype=t.bfloat16):
        loss = model(_features())
    loss.backward()

    assert t.isfinite(loss)
    # the point of the whole ablation: gradient has to reach the encoder
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
    )


def test_a_narrow_decoder_round_trips_through_a_checkpoint(tmp_path):
    model = _mae(dec_dim=16)
    path = tmp_path / "thin.ckpt"
    t.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    reloaded = MAE.load_from_checkpoint(path, map_location="cpu")

    assert reloaded.hparams.dec_dim == 16
    for a, b in zip(reloaded.state_dict().values(), model.state_dict().values()):
        assert t.equal(a, b)
