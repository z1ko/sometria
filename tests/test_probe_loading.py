"""A probe must load a checkpoint as whichever objective wrote it.

``scripts/probe_baseline_mae.load_pretrained`` resolves that class from the weights, and
every probe entry point now routes through it. It is the one branch in this change that
sits between a sweep and its results: get it wrong and nothing complains until a
``load_state_dict`` key error surfaces somewhere deep inside a matrix run, hours in.
"""

import sys
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_baseline_mae import BaselineBackbone, load_pretrained  # noqa: E402

from sometria.models.baseline import MAE  # noqa: E402
from sometria.models.jepa2 import JEPA  # noqa: E402
from sometria.models.simmim import SimMIM  # noqa: E402

SMALL = dict(
    num_dofs=3, num_frames=48, num_frames_in_patch=8, enc_depth=1,
    dim=32, num_heads=4, channels_input=(0, 1, 2, 3),
)
# SimMIM takes no `dec_depth` -- it has no decoder, only a linear head -- and rejects it
# rather than accepting it and leaving hyperparameters that claim a decoder depth which
# never existed. So the decoder size belongs to the two objectives that have one.
WITH_DECODER = SMALL | dict(dec_depth=1)


def _save(model, path: Path) -> Path:
    t.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    return path


def test_a_jepa_checkpoint_comes_back_as_jepa_reading_its_teacher(tmp_path):
    saved = _save(JEPA(**WITH_DECODER), tmp_path / "jepa.ckpt")
    loaded = load_pretrained(saved)

    assert isinstance(loaded, JEPA)

    x = t.randn(2, 48, 3, 5)
    with t.no_grad():
        # The paper probes the target encoder, so the adapter must reach the teacher and
        # not the student -- the two diverge the moment training starts.
        assert t.equal(
            BaselineBackbone(loaded).embed_tokens(x),
            loaded.encoder_teacher(x[..., list(SMALL["channels_input"])], None, None),
        )


def test_an_mae_checkpoint_still_comes_back_as_mae(tmp_path):
    saved = _save(MAE(**WITH_DECODER, channels_output=(0, 1, 2, 3)), tmp_path / "mae.ckpt")
    loaded = load_pretrained(saved)

    assert isinstance(loaded, MAE)
    assert BaselineBackbone(loaded).spec.d_model == SMALL["dim"]


def test_a_simmim_checkpoint_is_not_mistaken_for_an_mae_one(tmp_path):
    """The two collide on every prefix but one.

    Both carry `encoder.` and both once carried `decoder.`, so the discriminator keys on
    SimMIM's `head.` instead. Get it wrong and the checkpoint is rebuilt as the wrong class,
    which surfaces as a `load_state_dict` key error deep inside a sweep rather than here.
    """

    saved = _save(SimMIM(**SMALL, channels_output=(0, 1, 2, 3)), tmp_path / "simmim.ckpt")
    loaded = load_pretrained(saved)

    assert isinstance(loaded, SimMIM)
    assert not isinstance(loaded, MAE)
    assert BaselineBackbone(loaded).spec.d_model == SMALL["dim"]
