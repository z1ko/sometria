"""A corpus config may carry `model:` keys that the chosen objective does not take.

`config/dataloader/amass_smpl.yaml` sets `num_dofs` and both channel lists, because on that
corpus they are facts about the data: 21 body joints and 18 rot6d channels rather than 43
anatomical DOFs and 5. Every objective needs `channels_input`; only the reconstructive ones
have `channels_output`. Composing that corpus with `config/pretrain_jepa.yaml` is a
legitimate thing to want and used to die with a TypeError naming an argument nobody typed.

The filter has to stay narrow. Swallowing every unrecognised key would turn a misspelled
config line into a silently wrong run, which is worse than the crash it replaces.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("roottrain", ROOT / "train.py")
train = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(train)  # type: ignore[union-attr]

SMPL = {"num_dofs": 21, "channels_input": [0, 1, 2, 3, 4, 5], "channels_output": [0, 1, 2, 3, 4, 5]}


def test_jepa_drops_the_output_channels_a_corpus_declared_for_everyone_else():
    kwargs = train.model_kwargs("jepa", dict(SMPL))

    assert "channels_output" not in kwargs
    assert kwargs["channels_input"] == [0, 1, 2, 3, 4, 5]
    assert kwargs["num_dofs"] == 21
    # And it builds, which is the point of the whole exercise.
    assert train.MODELS["jepa"](**kwargs).hparams.num_dofs == 21


@pytest.mark.parametrize("name", ["mae", "simmim"])
def test_a_reconstructive_objective_keeps_them(name):
    """Nothing is dropped from an objective that actually has the argument."""

    assert train.model_kwargs(name, dict(SMPL)) == SMPL


def test_a_misspelled_key_is_still_an_error():
    """`channles_input` belongs to no objective, so it must not be quietly ignored."""

    with pytest.raises(SystemExit, match="channles_input"):
        train.model_kwargs("jepa", {"channles_input": [0, 1]})


def test_a_key_is_only_dropped_because_a_sibling_objective_has_it():
    """The rule is "another model takes it", not "this model does not"."""

    known = {p for m in train.MODELS.values() for p in __import__("inspect").signature(m).parameters}
    assert "channels_output" in known           # MAE and SimMIM
    assert "dec_depth" in known                 # MAE and JEPA, not SimMIM
    assert train.model_kwargs("simmim", {"dec_depth": 5}) == {}
