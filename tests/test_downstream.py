"""Run with: python tests/test_downstream.py"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.dataset import _tiles
from sometria.downstream.classifier import MotionWindowClassifier
from sometria.downstream.labels import window_multi_hot
from sometria.downstream.metrics import WindowMeanAveragePrecision
from sometria.models.masked import MaskedMotionAutoencoder

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4)
D = SPEC.num_dofs
C = SPEC.num_features
LABELS = 12


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


def _segments(rows):
    return t.tensor(rows, dtype=t.float32).reshape(-1, 3)


# --- label coverage -------------------------------------------------------------------

def test_a_segment_below_the_threshold_is_not_a_label():
    # 0.5 s of a 4 s window is 12.5%, under the 15% floor
    y = window_multi_hot(_segments([[0.0, 0.5, 3]]), 0.0, 4.0, LABELS)
    assert y.sum() == 0

    y = window_multi_hot(_segments([[0.0, 0.7, 3]]), 0.0, 4.0, LABELS)
    assert y[3] == 1.0 and y.sum() == 1


def test_a_window_spanning_several_segments_is_multi_hot():
    y = window_multi_hot(_segments([[0.0, 2.0, 1], [2.0, 4.0, 5]]), 0.0, 4.0, LABELS)
    assert y[1] == 1.0 and y[5] == 1.0 and y.sum() == 2


def test_coverage_is_measured_against_the_window_not_the_segment():
    """A long segment only partly overlapping the crop counts what it actually covers."""

    # a segment running far past the crop still only covers the 1 s it overlaps
    assert window_multi_hot(_segments([[3.0, 100.0, 2]]), 0.0, 4.0, LABELS)[2] == 1.0
    assert window_multi_hot(_segments([[3.9, 100.0, 2]]), 0.0, 4.0, LABELS).sum() == 0


def test_split_occurrences_of_one_label_add_up():
    """Two short bouts of the same action reach the threshold together."""

    segments = _segments([[0.0, 0.3, 7], [2.0, 2.4, 7]])
    assert window_multi_hot(segments, 0.0, 4.0, LABELS)[7] == 1.0


def test_a_window_with_no_scoreable_segment_is_all_negative():
    """`transition` has no vocabulary index at all; those windows still train."""

    y = window_multi_hot(t.zeros(0, 3), 0.0, 4.0, LABELS)
    assert y.shape == (LABELS,) and y.sum() == 0


# --- mAP ------------------------------------------------------------------------------

def test_map_ignores_labels_that_never_occur():
    metric = WindowMeanAveragePrecision(num_labels=LABELS)
    target = t.zeros(4, LABELS, dtype=t.int)
    target[[0, 2], 1] = 1
    logits = t.zeros(4, LABELS)
    logits[[0, 2], 1] = 9.0

    metric.update(logits, target)
    # only label 1 occurs, and it is ranked perfectly: 1.0, not 1/12
    assert abs(metric.compute().item() - 1.0) < 1e-6


# --- windows ---------------------------------------------------------------------------

def test_tiles_cover_the_whole_motion_including_its_tail():
    starts = _tiles(500, 240)
    assert starts == [0, 240, 260]                           # last one flush with the end
    assert starts[-1] + 240 == 500

    assert _tiles(480, 240) == [0, 240]                       # exact fit, no extra tile
    assert _tiles(239, 240) == []                             # too short to score at all


def test_both_learning_rates_decay():
    """One eta_min for every group would raise a small backbone rate instead of decaying it."""

    head = t.nn.Linear(2, 2)
    backbone = t.nn.Linear(2, 2)
    optimizer = t.optim.SGD(
        [{"params": head.parameters(), "lr": 1e-3},
         {"params": backbone.parameters(), "lr": 1e-5}]
    )
    schedule = lr_schedule(optimizer, warmup_steps=10, total_steps=100)

    peak = None
    for step in range(100):
        optimizer.step()
        schedule.step()
        if step == 9:
            peak = [g["lr"] for g in optimizer.param_groups]

    end = [g["lr"] for g in optimizer.param_groups]
    assert peak == [1e-3, 1e-5]
    assert all(e < p for e, p in zip(end, peak))


# --- protocol -------------------------------------------------------------------------

def test_the_two_poolings_give_the_head_the_widths_the_protocol_asks_for():
    probe = MotionWindowClassifier(SPEC, num_labels=LABELS, pool="window", head="linear")
    finetune = MotionWindowClassifier(SPEC, num_labels=LABELS, pool="dof", head="mlp",
                                      freeze_backbone=False)

    x = _features()
    assert probe(x).shape == (2, LABELS)
    assert finetune(x).shape == (2, LABELS)
    assert probe.head[1].in_features == SPEC.d_model
    assert finetune.head[1].in_features == D * SPEC.d_model


def test_a_frozen_backbone_stays_frozen_and_in_eval():
    model = MotionWindowClassifier(SPEC, num_labels=LABELS, freeze_backbone=True)
    model.train()

    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())

    # freezing is also exclusion from the optimizer, or weight decay keeps moving weights
    # that are supposed to be fixed
    model._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = model.configure_optimizers()["optimizer"]
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    assert not any(id(p) in backbone_ids for g in optimizer.param_groups for p in g["params"])


def test_an_unfrozen_backbone_receives_gradient():
    model = MotionWindowClassifier(SPEC, num_labels=LABELS, freeze_backbone=False, head="mlp",
                                   pool="dof")
    loss = model.loss(model(_features()), t.zeros(2, LABELS))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


def test_a_probe_leaves_the_backbone_untouched():
    model = MotionWindowClassifier(SPEC, num_labels=LABELS, freeze_backbone=True)
    loss = model.loss(model(_features()), t.zeros(2, LABELS))
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())


def test_a_pretrained_backbone_arrives_with_its_weights():
    objective = MaskedMotionAutoencoder(SPEC, decoder_depth=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pretrain.ckpt"
        t.save(
            {
                "state_dict": objective.state_dict(),
                "hyper_parameters": dict(objective.hparams),
                "pytorch-lightning_version": "2.0.0",
                "loops": {},
            },
            path,
        )
        model = MotionWindowClassifier.from_pretrained(str(path), num_labels=LABELS)

    assert model.backbone.spec == SPEC
    assert t.equal(
        model.backbone.projection.weight, objective.backbone.projection.weight
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
