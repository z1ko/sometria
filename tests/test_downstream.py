"""Run with: python tests/test_downstream.py"""

import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder, load_encoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.dataset import LabelledWindows, _tiles
from sometria.downstream.classifier import MotionLinearClassifier
from sometria.downstream.finetune import MotionFinetuneClassifier
from sometria.downstream.labels import window_multi_hot
from sometria.downstream.metrics import MultilabelTopKRecall, WindowMeanAveragePrecision
from sometria.models.mamp import MaskedMotionPredictor

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


class _FakeMotions:
    """A MotionDataset that loads nothing: n samples of one flat window each."""

    def __init__(self, n: int) -> None:
        self.samples = pl.DataFrame({"n_frames": [240] * n})

    def __getitem__(self, index: int) -> dict:
        return {
            "features": t.zeros(240, D, C),
            "time": t.arange(240) / 60.0,
            "hz": 60.0,
            "sample_id": f"s{index}",
        }


def test_a_windows_dataset_carries_no_tensor_into_a_worker():
    """torch's pickler moves every tensor into shared memory, and forkserver refuses past
    256 file descriptors -- `ValueError: too many fds` at 4,600 labelled samples, before
    the first batch. Segments stay numpy; the window still gets a tensor target.
    """

    n = 300
    segments = {f"s{i}": np.array([[0.0, 4.0, 1]], dtype=np.float32) for i in range(n)}
    dataset = LabelledWindows(_FakeMotions(n), segments, num_labels=LABELS, window_frames=240)

    assert not any(isinstance(v, t.Tensor) for v in dataset.segments.values())
    assert pickle.loads(pickle.dumps(dataset.segments))["s7"].shape == (1, 3)
    assert dataset[0]["labels"][1] == 1.0
    assert dataset[0]["features"].shape == (240, D, C)


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

def _probe(pool="mean", **kwargs):
    return MotionLinearClassifier(
        MotionTransformerEncoder(SPEC), num_labels=LABELS, pool=pool, **kwargs
    )


def test_every_pooling_gives_the_head_the_width_it_advertises():
    widths = {"mean": SPEC.d_model, "mean_max": 2 * SPEC.d_model,
              "attentive": SPEC.d_model, "attentive_factorized": SPEC.d_model}

    x = _features()
    for pool, width in widths.items():
        model = _probe(pool)
        assert model(x).shape == (2, LABELS), pool
        assert model.head[1].in_features == width, pool


def test_a_pooler_reads_the_grid_not_a_flat_sequence():
    """Factorized attention has to split T x D the way the encoder laid the tokens out."""

    model = _probe("attentive_factorized").eval()
    x = _features(batch=1)
    with t.no_grad():
        tokens = model.backbone.embed_tokens(x)
        # uniform scores collapse both softmaxes to a mean, which is the mean pooler
        for net in (model.pooler.s_score_net, model.pooler.t_score_net):
            t.nn.init.zeros_(net[0].weight); t.nn.init.zeros_(net[0].bias)
            t.nn.init.zeros_(net[2].weight)
        assert t.allclose(model.pooler(tokens), tokens.mean(dim=1), atol=1e-5)


def test_a_shorter_window_still_pools():
    """T comes from the token count; the pooler must not bake in the full window."""

    model = _probe("attentive_factorized")
    assert model(_features(frames=120)).shape == (2, LABELS)


def test_the_backbone_stays_frozen_and_in_eval():
    model = _probe()
    model.train()

    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())

    # freezing is also exclusion from the optimizer, or weight decay keeps moving weights
    # that are supposed to be fixed
    model._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = model.configure_optimizers()["optimizer"]
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    assert not any(id(p) in backbone_ids for g in optimizer.param_groups for p in g["params"])


def test_the_pooler_is_trained_along_with_the_head():
    """An attentive pooler left out of the optimizer is a random projection forever."""

    model = _probe("attentive_factorized")
    model._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = model.configure_optimizers()["optimizer"]

    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert optimized == {id(p) for p in [*model.pooler.parameters(), *model.head.parameters()]}

    model.loss(model(_features()), t.zeros(2, LABELS)).backward()
    assert all(p.grad is not None for p in model.pooler.parameters())


def test_a_probe_leaves_the_backbone_untouched():
    model = _probe()
    model.loss(model(_features()), t.zeros(2, LABELS)).backward()
    assert all(p.grad is None for p in model.backbone.parameters())


# --- finetune -------------------------------------------------------------------------

def _finetune(pool="mean", **kwargs):
    return MotionFinetuneClassifier(
        MotionTransformerEncoder(SPEC), num_labels=LABELS, pool=pool, **kwargs
    )


def test_a_finetune_trains_the_backbone():
    """Every way the probe holds the backbone still has to be undone here."""

    model = _finetune()
    model.train()

    assert model.backbone.training
    assert all(p.requires_grad for p in model.backbone.parameters())

    model.loss(model(_features()), t.zeros(2, LABELS)).backward()
    assert all(p.grad is not None for p in model.backbone.parameters())


def test_a_finetune_can_regularize_a_pretrained_backbone():
    """The config's `encoder:` block is dead for a loaded checkpoint, so the knob is here."""

    assert all(m.p == 0.0 for m in _finetune().backbone.modules() if isinstance(m, t.nn.Dropout))

    model = _finetune(dropout=0.2)
    rates = [m.p for m in model.backbone.modules() if isinstance(m, t.nn.Dropout)]
    assert rates and all(p == 0.2 for p in rates)
    # the spec still records what pretraining used
    assert model.backbone.spec.dropout == 0.0


def test_a_finetune_runs_the_backbone_slower_than_the_head():
    model = _finetune(lr=1e-3, backbone_lr=1e-5)
    model._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = model.configure_optimizers()["optimizer"]

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    rates = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            # initial_lr, not lr: the schedule has already applied its warmup factor
            rates[id(p)] = group["initial_lr"]

    assert {id(p) for p in model.parameters()} == set(rates)
    assert all(rates[i] == 1e-5 for i in backbone_ids)
    assert all(rates[id(p)] == 1e-3 for p in model.head.parameters())


def test_a_finetune_does_not_decay_norms_and_biases():
    """Decay on a 1-D parameter is not regularization, it is a pull toward zero."""

    model = _finetune(weight_decay=0.05)
    model._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = model.configure_optimizers()["optimizer"]

    for group in optimizer.param_groups:
        expected = 0.05 if group["params"] and group["params"][0].ndim >= 2 else 0.0
        assert all((p.ndim >= 2) == (expected > 0) for p in group["params"])
        assert group["weight_decay"] == expected


def test_both_finetune_rates_decay_together():
    """One factor schedule over groups whose base rates differ by 100x."""

    model = _finetune(lr=1e-3, backbone_lr=1e-5, min_lr_frac=0.01, warmup=0.1)
    model._trainer = SimpleNamespace(estimated_stepping_batches=100)
    bundle = model.configure_optimizers()
    optimizer, scheduler = bundle["optimizer"], bundle["lr_scheduler"]["scheduler"]

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert all(g["lr"] < 0.02 * g["initial_lr"] for g in optimizer.param_groups)


def test_a_pretrained_backbone_arrives_with_its_weights():
    objective = MaskedMotionPredictor(SPEC, decoder_depth=1)
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
        backbone = load_encoder(str(path), "backbone")

    assert backbone.spec == SPEC
    assert t.equal(backbone.projection.weight, objective.backbone.projection.weight)


def test_top_k_recall_counts_positives_inside_the_top_k():
    metric = MultilabelTopKRecall(top_k=2)
    # two positives per row; the first row ranks both first, the second ranks one
    logits = t.tensor([[9.0, 8.0, 0.0, 0.0], [9.0, 0.0, 0.0, 8.0]])
    target = t.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    metric.update(logits, target)
    assert metric.compute().item() == 0.75


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
