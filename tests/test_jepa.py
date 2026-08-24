"""Run with: python tests/test_jepa.py"""

import sys
import tempfile
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec
from sometria.downstream.classifier import MotionWindowClassifier
from sometria.models.jepa import MotionJEPA

SPEC = EncoderSpec(d_model=32, depth=1, num_heads=4)
NUM_DOFS = SPEC.num_dofs
NUM_FEATURES = SPEC.num_features
NUM_TOKENS = SPEC.grid_shape[0] * NUM_DOFS


def _features(batch=2, frames=240, seed=0):
    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, NUM_DOFS, NUM_FEATURES, generator=g)


def _model(**kwargs):
    return MotionJEPA(SPEC, predictor_depth=1, mask_ratio=0.25, **kwargs)


def test_jepa_predicts_teacher_embeddings_at_target_indices():
    model = _model()
    x = _features()
    prediction, target, target_valid, mask = model(x, generator=t.Generator().manual_seed(0))

    assert prediction.shape == target.shape == (2, mask.targets.shape[1], SPEC.d_model)
    assert target_valid.shape == mask.targets.shape
    assert mask.context.shape[1] == round(NUM_TOKENS * (1.0 - model.mask_ratio))
    assert mask.context.shape[1] + mask.targets.shape[1] == NUM_TOKENS
    assert model.prediction_loss(prediction, target, target_valid).isfinite()


def test_teacher_parameters_are_frozen():
    model = _model()

    assert all(not p.requires_grad for p in model.teacher.parameters())
    assert any(p.requires_grad for p in model.student.parameters())
    assert any(p.requires_grad for p in model.predictor.parameters())


def test_ema_moves_teacher_only_toward_student():
    model = _model()
    before = {name: p.detach().clone() for name, p in model.teacher.named_parameters()}
    student_before = {name: p.detach().clone() for name, p in model.student.named_parameters()}

    with t.no_grad():
        for p in model.student.parameters():
            p.add_(1.0)
    model.update_teacher(0.5)

    for name, teacher_param in model.teacher.named_parameters():
        expected = before[name] * 0.5 + model.student.state_dict()[name] * 0.5
        assert t.allclose(teacher_param, expected)
    for name, student_param in model.student.named_parameters():
        assert t.allclose(student_param, student_before[name] + 1.0)


def test_the_student_never_encodes_a_target_token():
    """Encoding must depend on context values only, or the objective is trivial."""

    model = _model().eval()
    x = _features(batch=1)
    g = t.Generator().manual_seed(0)
    with t.no_grad():
        _, _, _, mask = model(x, generator=t.Generator().manual_seed(0))
        before = model.student.embed_tokens(x, index=mask.context)

        scrambled = x.clone()
        scrambled[:] = t.randn(x.shape, generator=g)
        keep = t.zeros(NUM_TOKENS, dtype=t.bool)
        keep[mask.context[0]] = True
        grid = keep.reshape(-1, NUM_DOFS)
        for patch in range(grid.shape[0]):
            for dof in range(NUM_DOFS):
                if grid[patch, dof]:
                    lo = patch * SPEC.patch_size
                    scrambled[0, lo : lo + SPEC.patch_size, dof] = x[
                        0, lo : lo + SPEC.patch_size, dof
                    ]

        after = model.student.embed_tokens(scrambled, index=mask.context)

    assert t.allclose(before, after, atol=1e-5)


def test_checkpoint_round_trip_through_classifier_from_pretrained():
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "jepa.ckpt"
        t.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.0.0",
                "loops": {},
            },
            path,
        )
        teacher = MotionWindowClassifier.from_pretrained(path, num_labels=3)
        student = MotionWindowClassifier.from_pretrained(path, encoder="student", num_labels=3)

    assert teacher.backbone.spec == SPEC
    assert student.backbone.spec == SPEC
    teacher_state = teacher.backbone.state_dict()
    student_state = student.backbone.state_dict()
    for name, expected in model.teacher.state_dict().items():
        assert t.equal(teacher_state[name], expected), name
    for name, expected in model.student.state_dict().items():
        assert t.equal(student_state[name], expected), name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
