"""Run with: python tests/test_dataset.py"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.dataset import WindowCollate

WINDOW = 80


def _item(n_frames, sample_id="AMASS:take", fill=None):
    features = t.arange(n_frames, dtype=t.float32).reshape(-1, 1, 1).expand(n_frames, 6, 5).clone()
    return {"features": features if fill is None else fill, "sample_id": sample_id, "source_path": "a/b.csv"}


def test_evaluation_takes_the_centre_of_the_motion_every_time():
    """A random crop while validating moves val/loss for reasons unrelated to the model."""

    collate = WindowCollate(WINDOW, random_offset=False)
    batch = collate([_item(100)])

    assert batch["features"].shape == (1, WINDOW, 6, 5)
    # (100 - 80) // 2 == 10, so the window starts at frame 10 and does so every epoch
    assert batch["features"][0, 0, 0, 0].item() == 10.0
    assert t.equal(collate([_item(100)])["features"], batch["features"])


def test_a_training_crop_stays_inside_the_motion():
    collate = WindowCollate(WINDOW, random_offset=True)
    for _ in range(20):
        start = collate([_item(100)])["features"][0, 0, 0, 0].item()
        assert 0 <= start <= 20


def test_a_motion_shorter_than_a_window_is_refused_not_padded():
    """Padding scores as motionless, so motion-aware masking would keep it and spend the
    context budget on frames that are not there. min_frames drops these; this is its backstop."""

    try:
        WindowCollate(WINDOW)([_item(40, sample_id="AMASS:short")])
    except ValueError as error:
        assert "AMASS:short" in str(error) and "min_frames" in str(error)
    else:
        raise AssertionError("expected a ValueError naming the sample and the fix")


def test_a_batch_keeps_its_provenance():
    batch = WindowCollate(WINDOW)([_item(90, "AMASS:one"), _item(200, "AMASS:two")])

    assert batch["features"].shape == (2, WINDOW, 6, 5)
    assert batch["sample_id"] == ["AMASS:one", "AMASS:two"]
    assert "valid" not in batch


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
