"""Run with: python tests/test_preprocess.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.preprocess import _babel_annotation_rows


SEQUENCE = {
    "feat_p": "MPIHDM05/MPI_HDM05/dg/HDM_dg_03-11_03_120_poses.npz",
    "dur": 5.4,
    "seq_ann": {"labels": [{"raw_label": "walking", "proc_label": "walk", "act_cat": ["walk"]}]},
    "frame_ann": {
        "labels": [
            {"raw_label": "lifting", "proc_label": "lift", "act_cat": ["lift something", "bend"],
             "start_t": 1.235, "end_t": 2.818},
        ]
    },
}


def test_babel_rows_span_sequence_and_frame_labels():
    rows = _babel_annotation_rows(SEQUENCE)
    assert all(r["babel_match_key"] == "HDM05/dg/hdmdg031103120" for r in rows)

    seq = [r for r in rows if r["label_type"] == "sequence"]
    assert {r["ontology"] for r in seq} == {"raw", "proc", "act_cat"}
    assert all(r["start_t"] == 0.0 and r["end_t"] == 5.4 for r in seq)  # spans the whole take

    frame = [r for r in rows if r["label_type"] == "frame"]
    assert all(r["start_t"] == 1.235 and r["end_t"] == 2.818 for r in frame)
    # act_cat is a list, so one segment can contribute several rows in that ontology
    assert sorted(r["label"] for r in frame if r["ontology"] == "act_cat") == ["bend", "lift something"]


def test_babel_rows_tolerate_withheld_fields():
    # frame_ann is absent for ~40% of sequences
    assert all(r["label_type"] == "sequence" for r in _babel_annotation_rows(SEQUENCE | {"frame_ann": None}))

    # act_cat is null for every label in BABEL's test split; raw/proc must still survive
    withheld = {
        "feat_p": SEQUENCE["feat_p"], "dur": 3.0, "frame_ann": None,
        "seq_ann": {"labels": [{"raw_label": "jump", "proc_label": "jump", "act_cat": None}]},
    }
    rows = _babel_annotation_rows(withheld)
    assert {r["ontology"] for r in rows} == {"raw", "proc"}
    assert len(rows) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
