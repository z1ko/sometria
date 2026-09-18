#!/usr/bin/env python
"""Participation ratio and CKA over the features the convex probe fits, per matrix cell.

    uv run python scripts/representation_geometry.py \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed1/in_p__loss_p \\
        --output runs/representation/geometry/amass_clean/medium_100ep/seed1/in_p__loss_p

    uv run python scripts/representation_geometry.py --cka \\
        runs/representation/geometry/amass_clean/medium_100ep/seed*/in_* \\
        --output runs/representation/geometry/amass_clean/medium_100ep/cka

    uv run python scripts/representation_geometry.py --self-check

The cheapest thing in the evaluation plan and the one that runs first, because it can
invalidate a later table before that table is measured: if a cell's pooled features are
near rank-1, every task score for that cell is explained before anyone scores it. No
labels, no head, no baseline.

Same checkpoint/config contract as `probe_convex_mae.py`, and it builds the same
`MotionConvexClassifier` rather than a pooler of its own -- the statistic is only about the
probe's table if it is measured on the probe's exact tensor, pooler and parameter-free
`LayerNorm` included. The head is constructed and never used.

**The validation split, tiled, one pass.** Not the training split the probe literally fits,
and the reason is CKA: `LabelledWindows(tiles=False)` draws a random crop per sample per
pass, so two cells' training features are two different sets of windows and the row-wise
alignment CKA requires does not exist. Evaluation tiles are deterministic, so cell A's row
i and cell B's row i are the same frames by construction, and the participation ratio is
then measured on that same matrix instead of a second one. The alternative -- trusting two
processes to draw identical crops from identical seeds through identical worker counts --
is a correctness argument about `DataLoader` internals, and the failure mode is a silently
meaningless CKA rather than an error.

`features.pt` is written beside `metrics.json` because the CKA pass needs every cell at
once and the backbone forward is the whole bill. About 10 MB per cell at `mean` pooling.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import lightning as L
import torch as t
from omegaconf import OmegaConf

from sometria.downstream.classifier_convex import MotionConvexClassifier, extract_features
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.geometry import linear_cka, participation_ratio
from sometria.downstream.labels import load_label_vocabulary_index

from probe_baseline_mae import load_pretrained
from probe_convex_mae import BaselineBackbone, latest_checkpoint


@t.no_grad()
def cell_features(config, checkpoint: Path, device: str) -> tuple[t.Tensor, list[str]]:
    """Pooled + normalized validation features, exactly as the convex probe scores them."""

    # Deterministic tiles make this seed irrelevant to the features; it is set anyway so a
    # config that starts drawing anything random does not silently make runs incomparable.
    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)
    model = MotionConvexClassifier(
        BaselineBackbone(load_pretrained(checkpoint)),
        num_labels=num_labels,
        pool=config.model.get("pool", "mean"),
    ).to(device)

    data = LabelledMotionDataModule(config)
    data.setup()
    feats, _, sample_ids = extract_features(
        model.backbone, model.pooler, model.norm, data.val_dataloader(), device
    )
    return feats, sample_ids


def geometry(feats: t.Tensor) -> dict:
    """Participation ratio plus the context needed to read it: how many rows, how many
    dimensions it could have used, and how much variance there was to spread."""

    n, d = feats.shape
    return {
        "metrics": {
            "participation_ratio": participation_ratio(feats),
            # tr(C) -- the numerator's square root. Reported because PR is scale-free and
            # a cell can spread its variance over many directions while having almost none.
            "total_variance": float(feats.double().var(dim=0).sum()),
        },
        "n": int(n),
        "d": int(d),
    }


def cka_report(dirs: list[Path]) -> dict:
    """All-pairs linear CKA over cached cell features, split into cross-cell and cross-seed.

    The split is the reading, not a convenience: a CKA of 0.934 between two cells means
    nothing until the same number is computed between two seeds of one cell, because that
    is the floor a difference has to clear to be about the objective rather than about
    initialization. `results/decoder_capacity/README.md` reports 0.934 against 0.959 and
    concludes from the gap, not from the 0.934.

    Cell and seed come off the path -- `.../seed<S>/<cell>` -- because that is what
    `geometry_matrix.sh` writes and inventing a second place to record them is a second
    place for them to disagree. Pairs differing in both are dropped: they confound the two
    axes and belong to neither bucket.
    """

    loaded = {}
    for directory in dirs:
        payload = t.load(directory / "features.pt")
        seed = directory.parent.name if directory.parent.name.startswith("seed") else ""
        loaded[(seed, directory.name)] = payload

    reference = next(iter(loaded.values()))["sample_ids"]
    for key, payload in loaded.items():
        # A silently misaligned CKA is a plausible-looking number, so this is checked
        # rather than assumed: two cells scored on different window sets (a changed split,
        # a changed window_frames, a half-written cache) would otherwise just read low.
        if payload["sample_ids"] != reference:
            raise SystemExit(f"{key} was extracted over different windows than {next(iter(loaded))}")

    pairs, cross_cell, cross_seed = {}, [], []
    for (seed_a, cell_a), (seed_b, cell_b) in combinations(sorted(loaded), 2):
        value = linear_cka(loaded[(seed_a, cell_a)]["features"], loaded[(seed_b, cell_b)]["features"])
        pairs[f"{seed_a}/{cell_a} vs {seed_b}/{cell_b}"] = value
        if seed_a == seed_b and cell_a != cell_b:
            cross_cell.append(value)
        elif cell_a == cell_b and seed_a != seed_b:
            cross_seed.append(value)

    def summary(values: list[float], prefix: str) -> dict:
        if not values:
            return {}
        return {
            f"{prefix}_mean": mean(values),
            f"{prefix}_sd": stdev(values) if len(values) > 1 else 0.0,
            f"{prefix}_n": len(values),
        }

    return {
        "metrics": summary(cross_cell, "cross_cell") | summary(cross_seed, "cross_seed"),
        "pairs": pairs,
        "n": int(len(reference)),
        "d": int(next(iter(loaded.values()))["features"].shape[1]),
    }


def demo() -> None:
    """Self-check: the two statistics against cases whose answer is known in closed form."""

    t.manual_seed(0)
    x = t.randn(8192, 16)

    # CKA of a matrix against itself is 1 by construction of the denominator; anything else
    # means the centring is applied asymmetrically.
    assert abs(linear_cka(x, x) - 1.0) < 1e-9, linear_cka(x, x)

    # Invariance to isotropic scale and orthogonal rotation -- the two symmetries that make
    # CKA the right instrument for comparing transformers at all.
    rotation = t.linalg.qr(t.randn(16, 16))[0]
    assert abs(linear_cka(x, 3.0 * x @ rotation) - 1.0) < 1e-6, linear_cka(x, 3.0 * x @ rotation)

    # A constant offset must vanish. This is the check that fails if the columns are not
    # centred, and an uncentred "CKA" reads ~1.0 for everything.
    shifted = x + 100.0
    assert abs(linear_cka(x, shifted) - 1.0) < 1e-6
    assert abs(linear_cka(x, t.randn(8192, 16) + 5.0) - linear_cka(x, t.randn(8192, 16))) < 0.05

    # Independent gaussians: not 0 at finite n, but far from anything a real pair reaches.
    assert linear_cka(x, t.randn(8192, 16)) < 0.1, linear_cka(x, t.randn(8192, 16))

    # Isotropic -> d. Not exactly d: the sample spectrum of a Wishart draw has width, which
    # pulls PR to d / (1 + d/n) ~= 15.97 here, so the tolerance is the estimator's, not the
    # implementation's.
    assert abs(participation_ratio(x) - 16) < 0.5, participation_ratio(x)

    # Rank-1 -> 1, the case the whole statistic exists to catch.
    rank_one = t.randn(8192, 1) @ t.randn(1, 16)
    assert abs(participation_ratio(rank_one) - 1.0) < 1e-6, participation_ratio(rank_one)

    # And a spectrum known exactly: variances (4,1,1,1) give 7^2/19 = 2.58, so this fails
    # if PR is counting nonzero directions instead of weighting them by variance.
    scaled = t.randn(200000, 4) * t.tensor([2.0, 1.0, 1.0, 1.0])
    assert abs(participation_ratio(scaled) - 49 / 19) < 0.02, participation_ratio(scaled)

    print("representation_geometry self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, help="pretrained checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--cka", type=Path, nargs="+", metavar="DIR", help="cell dirs holding features.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args, overrides = parser.parse_known_args()

    if args.self_check:
        return demo()
    if not args.output or bool(args.checkpoint) == bool(args.cka):
        raise SystemExit("need --output and exactly one of --checkpoint / --cka")

    args.output.mkdir(parents=True, exist_ok=True)

    if args.cka:
        result = cka_report(list(args.cka))
    else:
        checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
        config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
        OmegaConf.update(config, "model.checkpoint", str(checkpoint))
        OmegaConf.save(config, args.output / "config.yaml")

        feats, sample_ids = cell_features(config, checkpoint, args.device)
        t.save({"features": feats, "sample_ids": sample_ids}, args.output / "features.pt")
        result = geometry(feats) | {"pool": config.model.get("pool", "mean")}

    # Same envelope as probe_convex_mae.py -- a "metrics" dict inside metrics.json -- so
    # the matrix runner's skip check and the results/ generators need no special case.
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
