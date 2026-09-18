#!/usr/bin/env python
"""How much of a cell's probe score survives cutting the labelled training set.

    uv run python scripts/label_efficiency.py \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed1/in_p__loss_p \\
        --output runs/representation/label_efficiency/amass_clean/medium_100ep/seed1/in_p__loss_p

    uv run python scripts/label_efficiency.py --moments \\
        --output runs/representation/label_efficiency/amass_clean/medium_100ep/moments

    uv run python scripts/label_efficiency.py --self-check

Phase 2 of `EVALUATION_PLAN.md`. The convex probe refit at 5 / 10 / 25 / 50 / 100% of the
labelled training set, against the full validation split every time. Window mAP at 100% is
weakly discriminative -- the cells this matrix ranks differ by 0.003-0.010 -- and the claim
a representation is supposed to support is that it needs *fewer* labels, which the 100%
number cannot express at all.

Same checkpoint/config contract as `probe_convex_mae.py`, and the same
`MotionConvexClassifier`, so a fraction of 1.0 here is that script's number recomputed
rather than a second, differently-shaped probe.

**One extraction, five fits.** The backbone forward is the whole bill and does not depend
on which rows are kept, exactly as `MotionConvexClassifier.sweep` extracts once and refits
per `weight_decay`. Re-extracting per fraction would multiply the cost of this phase by
five and change nothing about the answer.

**The weight decay sweep is redone per fraction**, and this is not optional. The L-BFGS
objective is `mean BCE + wd * ||W||^2`: the mean is over rows, so at 5% the same `wd`
shrinks against 1/20th of the data and is effectively 20x stronger. Carrying the 100%
optimum down the curve would report a regularization artefact as a data-efficiency result.

`--moments` fits the same head on `scripts/moments_baseline.py`'s four summary statistics
instead of a backbone. That is the floor the standing rules require beside every task, and
on this axis it is the interesting comparison rather than a formality: a curve that is
merely parallel to the moments curve, higher up, is a representation that separates better
without needing fewer labels.
"""

import argparse
import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from sometria.downstream.classifier_convex import (
    DEFAULT_TRAIN_PASSES,
    DEFAULT_WEIGHT_DECAYS,
    MotionConvexClassifier,
    extract_features,
    score_predictions,
)
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index

from moments_baseline import moments
from probe_baseline_mae import load_pretrained
from probe_convex_mae import BaselineBackbone, latest_checkpoint

#: The curve's x axis. Five points over a 20x range, ending at the published probe's own
#: training set, so the last point is a number that already exists to check against.
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)


class _MomentsBackbone(nn.Module):
    """The dumb floor wearing the shape :class:`MotionConvexClassifier` expects.

    ``embed_tokens`` is never called: :func:`moments_features` reduces a window itself,
    because with moments the reduction *is* the pooling and routing it through a token
    pooler would average 860 numbers that are not tokens. The spec exists only so the
    classifier sizes its ``LayerNorm`` and ``Linear`` to ``num_dofs * channels * 4``.
    """

    def __init__(self, num_dofs: int, num_channels: int) -> None:
        super().__init__()
        self.spec = SimpleNamespace(d_model=num_dofs * num_channels * 4, num_dofs=num_dofs)


class _Limited:
    """First ``n`` batches of a loader, re-iterable -- ``islice`` alone is consumed once and
    ``extract_features`` sweeps the loader ``passes`` times."""

    def __init__(self, loader, n: int) -> None:
        self.loader, self.n = loader, n

    def __iter__(self):
        return islice(iter(self.loader), self.n)


@t.no_grad()
def moments_features(norm, loader, device: str, passes: int = 1):
    """:func:`extract_features`' signature over raw-window moments instead of a backbone."""

    feats, labels, ids = [], [], []
    for _ in range(passes):
        for batch in loader:
            feats.append(norm(moments(batch["features"]).to(device)).cpu())
            labels.append(batch["labels"].float())
            ids.extend(batch["sample_id"])
    return t.cat(feats), t.cat(labels), ids


def sample_masks(
    sample_ids: list[str], fractions=FRACTIONS, seed: int = 13
) -> dict[float, t.Tensor]:
    """Row masks for each fraction, drawn over **samples** and nested.

    By `sample_id`, never by window, and this is the whole correctness of the x axis.
    Windows are crops of samples -- `LabelledWindows` draws a fresh random offset per
    sample per pass, so 20 passes give 20 overlapping views of one motion. Subsampling
    windows would keep a few crops of nearly every sample, which is not 5% of the labelled
    data: it is the full label set with less augmentation, and the curve would flatten for
    a reason that has nothing to do with label efficiency.

    Nested, because the draw is a prefix of one shuffle rather than an independent sample
    per fraction. Free, and it makes the curve a statement about *adding* labels to a fixed
    starting set instead of five unrelated draws whose differences include which samples
    each happened to get.

    Its own `Generator`, not the global RNG, so the draw does not move when the number of
    extraction passes or dataloader workers changes upstream of it.
    """

    unique = sorted(set(sample_ids))
    order = t.randperm(len(unique), generator=t.Generator().manual_seed(seed))
    rank = {unique[int(position)]: index for index, position in enumerate(order)}
    row_rank = t.tensor([rank[sample_id] for sample_id in sample_ids])

    # max(1, ...) so a tiny split still yields a fit rather than an empty design matrix.
    return {
        fraction: row_rank < max(1, round(fraction * len(unique))) for fraction in fractions
    }


def fixed_label_set(train_labels: t.Tensor, smallest: t.Tensor, val_labels: t.Tensor) -> t.Tensor:
    """The labels every fraction is scored on: those with a positive at the smallest one.

    The single most important property of this curve. `WindowMeanAveragePrecision` averages
    over the labels that have support and silently drops the rest, so scoring each fraction
    on whatever it happens to cover makes the 5% number a macro average over a *different
    vocabulary* than the 100% number -- five metrics printed as one curve, and the direction
    of the bias is not even fixed, since the labels a small draw loses are the rare ones,
    which are also the hard ones.

    Intersected with validation support so the reported label count is the count actually
    averaged over rather than an upper bound. That intersection is constant across
    fractions by construction: the validation split is never subsampled.

    The head is still fit over the *full* vocabulary and only scored through this mask. The
    fit is separable per label -- BCE means over columns, the penalty sums over them -- so
    the kept columns get identical weights either way, while slicing before the fit would
    change the loss/penalty ratio and quietly re-tune the regularization with the mask.
    """

    keep = (train_labels[smallest].sum(dim=0) > 0) & (val_labels.sum(dim=0) > 0)
    if not bool(keep.any()):
        raise SystemExit("no label survives the smallest fraction -- nothing to score")
    return keep


def curve(
    model: MotionConvexClassifier,
    train: tuple[t.Tensor, t.Tensor, list[str]],
    val: tuple[t.Tensor, t.Tensor],
    weight_decays=DEFAULT_WEIGHT_DECAYS,
    device: str = "cpu",
    seed: int = 13,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> dict:
    """Refit the head at every fraction, sweeping `weight_decay` within each."""

    train_feats, train_labels, train_ids = train
    val_feats, val_labels = val
    masks = sample_masks(train_ids, seed=seed)
    keep = fixed_label_set(train_labels, masks[min(masks)], val_labels)

    train_feats, train_labels = train_feats.to(device), train_labels.to(device)
    val_feats = val_feats.to(device)
    rows = []
    for fraction, mask in sorted(masks.items()):
        feats, labels = train_feats[mask.to(device)], train_labels[mask.to(device)]

        best = None
        sweep = {}
        for weight_decay in weight_decays:
            model.fit_features(feats, labels, max_iter, tol, weight_decay=weight_decay)
            logits = model.head(val_feats).cpu()
            # Sliced here and nowhere else: one `keep` for the whole curve is what makes
            # the five points one metric.
            metrics = score_predictions(logits[:, keep], val_labels[:, keep].int())
            sweep[weight_decay] = metrics
            if best is None or metrics["macro_map"] > sweep[best]["macro_map"]:
                best = weight_decay

        rows.append(
            {
                "fraction": fraction,
                "samples": int(len(set(i for i, m in zip(train_ids, mask) if m))),
                "windows": int(mask.sum()),
                "weight_decay": best,
                "metrics": sweep[best],
                "sweep": sweep,
            }
        )

    return {
        "metrics": {f"macro_map_{round(r['fraction'] * 100):03d}": r["metrics"]["macro_map"] for r in rows},
        "fractions": rows,
        "labels": {
            "scored": int(keep.sum()),
            "vocabulary": int(len(keep)),
            # Written out so two cells can be checked for having scored the same labels,
            # rather than trusted to have drawn the same subsample.
            "indices": [int(i) for i in keep.nonzero().flatten()],
        },
        "train_samples": len(set(train_ids)),
    }


def run(config, checkpoint: Path | None, device: str, limit_batches: int | None) -> dict:
    """Extract once, then the curve. Mirrors `probe_convex_mae.run_probe`'s setup."""

    seed = config.training.get("seed", 13)
    L.seed_everything(seed, workers=True)
    t.set_float32_matmul_precision("high")
    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)

    if checkpoint is None:
        backbone = _MomentsBackbone(config.encoder.num_dofs, config.encoder.num_features)
    else:
        backbone = BaselineBackbone(load_pretrained(checkpoint))
    model = MotionConvexClassifier(
        backbone, num_labels=num_labels, pool=config.model.get("pool", "mean")
    ).to(device)

    data = LabelledMotionDataModule(config)
    data.setup()
    loaders = [data.train_dataloader(), data.val_dataloader()]
    if limit_batches:
        loaders = [_Limited(loader, limit_batches) for loader in loaders]

    def extract(loader, passes):
        if checkpoint is None:
            return moments_features(model.norm, loader, device, passes)
        return extract_features(model.backbone, model.pooler, model.norm, loader, device, passes)

    # The window probe's 20 passes, for the same reason: one pass is one crop per sample,
    # and the fractions have to be cut from the training set the probe actually fits.
    train = extract(loaders[0], config.training.get("train_passes", DEFAULT_TRAIN_PASSES))
    val_feats, val_labels, _ = extract(loaders[1], 1)

    single_wd = config.model.get("weight_decay")
    weight_decays = [single_wd] if single_wd is not None else config.model.get("weight_decays", DEFAULT_WEIGHT_DECAYS)

    return curve(
        model,
        train,
        (val_feats, val_labels),
        weight_decays=list(weight_decays),
        device=device,
        seed=seed,
        max_iter=config.training.get("max_iter", 200),
        tol=config.training.get("tol", 1e-7),
    ) | {"source": "moments" if checkpoint is None else str(checkpoint)}


def demo() -> None:
    """Self-check: the subsample is by sample, and the vocabulary does not move."""

    ids = [f"take{i // 4}" for i in range(400)]     # 100 takes, 4 windows each
    masks = sample_masks(ids)

    previous = None
    for fraction, mask in sorted(masks.items()):
        kept = {sample_id for sample_id, keep in zip(ids, mask) if keep}
        dropped = {sample_id for sample_id, keep in zip(ids, mask) if not keep}
        # The property the whole x axis rests on: a take is wholly in or wholly out, so no
        # held-out window is a crop of a motion the head was fit on.
        assert not (kept & dropped), sorted(kept & dropped)[:4]
        assert len(kept) == round(fraction * 100), (fraction, len(kept))
        assert int(mask.sum()) == 4 * len(kept), (fraction, int(mask.sum()))
        # Nested: every fraction adds takes, never swaps them.
        assert previous is None or previous <= kept, fraction
        previous = kept

    t.manual_seed(0)
    labels = (t.rand(400, 6) > 0.6).float()
    # A label carried only by the last takes, so it is absent from every fraction but 100%
    # -- the case that silently changes the macro average's denominator.
    labels[:, 5] = 0.0
    labels[-8:, 5] = 1.0
    val_labels = (t.rand(120, 6) > 0.5).float()
    val_labels[:, 5] = 1.0                          # present in val: only train support decides

    keep = fixed_label_set(labels, masks[0.05], val_labels)
    assert not bool(keep[5]), "a label with no support at 5% must not be scored at 100% either"

    class _Stub(nn.Module):
        spec = SimpleNamespace(d_model=8, num_dofs=2)

    model = MotionConvexClassifier(_Stub(), num_labels=6, pool="mean")
    feats = t.randn(400, 8) + labels[:, :1]         # label 0 is learnable, the rest noise
    result = curve(
        model,
        (feats, labels, ids),
        (t.randn(120, 8), val_labels),
        weight_decays=(1e-2, 1e-4),
    )

    assert [row["fraction"] for row in result["fractions"]] == list(FRACTIONS)
    assert result["labels"]["scored"] == 5, result["labels"]
    # Every fraction scored the same vocabulary. `score_predictions` reports the recall
    # ladder only up to the vocabulary size, so an equal key set is itself evidence the
    # number of columns never changed under it.
    assert all(set(row["metrics"]) == set(result["fractions"][0]["metrics"]) for row in result["fractions"])
    assert all(row["weight_decay"] in (1e-2, 1e-4) for row in result["fractions"])
    assert result["fractions"][0]["samples"] == 5 and result["fractions"][-1]["samples"] == 100

    print("label_efficiency self-check ok:", result["metrics"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, help="pretrained checkpoint or run directory")
    parser.add_argument("--moments", action="store_true", help="the floor: no backbone")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    # Smoke only. Every reported number is over the whole split; this exists so the script
    # can be exercised end to end against a real checkpoint in under a minute.
    parser.add_argument("--limit-batches", type=int, help="first N batches only (smoke runs)")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args, overrides = parser.parse_known_args()

    if args.self_check:
        return demo()
    if not args.output or bool(args.checkpoint) == bool(args.moments):
        raise SystemExit("need --output and exactly one of --checkpoint / --moments")

    checkpoint = None
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    if args.checkpoint:
        checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
        OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output / "config.yaml")

    result = run(config, checkpoint, args.device, args.limit_batches)
    # Same envelope as probe_convex_mae.py and representation_geometry.py -- a "metrics"
    # dict inside metrics.json -- so the matrix runner's skip check and results/ generators
    # need no special case.
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"scored {result['labels']['scored']} of {result['labels']['vocabulary']} labels")
    for row in result["fractions"]:
        print(
            f"{row['fraction']:>6.0%}  {row['samples']:>5} samples  "
            f"wd {row['weight_decay']:g}  macro_map {row['metrics']['macro_map']:.4f}"
        )


if __name__ == "__main__":
    main()
