"""How much of the probe's score is available without learning a representation at all?

Four moments -- mean, standard deviation, min, max -- of every DOF and channel over the
window, fed to the same multilabel linear head the probe uses. 43 DOFs x 5 channels x 4
moments = 860 numbers per window, computed arithmetically, no pretraining involved.

This is the baseline that says whether a probe score is *good*, which the random-backbone
control cannot: that control shows pretraining beats no pretraining, and says nothing
about whether the task was hard. If the moments come close to the probe, most of the
score was available from summary statistics and the representation is earning little.

    uv run python scripts/moments_baseline.py
    uv run python scripts/moments_baseline.py --config config/experiment_linear_probe.yaml
    uv run python scripts/moments_baseline.py --passes 4 --epochs 40
    uv run python scripts/moments_baseline.py --channels 0,1 0,1,4 all

``--channels`` restricts the moments to a subset of feature channels and trains one head
per subset, mirroring the ``loss_channels`` arms of ``scripts/ablate.sh channels``. It
answers a question that ablation cannot: those arms vary what the *loss* scores while the
*input* always carries all five channels, so a gap from adding torque to the objective
could equally be a gap from torque being an informative input. Here the input is the only
thing that changes. A subset is a slice of the collected moments, so every arm costs one
collection between them.

Read the output beside `scripts/results.py runs/<probes> val/macro_map`. The data, the
label set, the head and the metrics are the probe's, so the numbers are comparable; the
only difference is what enters the head.

Given a config that sets ``dataloader.label_patches`` -- config/experiment_segmentation.yaml
-- the target is per time patch and two baselines are reported instead of one:

``patch``    moments of each patch's own frames, predicting that patch. Statistics that
             can vary in time, but see 8 frames and no context. This is the real
             competitor for a segmentation probe.
``window``   moments of the whole window, one prediction broadcast to every patch. Zero
             temporal resolution by construction, so its ``boundary_f1s`` is exactly 0.
             The floor that says how much of the patch score is available without
             localizing anything at all.

``--passes`` collects the training set more than once. Training windows are drawn at a
random offset per epoch, so a probe sees a fresh crop each time and this, caching once,
would not. More passes buy back that variety at a linear cost in memory.
"""

import argparse
from pathlib import Path

import torch as t
import torch.nn as nn

from omegaconf import OmegaConf
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.downstream.metrics import MultilabelTopKRecall, WindowMeanAveragePrecision
from sometria.downstream.segmentation import boundary_f1


#: Names of the moments, in the order :func:`moments` stacks them.
MOMENTS = ("mean", "std", "min", "max")


def channel_slice(features: t.Tensor, channels: tuple[int, ...], num_channels: int) -> t.Tensor:
    """Keep only ``channels`` of a ``(B, D * C * len(MOMENTS))`` moment vector.

    The flatten is ``(D, C, moment)``, so a channel is a stride rather than a block --
    the same layout trap as :meth:`~sometria.models.mae.MaskedAutoencoder.scored_values`,
    and wrong here would silently select the wrong DOFs instead of the wrong channels.
    """

    grid = features.unflatten(1, (-1, num_channels, len(MOMENTS)))
    return grid[:, :, t.tensor(channels), :].flatten(start_dim=1)


def moments(features: t.Tensor) -> t.Tensor:
    """``(B, T, D, C)`` -> ``(B, D * C * 4)``: mean, std, min and max over time.

    Deliberately the dumbest summary that still distinguishes a joint that swings from
    one that is held: no derivatives beyond the ones already in the channels, no
    frequency content, no cross-joint terms.
    """

    stacked = t.stack(
        [
            features.mean(dim=1),
            features.std(dim=1),
            features.amin(dim=1),
            features.amax(dim=1),
        ],
        dim=-1,
    )
    return stacked.flatten(start_dim=1)


def patch_moments(features: t.Tensor, num_patches: int) -> t.Tensor:
    """``(B, T, D, C)`` -> ``(B, num_patches, D * C * 4)``: :func:`moments` per time patch.

    The same four numbers over each patch's own frames rather than over the window. This
    is what an order-invariant model looks like when it is allowed to vary in time: it
    can localize, but only from 8 frames at a time and with no context on either side.
    """

    batch, frames, dofs, channels = features.shape
    if frames % num_patches:
        raise ValueError(f"{frames} frames do not divide into {num_patches} patches")

    split = features.reshape(batch, num_patches, frames // num_patches, dofs, channels)
    stacked = t.stack(
        [split.mean(dim=2), split.std(dim=2), split.amin(dim=2), split.amax(dim=2)],
        dim=-1,
    )
    return stacked.flatten(start_dim=2)


@t.no_grad()
def collect(loader, passes: int = 1, num_patches: int | None = None) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """``(window moments, patch moments, labels)`` for a whole loader.

    Both feature sets come out of one pass, because collecting the data twice to compute
    two summaries of it is the expensive half done twice.
    """

    window, patch, labels = [], [], []
    for _ in range(passes):
        for batch in loader:
            window.append(moments(batch["features"]))
            if num_patches:
                patch.append(patch_moments(batch["features"], num_patches))
            labels.append(batch["labels"])

    return t.cat(window), (t.cat(patch) if patch else t.zeros(0)), t.cat(labels)


def train_head(
    train: tuple[t.Tensor, t.Tensor],
    num_labels: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> nn.Module:
    """One Linear over standardized moments -- the probe's head, on different inputs.

    LayerNorm rather than a fitted mean and variance for the same reason the probe uses
    one: it is the head that has to be identical, and the probe normalizes its pooled
    vector this way.
    """

    x, y = train
    head = nn.Sequential(
        nn.LayerNorm(x.shape[1], elementwise_affine=False, eps=1e-6),
        nn.Linear(x.shape[1], num_labels),
    ).to(device)

    optimizer = t.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    schedule = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = nn.BCEWithLogitsLoss()

    x, y = x.to(device), y.to(device)
    for epoch in range(epochs):
        head.train()
        order = t.randperm(len(x), device=device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            index = order[start : start + batch_size]
            loss = loss_function(head(x[index]), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        schedule.step()
        print(f"  epoch {epoch + 1:>3}/{epochs}  train loss {total / len(x):.4f}", end="\r")

    print()
    return head


@t.no_grad()
def evaluate(
    head: nn.Module,
    validation: tuple[t.Tensor, t.Tensor],
    num_labels: int,
    device: str,
    *,
    patches: int | None = None,
) -> dict:
    """The probe's metric set, so the numbers land in the same units as its own.

    For a segmentation target the rows are ``(window, patch)`` pairs, which is what
    :class:`~sometria.downstream.segmentation.MotionSegmenter` scores too -- plus
    ``boundary_f1s``, which only exists once a prediction can change within a window.
    """

    x, y = validation
    head.eval()
    logits = head(x.to(device)).cpu()

    metrics = {
        "val/macro_map": WindowMeanAveragePrecision(num_labels=num_labels),
        "val/micro_map": MultilabelAveragePrecision(num_labels=num_labels, average="micro"),
        "val/macro_f1s": MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro"),
        "val/top_1_rec": MultilabelTopKRecall(top_k=1),
        "val/top_3_rec": MultilabelTopKRecall(top_k=3),
        "val/top_5_rec": MultilabelTopKRecall(top_k=5),
    }
    if patches is None:
        return {name: float(metric(logits, y.int())) for name, metric in metrics.items()}

    shaped = logits.unflatten(0, (-1, patches))
    scores = {
        name: float(metric(logits, y.int())) for name, metric in metrics.items()
    }
    scores["val/boundary_f1s"] = float(boundary_f1(shaped, y.unflatten(0, (-1, patches))))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--passes", type=int, default=1, help="times to collect the training set")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["all"],
        metavar="SPEC",
        help='channel subsets, comma-separated indices or "all" (e.g. 0,1 0,1,4 all)',
    )
    arguments = parser.parse_args()

    t.manual_seed(arguments.seed)
    device = "cuda" if t.cuda.is_available() else "cpu"

    config = OmegaConf.load(arguments.config)
    OmegaConf.update(config, "dataloader.num_workers", 4)
    _, num_labels = load_label_vocabulary_index(
        config.dataloader.root, config.dataloader.label_set
    )

    data = LabelledMotionDataModule(config)
    data.setup("fit")

    # Set by config/experiment_segmentation.yaml, absent for the classification configs.
    patches = config.dataloader.get("label_patches")

    print(f"collecting moments ({arguments.passes} pass over train, 1 over val)")
    train_window, train_patch, train_y = collect(data.train_dataloader(), arguments.passes, patches)
    val_window, val_patch, val_y = collect(data.val_dataloader(), 1, patches)
    print(f"  train {tuple(train_window.shape)}   val {tuple(val_window.shape)}   {num_labels} labels")

    if patches:
        # One row per (window, patch). "patch" reads that patch's own frames; "window"
        # reads the whole window and predicts the same thing for every patch, so it can
        # name the action and never say when.
        variants = {
            "patch": (
                train_patch.flatten(end_dim=1), train_y.flatten(end_dim=1),
                val_patch.flatten(end_dim=1), val_y.flatten(end_dim=1),
            ),
            "window": (
                train_window.repeat_interleave(patches, dim=0), train_y.flatten(end_dim=1),
                val_window.repeat_interleave(patches, dim=0), val_y.flatten(end_dim=1),
            ),
        }
        print(f"  segmentation: {patches} patches per window, "
              f"{tuple(variants['patch'][0].shape)} rows")
    else:
        variants = {"window": (train_window, train_y, val_window, val_y)}

    num_channels = config.encoder.num_features
    results = {}
    for variant, (train_x, train_labels, val_x, val_labels) in variants.items():
        for spec in arguments.channels:
            channels = (
                tuple(range(num_channels)) if spec == "all"
                else tuple(int(c) for c in spec.split(","))
            )
            if not all(0 <= c < num_channels for c in channels):
                raise SystemExit(f"channels {spec} out of range for {num_channels} channels")

            subset = (channel_slice(train_x, channels, num_channels), train_labels)
            held_out = (channel_slice(val_x, channels, num_channels), val_labels)

            name = spec if len(variants) == 1 else f"{variant}/{spec}"
            print(f"\n{name} -- {subset[0].shape[1]} features, {len(subset[0])} rows")

            # Reseeded per arm so a subset is not also a different initialization.
            t.manual_seed(arguments.seed)
            head = train_head(
                subset,
                num_labels,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                lr=arguments.lr,
                device=device,
            )
            results[name] = evaluate(
                head, held_out, num_labels, device, patches=patches if patches else None
            )

    print(f"\n=== moments + linear, {arguments.config.name}, {arguments.passes} passes ===")
    metrics = list(next(iter(results.values())))
    width = max(len(name) for name in results)
    print(f"  {'arm':<{width}}  " + "  ".join(f"{m.removeprefix('val/'):>11}" for m in metrics))
    for name, scores in results.items():
        print(f"  {name:<{width}}  " + "  ".join(f"{scores[m]:>11.4f}" for m in metrics))


if __name__ == "__main__":
    main()
