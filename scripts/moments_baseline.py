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


@t.no_grad()
def collect(loader, passes: int = 1) -> tuple[t.Tensor, t.Tensor]:
    """``(features, labels)`` for a whole loader, moments already taken."""

    features, labels = [], []
    for _ in range(passes):
        for batch in loader:
            features.append(moments(batch["features"]))
            labels.append(batch["labels"])
    return t.cat(features), t.cat(labels)


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
def evaluate(head: nn.Module, validation: tuple[t.Tensor, t.Tensor], num_labels: int, device: str) -> dict:
    """The probe's metric set, so the numbers land in the same units as its own."""

    x, y = validation
    head.eval()
    logits = head(x.to(device)).cpu()
    target = y.int()

    metrics = {
        "val/macro_map": WindowMeanAveragePrecision(num_labels=num_labels),
        "val/micro_map": MultilabelAveragePrecision(num_labels=num_labels, average="micro"),
        "val/macro_f1s": MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro"),
        "val/top_1_rec": MultilabelTopKRecall(top_k=1),
        "val/top_3_rec": MultilabelTopKRecall(top_k=3),
        "val/top_5_rec": MultilabelTopKRecall(top_k=5),
    }
    return {name: float(metric(logits, target)) for name, metric in metrics.items()}


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

    print(f"collecting moments ({arguments.passes} pass over train, 1 over val)")
    train = collect(data.train_dataloader(), arguments.passes)
    validation = collect(data.val_dataloader())
    print(f"  train {tuple(train[0].shape)}   val {tuple(validation[0].shape)}   {num_labels} labels")

    num_channels = config.encoder.num_features
    results = {}
    for spec in arguments.channels:
        channels = (
            tuple(range(num_channels)) if spec == "all"
            else tuple(int(c) for c in spec.split(","))
        )
        if not all(0 <= c < num_channels for c in channels):
            raise SystemExit(f"channels {spec} out of range for {num_channels} channels")

        subset = (
            channel_slice(train[0], channels, num_channels),
            train[1],
        )
        held_out = (
            channel_slice(validation[0], channels, num_channels),
            validation[1],
        )
        print(f"\nchannels {channels} -- {subset[0].shape[1]} features")

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
        results[spec] = evaluate(head, held_out, num_labels, device)

    print(f"\n=== moments + linear, {arguments.config.name}, {arguments.passes} passes ===")
    metrics = list(next(iter(results.values())))
    width = max(len(s) for s in results)
    print(f"  {'channels':<{width}}  " + "  ".join(f"{m.removeprefix('val/'):>9}" for m in metrics))
    for spec, scores in results.items():
        print(f"  {spec:<{width}}  " + "  ".join(f"{scores[m]:>9.4f}" for m in metrics))


if __name__ == "__main__":
    main()
