"""Do the encoder and the moments know different things?

The moments reach 87% of a pretrained backbone's macro mAP on window classification,
which invites an obvious idea: feed the head both and get a stronger model. Before
building that into the architecture, this asks whether there is anything to gain.

The two streams carry the same *information* -- mean, std, min and max over the window
are a deterministic function of the frames the encoder already reads. What they might
carry differently is *inductive bias*: min and max over 240 frames are global order
statistics, and softmax attention over patch tokens approximates them poorly. So the
question is empirical, and one linear head answers it.

Three heads, identical except for their input:

    moments   860 numbers, nothing learned
    encoder   mean-pooled backbone tokens
    both      the two concatenated, each stream standardized on its own first

If ``both`` beats ``encoder`` by more than the seed noise, the streams are complementary
and an architectural fusion is worth building. If it lands on ``encoder``, the backbone
already represents everything the moments carry and the idea is dead for the cost of one
script.

    uv run python scripts/fusion_probe.py \\
        --checkpoint runs/ablation_long/pretrain/seed_13/lightning_logs/version_0/checkpoints/last.ckpt

Mean pooling, not the probe's attentive pooler: concatenation has to happen on a fixed
vector, and a learned pooler would make ``encoder`` and ``both`` differ by two things at
once. Absolute numbers therefore sit below the attentive probe's; the comparison between
the three rows is the point.
"""

import argparse
from pathlib import Path
import sys

import torch as t
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moments_baseline import evaluate, moments, train_head  # noqa: E402

from sometria.architecture.encoder import load_encoder
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index


@t.no_grad()
def collect(loader, backbone, device, passes: int = 1):
    """``(moments, encoder, labels)`` over ``passes`` sweeps of a loader."""

    moment_rows, encoder_rows, labels = [], [], []
    for _ in range(passes):
        for batch in loader:
            features = batch["features"]
            moment_rows.append(moments(features))
            encoder_rows.append(backbone.embed_tokens(features.to(device)).mean(dim=1).cpu())
            labels.append(batch["labels"])
    return t.cat(moment_rows), t.cat(encoder_rows), t.cat(labels)


def standardize(train: t.Tensor, other: t.Tensor):
    """Per-feature z-score by the *training* statistics.

    Per stream rather than over the concatenation: 860 moments and 256 encoder dimensions
    live on unrelated scales, and a single normalization would let whichever stream
    happens to be larger dominate the head's initial gradients.
    """

    mean, std = train.mean(dim=0, keepdim=True), train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train - mean) / std, (other - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="teacher", choices=("backbone", "teacher", "student"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    arguments = parser.parse_args()

    t.manual_seed(arguments.seed)
    device = "cuda" if t.cuda.is_available() else "cpu"

    config = OmegaConf.load(arguments.config)
    if config.dataloader.get("label_patches"):
        raise SystemExit("this config's target is per patch; use a classification config")

    _, num_labels = load_label_vocabulary_index(config.dataloader.root, config.dataloader.label_set)
    data = LabelledMotionDataModule(config)
    data.setup("fit")

    backbone = load_encoder(str(arguments.checkpoint), arguments.encoder).to(device).eval()

    print(f"collecting ({arguments.passes} pass over train, 1 over val)")
    train_moments, train_encoder, train_y = collect(
        data.train_dataloader(), backbone, device, arguments.passes
    )
    val_moments, val_encoder, val_y = collect(data.val_dataloader(), backbone, device, 1)
    print(f"  moments {tuple(train_moments.shape)}  encoder {tuple(train_encoder.shape)}")

    train_moments, val_moments = standardize(train_moments, val_moments)
    train_encoder, val_encoder = standardize(train_encoder, val_encoder)

    arms = {
        "moments": (train_moments, val_moments),
        "encoder": (train_encoder, val_encoder),
        "both": (
            t.cat([train_moments, train_encoder], dim=1),
            t.cat([val_moments, val_encoder], dim=1),
        ),
    }

    results = {}
    for name, (train_x, val_x) in arms.items():
        print(f"\n{name} -- {train_x.shape[1]} features")
        t.manual_seed(arguments.seed)     # same init and same batch order for every arm
        head = train_head(
            (train_x, train_y.float()), num_labels,
            epochs=arguments.epochs, batch_size=arguments.batch_size,
            lr=arguments.lr, device=device,
        )
        results[name] = evaluate(head, (val_x, val_y), num_labels, device)

    print(f"\n=== fusion probe, {arguments.config.name} ===")
    metrics = list(next(iter(results.values())))
    print(f"  {'arm':<10}" + "".join(f"{m.removeprefix('val/'):>13}" for m in metrics))
    for name, scores in results.items():
        print(f"  {name:<10}" + "".join(f"{scores[m]:>13.4f}" for m in metrics))

    gain = results["both"]["val/macro_map"] - results["encoder"]["val/macro_map"]
    print(f"\n  both - encoder = {gain:+.4f} macro mAP")


if __name__ == "__main__":
    main()
