"""Is the representation better *organized* than the moments, or only better separable?

The linear probe answers one question -- how much of the label is a linear function of
the features -- and the moments baseline showed the answer is nearly the same for a
pretrained transformer and for four summary statistics. That leaves a gap the linear
probe cannot see: a representation can carry structure a linear head cannot reach.

kNN asks the other question. No head is trained and no parameters are fitted at all: a
validation window is scored by the labels of its nearest training windows, so what is
being measured is whether same-action windows land near each other under *any* geometry,
not whether a hyperplane separates them. If pretraining pulls further ahead here than it
does linearly, the representation has structure worth fine-tuning for; if it tracks the
linear result, that is the fine-tuning question answered cheaply in the negative.

    uv run python scripts/knn_probe.py --sources moments random \\
        runs/ablation_long/pretrain/seed_13/lightning_logs/version_0/checkpoints/last.ckpt

Each ``--sources`` entry is ``moments``, ``random`` (an untrained backbone, the control),
or a path to a pretraining checkpoint. Every source is scored on the same windows with
the same metrics, so the rows are comparable to each other.

They are only *loosely* comparable to the linear probe's numbers. A backbone source is
mean-pooled over the token grid, because kNN must not fit anything and the probe's
attentive pooler is learned -- so the honest linear row to read these against is the
``mean`` pooler probe, not the ``attentive`` one.

Windows only. A segmentation config's per-patch target is a different question and this
script refuses it rather than quietly averaging it away.
"""

import argparse
from pathlib import Path
import sys

import torch as t

from omegaconf import OmegaConf
from torchmetrics.classification import MultilabelAveragePrecision

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moments_baseline import moments  # noqa: E402

from sometria.architecture.encoder import MotionTransformerEncoder, EncoderSpec, load_encoder
from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.labels import load_label_vocabulary_index
from sometria.downstream.metrics import MultilabelTopKRecall, WindowMeanAveragePrecision

#: Neighbour counts scored in one pass. The distances are computed once, so the sweep is
#: nearly free and every source is read at the same set of k.
NEIGHBOURS = (1, 5, 10, 20, 50)

#: Softmax temperature on cosine similarity, the standard SSL kNN evaluation weighting
#: (Wu et al. 2018, as used by MoCo and DINO). Without a weighting, k=1 predicts a hard
#: 0/1 vector and average precision has nothing left to rank.
TEMPERATURE = 0.07


def encoder_features(backbone: MotionTransformerEncoder, device: str):
    """Mean-pooled backbone tokens: ``(B, T, D, C)`` -> ``(B, d_model)``.

    Mean rather than the probe's attentive pooling because kNN fits nothing -- an
    attentive pooler is parameters, and parameters trained on the labels would make this
    a probe with extra steps.
    """

    backbone = backbone.to(device).eval()

    def encode(features: t.Tensor) -> t.Tensor:
        return backbone.embed_tokens(features.to(device)).mean(dim=1).cpu()

    return encode


def source_features(name: str, config, device: str):
    """``(encode, label)`` for one ``--sources`` entry."""

    if name == "moments":
        return (lambda features: moments(features)), "moments"
    if name == "random":
        spec = EncoderSpec(**OmegaConf.to_container(config.encoder, resolve=True))
        return encoder_features(MotionTransformerEncoder(spec), device), "random-init"

    checkpoint = Path(name)
    if not checkpoint.exists():
        raise SystemExit(f"not 'moments', not 'random', and not a file: {name}")
    # The run directory, not a fixed number of levels up: a path ends
    # .../<run>/lightning_logs/version_N/checkpoints/last.ckpt, so counting parents lands
    # on "lightning_logs" and labels every row identically.
    parts = checkpoint.resolve().parts
    run = parts[parts.index("lightning_logs") - 1] if "lightning_logs" in parts else checkpoint.stem

    # "backbone" rather than "teacher": both resolve to the same weights for a masked
    # objective, and this way a JEPA checkpoint fails loudly instead of silently scoring
    # its teacher when the caller meant something else.
    return encoder_features(load_encoder(str(checkpoint), "backbone"), device), run


@t.no_grad()
def collect(loader, encode, passes: int = 1) -> tuple[t.Tensor, t.Tensor]:
    """``(features, labels)`` over ``passes`` sweeps of a loader."""

    features, labels = [], []
    for _ in range(passes):
        for batch in loader:
            features.append(encode(batch["features"]))
            labels.append(batch["labels"])
    return t.cat(features), t.cat(labels)


def standardize(train: t.Tensor, other: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Z-score both sets by the *training* mean and variance, per feature.

    Per feature, not per sample: a moment vector mixes joint angles with torques, whose
    scales differ by orders of magnitude, and an unstandardized distance would be a
    distance in whichever channel happens to be largest. This is where kNN departs from
    the linear head, which normalizes per sample with a LayerNorm -- a head can rescale
    its inputs with its weights and a distance cannot.
    """

    mean, std = train.mean(dim=0, keepdim=True), train.std(dim=0, keepdim=True)
    std = std.clamp(min=1e-6)
    return (train - mean) / std, (other - mean) / std


@t.no_grad()
def knn_predict(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    neighbours: tuple[int, ...],
    device: str,
    chunk: int = 1024,
) -> dict[int, t.Tensor]:
    """Similarity-weighted neighbour label averages, one prediction set per k.

    Cosine similarity on L2-normalized features, so the comparison is of direction rather
    than magnitude -- the same choice every SSL kNN evaluation makes, and the reason the
    magnitude differences standardization leaves behind do not decide the neighbours.
    """

    train_x = t.nn.functional.normalize(train_x, dim=1).to(device)
    train_y = train_y.float().to(device)
    val_x = t.nn.functional.normalize(val_x, dim=1).to(device)

    largest = max(neighbours)
    if largest > len(train_x):
        raise SystemExit(f"k={largest} needs more than {len(train_x)} training windows")

    predictions = {k: [] for k in neighbours}
    for start in range(0, len(val_x), chunk):
        similarity = val_x[start : start + chunk] @ train_x.T       # (n, train)
        top_similarity, top_index = similarity.topk(largest, dim=1)
        for k in neighbours:
            weight = (top_similarity[:, :k] / TEMPERATURE).softmax(dim=1)
            neighbour_labels = train_y[top_index[:, :k]]            # (n, k, labels)
            predictions[k].append(t.einsum("nk,nkl->nl", weight, neighbour_labels).cpu())

    return {k: t.cat(parts) for k, parts in predictions.items()}


def score(prediction: t.Tensor, target: t.Tensor, num_labels: int) -> dict:
    """The probe's metric set, so these land in the same units as everything else."""

    metrics = {
        "val/macro_map": WindowMeanAveragePrecision(num_labels=num_labels),
        "val/micro_map": MultilabelAveragePrecision(num_labels=num_labels, average="micro"),
        "val/top_1_rec": MultilabelTopKRecall(top_k=1),
        "val/top_3_rec": MultilabelTopKRecall(top_k=3),
    }
    return {name: float(metric(prediction, target.int())) for name, metric in metrics.items()}


def demo() -> None:
    """Self-check: a validation set that *is* the training set must be recovered at k=1."""

    t.manual_seed(0)
    x = t.randn(64, 16)
    y = (t.rand(64, 5) > 0.7).float()

    prediction = knn_predict(x, y, x, (1,), "cpu")[1]
    assert prediction.shape == y.shape, prediction.shape
    # k=1 against itself finds each point as its own nearest neighbour, so the labels
    # come back exactly. If the index arithmetic were off this is what would break.
    assert t.allclose(prediction, y), (prediction[:2], y[:2])

    # Standardization uses training statistics for both sets, so the training half comes
    # out with zero mean -- and the held-out half must not be re-centred on its own.
    train, held_out = standardize(t.randn(100, 8) * 5 + 3, t.randn(50, 8) * 5 + 3)
    assert train.mean().abs() < 1e-5, train.mean()
    assert train.std().sub(1).abs() < 0.1, train.std()

    print("knn_probe self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["moments"],
        metavar="SRC",
        help='"moments", "random", or a pretraining checkpoint path',
    )
    parser.add_argument("--passes", type=int, default=1, help="times to collect the training set")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    arguments = parser.parse_args()

    if arguments.self_check:
        return demo()

    t.manual_seed(arguments.seed)
    device = "cuda" if t.cuda.is_available() else "cpu"

    config = OmegaConf.load(arguments.config)
    OmegaConf.update(config, "dataloader.num_workers", 4)
    if config.dataloader.get("label_patches"):
        raise SystemExit(
            "this config has label_patches set, so its target is per time patch. "
            "kNN here scores whole windows -- use a classification config."
        )

    _, num_labels = load_label_vocabulary_index(
        config.dataloader.root, config.dataloader.label_set
    )
    data = LabelledMotionDataModule(config)
    data.setup("fit")

    results = {}
    for name in arguments.sources:
        # Rebuilt per source rather than collected once: the encoders disagree about what
        # a feature is, and only the loader is shared.
        encode, label = source_features(name, config, device)
        print(f"\ncollecting {label} ({arguments.passes} pass over train, 1 over val)")

        t.manual_seed(arguments.seed)   # so "random" is one fixed control, not a lottery
        train_x, train_y = collect(data.train_dataloader(), encode, arguments.passes)
        val_x, val_y = collect(data.val_dataloader(), encode, 1)
        print(f"  train {tuple(train_x.shape)}   val {tuple(val_x.shape)}   {num_labels} labels")

        train_x, val_x = standardize(train_x, val_x)
        for k, prediction in knn_predict(train_x, train_y, val_x, NEIGHBOURS, device).items():
            results[f"{label} k={k}"] = score(prediction, val_y, num_labels)

    print(f"\n=== kNN, no head trained, {arguments.config.name} ===")
    metrics = list(next(iter(results.values())))
    width = max(len(name) for name in results)
    print(f"  {'source':<{width}}  " + "  ".join(f"{m.removeprefix('val/'):>11}" for m in metrics))
    for name, scores in results.items():
        print(f"  {name:<{width}}  " + "  ".join(f"{scores[m]:>11.4f}" for m in metrics))


if __name__ == "__main__":
    main()
