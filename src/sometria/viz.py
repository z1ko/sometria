"""Figures for a finished run: what the reconstruction misses, what the probe confuses.

Every function returns a :class:`~matplotlib.figure.Figure` and draws nothing, so a
notebook displays it and a script saves it without either knowing about the other.

Two things are worth knowing before reading a reconstruction plot. The loss runs on
:func:`~sometria.models.window.standardize_tokens` output -- zero mean and unit variance
*per token* -- so the model never predicts a patch's magnitude, only its shape, and a
"true vs predicted" overlay in raw joint angles would be showing scale the model was
never asked for. Everything here therefore plots error in standardized space. And a
token is one DOF over ``patch_size`` frames, so the natural picture of where the error
lives is the ``T x D`` grid, not a time series.

Paths -- run directories, and the data paths inside a run's own config -- are relative
to the repository root, so a caller starting anywhere else wants
``os.chdir(project_root())`` first.

The probe side has no confusion matrix, because a window carries several labels at once
and there is no single prediction to confuse. Per-label average precision against label
prevalence answers the question a confusion matrix would have been asked here: whether
the macro mAP comes from the whole vocabulary or from its head.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch as t

from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torchmetrics.classification import MultilabelAveragePrecision

from sometria.catalog import load_label_vocabulary
from sometria.models.window import standardize_tokens
from sometria.train import build


def project_root(start: str | Path | None = None) -> Path:
    """The directory holding ``pyproject.toml``, walking up from ``start`` (default: cwd).

    Run directories are named relative to the repository root, and so are the paths
    *inside* a run's config -- ``data/processed``, ``config/human.yaml``, the
    normalization artifact. A notebook starts in ``notebooks/``, so it has to move before
    any of that resolves; ``os.chdir(project_root())`` is the whole fix.
    """

    start = Path(start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").exists():
            return directory
    raise FileNotFoundError(f"no pyproject.toml at or above {start}")


def latest_checkpoint(run: str | Path) -> Path:
    """The newest version's best checkpoint, falling back to its ``last.ckpt``.

    ``ModelCheckpoint`` runs with ``save_top_k=1`` on the config's monitored metric, so
    the ``epoch=*.ckpt`` is the epoch ``scripts/results.py`` reports -- which is the model
    a figure should be drawn from. ``last.ckpt`` is the final epoch, and after an
    interrupted run it can be a half-written file.

    Versions are sorted numerically, so ``version_10`` comes after ``version_9``.
    """

    versions = sorted(
        Path(run).glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.removeprefix("version_")),
    )
    for version in reversed(versions):
        best = sorted((version / "checkpoints").glob("epoch=*.ckpt"))
        if best:
            return best[-1]
        if (version / "checkpoints" / "last.ckpt").exists():
            return version / "checkpoints" / "last.ckpt"
    raise FileNotFoundError(f"no checkpoint under {run}/lightning_logs/version_*/checkpoints")


def load_run(run: str | Path, device: str | None = None) -> tuple[t.nn.Module, object]:
    """``(model, datamodule)`` for a finished run, rebuilt from the config it saved.

    The resolved config beside the run is what produced it, so this reconstructs the
    experiment rather than a fresh guess at it -- including, for a probe, which
    pretraining checkpoint its backbone came from. Weights are then loaded over the top.

    ``device`` defaults to CUDA when there is one. Nothing here runs a Lightning
    ``Trainer``, so no accelerator moves the model on anyone's behalf, and a checkpoint
    loads to CPU by default -- forgetting this does not fail, it just runs the encoder on
    the CPU at a fraction of the speed.

    The datamodule is returned set up and with ``num_workers`` forced to 0: a notebook
    kernel and a forked loader do not survive each other.
    """

    device = device or ("cuda" if t.cuda.is_available() else "cpu")

    run = Path(run)
    config = OmegaConf.load(run / "config.yaml")
    OmegaConf.update(config, "dataloader.num_workers", 0)

    # A probe's own checkpoint holds its backbone -- the module is registered, only the
    # constructor argument was excluded from the hyperparameters -- so the pretraining
    # checkpoint it names is not needed here, and depending on it would make a figure
    # impossible whenever that file has been moved or deleted. The `encoder:` block still
    # has to describe the same architecture, which the strict load below verifies.
    if config.model.get("name") == "classifier":
        OmegaConf.update(config, "model.checkpoint", None)

    model, datamodule = build(config)
    state = t.load(latest_checkpoint(run), map_location="cpu")

    # Buffers may be absent from a checkpoint older than the buffer -- `loss_channel_index`
    # is one -- and they are derived from the hyperparameters the constructor already read,
    # so the rebuilt model has the right ones. Missing *parameters* are a different story
    # and still an error, which is what the assertion below separates.
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    buffers = {name for name, _ in model.named_buffers()}
    if set(missing) - buffers or unexpected:
        raise RuntimeError(
            f"{run} does not fit its own config: "
            f"missing {sorted(set(missing) - buffers)}, unexpected {sorted(unexpected)}"
        )
    model.eval()
    model.to(device)

    datamodule.setup("fit")
    return model, datamodule


def dof_names(datamodule) -> tuple[str, ...] | None:
    """DOF names off whatever dataset the datamodule built, or ``None`` if it has none."""

    for attribute in ("val", "val_dataset", "validation"):
        dataset = getattr(datamodule, attribute, None)
        representation = getattr(dataset, "representation", None)
        if representation is not None:
            return representation.dofs
    return None


# --------------------------------------------------------------------------- pretraining


def plot_channel_error(run: str | Path, split: str = "val") -> Figure:
    """Per-channel reconstruction error over epochs, straight off the run's metrics.csv.

    This is the plot the channels ablation argues from: ``acc`` and ``tau`` sit an order
    of magnitude above ``sin`` and ``cos`` and barely move, which is what "most of the
    gradient and almost none of the learnable signal" looks like.
    """

    metrics = sorted(Path(run).glob("lightning_logs/version_*/metrics.csv"))
    if not metrics:
        raise FileNotFoundError(f"no metrics.csv under {run}")

    frame = pl.read_csv(metrics[-1], infer_schema_length=None)
    columns = [c for c in frame.columns if c.startswith(f"{split}/mse/")]
    if not columns:
        raise ValueError(f"{metrics[-1]} logs no {split}/mse/* columns")

    figure, axes = plt.subplots(figsize=(6, 4))
    for column in columns:
        rows = frame.select("epoch", column).drop_nulls()
        axes.plot(rows["epoch"], rows[column], label=column.rsplit("/", 1)[-1], marker="o", ms=3)

    axes.set_xlabel("epoch")
    axes.set_ylabel(f"{split} MSE (standardized tokens)")
    axes.set_title(f"per-channel reconstruction error -- {Path(run).name}")
    axes.legend(title="channel")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


@t.no_grad()
def reconstruction_error_grid(model, batch: dict) -> t.Tensor:
    """``(time_patches, num_dofs)`` mean squared error, over the tokens that were masked.

    Flat token index ``i`` addresses time patch ``i // D`` of DOF ``i % D`` -- the layout
    :func:`~sometria.masking.patchify` produces -- so the grid is one reshape once the
    per-token errors are scattered back to where they came from.

    Positions no sample in the batch masked come back NaN rather than 0, so an unvisited
    cell reads as absent instead of as perfectly reconstructed.
    """

    device = next(model.parameters()).device
    prediction, target, window = model(batch["features"].to(device))
    if getattr(model, "norm_targets", False):
        target = standardize_tokens(target)

    error = (prediction - target).square().mean(dim=-1)   # (B, N) one number per token

    batch_size, length, _ = window.values.shape
    full = t.full((batch_size, length), float("nan"), device=device)
    full.scatter_(dim=1, index=window.mask.targets, src=error)

    grid = full.nanmean(dim=0).reshape(window.num_time_patches, model.spec.num_dofs)
    return grid.cpu()


def plot_reconstruction_grid(model, batch: dict, dofs: tuple[str, ...] | None = None) -> Figure:
    """The ``T x D`` error grid as a heatmap: which joints, at which times, stay wrong."""

    grid = reconstruction_error_grid(model, batch)

    figure, axes = plt.subplots(figsize=(7, 8))
    image = axes.imshow(grid.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")

    axes.set_xlabel("time patch")
    axes.set_ylabel("DOF")
    axes.set_title("masked-token reconstruction error")
    if dofs is not None:
        axes.set_yticks(range(len(dofs)), dofs, fontsize=5)
    figure.colorbar(image, ax=axes, label="MSE (standardized tokens)")
    figure.tight_layout()
    return figure


# ----------------------------------------------------------------------------- the probe


@t.no_grad()
def collect_predictions(model, loader, limit: int | None = None) -> tuple[t.Tensor, t.Tensor]:
    """``(logits, targets)`` over the loader -- collected once, plotted several ways.

    ``limit`` caps the number of batches, which is what makes this usable interactively;
    leave it None for a number that matches the run's own reported metric.
    """

    device = next(model.parameters()).device
    logits, targets = [], []
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        logits.append(model(batch["features"].to(device)).cpu())
        targets.append(batch["labels"].cpu())

    return t.cat(logits), t.cat(targets)


def per_label_ap(logits: t.Tensor, targets: t.Tensor) -> t.Tensor:
    """Average precision per label; NaN where the label never occurs in ``targets``.

    NaN rather than 0 for the same reason
    :class:`~sometria.downstream.metrics.WindowMeanAveragePrecision` drops those labels:
    a label with no positive has no AP, and scoring it 0 reports how much of the
    vocabulary the split happens to contain.
    """

    metric = MultilabelAveragePrecision(num_labels=targets.shape[1], average="none")
    scores = metric(logits, targets.int()).clone()
    scores[~(targets > 0).any(dim=0)] = float("nan")
    return scores


def label_names(root: str | Path, label_set: str) -> list[str]:
    """Vocabulary labels in ``label_index`` order, which is the order the model emits."""

    vocabulary = load_label_vocabulary(root, label_set).sort("label_index")
    return vocabulary["label"].to_list()


def plot_label_ap(
    logits: t.Tensor,
    targets: t.Tensor,
    names: list[str] | None = None,
) -> Figure:
    """Per-label AP as a bar chart, ordered by prevalence -- the macro mAP, decomposed.

    The dashed line is each label's prevalence, which is the AP a random ranker scores.
    A bar that fails to clear its own line is a label the probe reads no better than the
    base rate, whatever the macro average says.
    """

    scores = per_label_ap(logits, targets)
    prevalence = (targets > 0).float().mean(dim=0)

    order = t.argsort(prevalence, descending=True)
    order = order[~scores[order].isnan()]

    figure, axes = plt.subplots(figsize=(11, 4))
    positions = range(len(order))
    axes.bar(positions, scores[order], color="tab:blue", label="AP")
    axes.plot(positions, prevalence[order], "k--", lw=1, label="prevalence (chance AP)")

    axes.set_ylabel("average precision")
    axes.set_title(
        f"per-label AP, {len(order)} labels present  "
        f"(macro mAP {scores[order].mean():.4f})"
    )
    ticks = [names[i] for i in order.tolist()] if names else [str(i) for i in order.tolist()]
    axes.set_xticks(positions, ticks, rotation=90, fontsize=6)
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_prevalence_vs_ap(
    logits: t.Tensor,
    targets: t.Tensor,
    names: list[str] | None = None,
    annotate: int = 8,
) -> Figure:
    """AP against prevalence, one point per label, over the chance diagonal.

    The control for "did it only learn which actions are common": points hugging the
    diagonal are labels ranked no better than their base rate, and a cloud that rises
    with prevalence and nothing else is a probe reporting the label distribution.
    ``annotate`` names the highest-AP labels, since the interesting ones are usually few.
    """

    scores = per_label_ap(logits, targets)
    prevalence = (targets > 0).float().mean(dim=0)
    present = ~scores.isnan()

    figure, axes = plt.subplots(figsize=(6, 5))
    axes.scatter(prevalence[present], scores[present], s=18, alpha=0.75)

    limit = float(max(prevalence[present].max(), scores[present].max())) * 1.1
    axes.plot([0, limit], [0, limit], "k--", lw=1, label="chance (AP = prevalence)")

    if names is not None and annotate:
        ranked = t.argsort(t.nan_to_num(scores, nan=-1.0), descending=True)[:annotate]
        for index in ranked.tolist():
            axes.annotate(
                names[index],
                (prevalence[index], scores[index]),
                fontsize=7,
                xytext=(4, 2),
                textcoords="offset points",
            )

    axes.set_xlabel("prevalence")
    axes.set_ylabel("average precision")
    axes.set_title("is the probe doing more than the base rate?")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_f1_threshold(logits: t.Tensor, targets: t.Tensor) -> Figure:
    """Macro and micro F1 across decision thresholds, against the 0.5 the run logged.

    ``MultilabelF1Score`` defaults to a 0.5 threshold, which assumes a balanced problem.
    This one is not: the median window carries 2 labels out of 60, so a calibrated model
    puts almost every sigmoid below 0.5 and the logged F1 measures the threshold rather
    than the representation. The gap between the marked points is how much.

    mAP is unaffected -- it integrates over every threshold, which is why it is the
    headline metric here and F1 is a diagnostic.
    """

    from torchmetrics.classification import MultilabelF1Score

    labels = targets.shape[1]
    scores = logits.sigmoid()
    thresholds = t.linspace(0.02, 0.7, 35)

    curves = {}
    for average in ("macro", "micro"):
        curves[average] = t.tensor([
            MultilabelF1Score(num_labels=labels, threshold=float(threshold), average=average)(
                scores, targets.int()
            )
            for threshold in thresholds
        ])

    figure, axes = plt.subplots(figsize=(6, 4))
    for average, curve in curves.items():
        line, = axes.plot(thresholds, curve, label=f"{average} F1")
        peak = int(curve.argmax())
        axes.plot(thresholds[peak], curve[peak], "o", color=line.get_color())
        axes.annotate(
            f"{curve[peak]:.3f} @ {thresholds[peak]:.2f}",
            (thresholds[peak], curve[peak]),
            fontsize=8,
            xytext=(6, 4),
            textcoords="offset points",
        )

    axes.axvline(0.5, color="k", ls="--", lw=1, label="0.5 (the logged threshold)")
    axes.set_xlabel("decision threshold")
    axes.set_ylabel("F1")
    axes.set_title("F1 is a threshold measurement, not a model measurement")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_topk_recall(logits: t.Tensor, targets: t.Tensor, ks: tuple[int, ...] = (1, 2, 3, 5, 10, 20)) -> Figure:
    """Top-k recall against the best any ranker could do at that k.

    A window carrying more than ``k`` labels cannot score 1 at ``k``, so the reachable
    recall is ``sum(min(labels, k)) / sum(labels)`` and the raw number alone says nothing.
    Plotted together, the gap is the part that is the model's fault.
    """

    positives = targets.sum()
    counts = targets.sum(dim=1)

    recall, ceiling = [], []
    for k in ks:
        top = logits.topk(k, dim=1).indices
        recall.append(float(t.gather(targets, 1, top).sum() / positives))
        ceiling.append(float(t.minimum(counts, t.tensor(float(k))).sum() / positives))

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(ks, ceiling, "k--", marker="s", ms=4, label="reachable (windows carry >1 label)")
    axes.plot(ks, recall, marker="o", label="top-k recall")
    axes.fill_between(ks, recall, ceiling, alpha=0.15)

    for k, got, best in zip(ks, recall, ceiling):
        axes.annotate(f"{got / best:.0%}", (k, got), fontsize=7, xytext=(0, -12), textcoords="offset points", ha="center")

    axes.set_xlabel("k")
    axes.set_ylabel("recall of true positives")
    axes.set_title(f"top-k recall vs its ceiling ({counts.float().mean():.2f} labels per window)")
    axes.set_xscale("log")
    axes.set_xticks(ks, [str(k) for k in ks])
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_label_lift(logits: t.Tensor, targets: t.Tensor, names: list[str] | None = None) -> Figure:
    """AP divided by prevalence: how many times better than chance each label is ranked.

    The companion to :func:`plot_label_ap`, and the correction to reading it. AP is
    bounded by prevalence, so a rare label scores low however well it is ranked -- ``hop``
    at 0.5% prevalence cannot reach 0.2 no matter what. Dividing normalizes that out, and
    inverts the order: the head classes have the highest AP and the *lowest* lift.
    """

    scores = per_label_ap(logits, targets)
    prevalence = (targets > 0).float().mean(dim=0)
    lift = scores / prevalence

    order = t.argsort(t.nan_to_num(lift, nan=-1.0), descending=True)
    order = order[~lift[order].isnan()]

    figure, axes = plt.subplots(figsize=(11, 4))
    positions = range(len(order))
    axes.bar(positions, lift[order], color="tab:green")
    axes.axhline(1.0, color="k", ls="--", lw=1, label="chance")

    axes.set_ylabel("AP / prevalence")
    axes.set_yscale("log")
    axes.set_title(
        f"how many times better than chance  "
        f"(worst label {float(lift[order].min()):.1f}x, all {len(order)} above 1x)"
    )
    ticks = [names[i] for i in order.tolist()] if names else [str(i) for i in order.tolist()]
    axes.set_xticks(positions, ticks, rotation=90, fontsize=6)
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_probe_curves(runs: dict[str, str | Path], metric: str = "val/macro_map") -> Figure:
    """One metric over epochs for several runs on one axis -- straight off their CSVs.

    No checkpoint is loaded, so this is instant and works on any run that logged the
    column. Reads whether an ablation arm has converged or is still climbing, which is
    the question a table of single best numbers cannot answer.
    """

    figure, axes = plt.subplots(figsize=(7, 4))
    for label, run in runs.items():
        files = sorted(Path(run).glob("lightning_logs/version_*/metrics.csv"))
        if not files:
            raise FileNotFoundError(f"no metrics.csv under {run}")
        frame = pl.read_csv(files[-1], infer_schema_length=None)
        if metric not in frame.columns:
            raise ValueError(f"{files[-1]} logs no {metric!r}")
        rows = frame.select("epoch", metric).drop_nulls()
        axes.plot(rows["epoch"], rows[metric], marker="o", ms=3, label=label)

    axes.set_xlabel("epoch")
    axes.set_ylabel(metric)
    axes.set_title(metric)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_f1_at_k(logits: t.Tensor, targets: t.Tensor, ks: tuple[int, ...] = tuple(range(1, 11))) -> Figure:
    """Precision, recall and F1 when the prediction is the top ``k`` labels of a window.

    A different decision rule from :func:`plot_f1_threshold`, not a different setting of
    it: ``k`` fixes how *many* labels are predicted per window, a threshold fixes how
    *confident* a prediction has to be and lets the count vary. Averaged over windows
    here, where the threshold curves average over labels -- so the two are not comparable
    numbers, only comparable shapes.

    F1@k peaks where ``k`` meets the typical number of true labels, which is a fact about
    the annotation density rather than about the model. What *is* about the model is the
    distance to the dashed ceiling, and whether that distance grows with ``k``.

    Windows with no label are dropped: recall would divide by zero. They are worth
    remembering anyway -- top-k has to emit ``k`` guesses on a window that has no right
    answer, which a threshold rule can decline to do.
    """

    counts = targets.sum(dim=1).float()
    labelled = counts > 0
    dropped = int((~labelled).sum())
    logits, targets, counts = logits[labelled], targets[labelled].float(), counts[labelled]

    precision, recall, f1, ceiling = [], [], [], []
    for k in ks:
        hit = t.gather(targets, 1, logits.topk(k, dim=1).indices).sum(dim=1)
        p, r = float((hit / k).mean()), float((hit / counts).mean())
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)

        reachable = t.minimum(counts, t.tensor(float(k)))
        cp, cr = float((reachable / k).mean()), float((reachable / counts).mean())
        ceiling.append(2 * cp * cr / (cp + cr))

    figure, axes = plt.subplots(figsize=(6.5, 4))
    axes.plot(ks, precision, marker="o", ms=4, label="precision@k")
    axes.plot(ks, recall, marker="o", ms=4, label="recall@k")
    axes.plot(ks, f1, marker="o", ms=5, lw=2, color="tab:red", label="F1@k")
    axes.plot(ks, ceiling, "k--", lw=1, label="F1@k ceiling")

    peak = max(range(len(ks)), key=lambda i: f1[i])
    axes.annotate(
        f"F1@{ks[peak]} = {f1[peak]:.3f}  ({f1[peak] / ceiling[peak]:.0%} of ceiling)",
        (ks[peak], f1[peak]),
        fontsize=8,
        xytext=(8, -12),
        textcoords="offset points",
    )

    axes.set_xlabel("k (labels predicted per window)")
    axes.set_ylabel("score, averaged over windows")
    axes.set_title(
        f"top-k as a decision rule  "
        f"({float(counts.mean()):.2f} labels per window, {dropped} unlabelled dropped)"
    )
    axes.set_xticks(list(ks))
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def _self_check() -> None:
    """The two pieces with an index in them: the grid layout and the AP/prevalence align."""

    # Flat index t * D + d must land on row t, column d after the reshape.
    time_patches, dofs = 4, 3
    flat = t.arange(time_patches * dofs, dtype=t.float)
    grid = flat.reshape(time_patches, dofs)
    assert grid[2, 1] == 2 * dofs + 1

    # A label with no positive scores NaN, not 0, and does not enter the macro average.
    targets = t.zeros(8, 3, dtype=t.int)
    targets[:4, 0] = 1
    targets[:2, 1] = 1
    logits = t.zeros(8, 3)
    logits[:4, 0] = 5.0        # label 0 ranked perfectly
    scores = per_label_ap(logits, targets)
    assert scores[2].isnan(), scores
    assert abs(float(scores[0]) - 1.0) < 1e-6, scores

    print("ok")


if __name__ == "__main__":
    _self_check()
