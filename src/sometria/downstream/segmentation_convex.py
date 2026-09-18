"""Convex action segmentation: :mod:`classifier_convex`'s fit, one axis over.

:class:`~sometria.downstream.segmentation.MotionSegmenter` already argues why this task
exists and why its pooling reduces DOFs only. This module is that head made convex, for
the reason :mod:`sometria.downstream.classifier_convex` exists at all: the cells this
ranks differ by 0.003-0.010 macro mAP, and a head that also varies with lr, warmup and
epoch budget cannot resolve a gap that size. The AdamW/attentive ``MotionSegmenter``
stays available as the optional ceiling run; it is not the ranking instrument.

What changes against ``classifier_convex``: a row is a ``(window, patch)`` pair rather
than a window. The backbone's ``(B, TP * D, d_model)`` grid is folded back to
``(B * TP, D, d_model)`` and mean-pooled over the DOFs, so the time axis survives as rows
-- a window becomes ``30 x d_model``, never one vector. The head is still one ``Linear``
over a parameter-free pooling, which is still per-label logistic regression, so the whole
L-BFGS + weight-decay-sweep rig transfers unchanged.

Pooling is ``mean`` and is not a knob. ``moments`` was considered and rejected: it is the
*window* task's escape hatch from a near-rank-1 mean over ``TP * D`` tokens, and the mean
here is over 43 DOFs of one 133 ms patch, which is not the reduction that drowns a
signal. The attentive poolers are rejected for the parent module's reason -- they carry
weights and the fit stops being convex -- and ``attentive_factorized`` additionally pools
the axis this task predicts along.

Rows here are ``passes x windows x 30 patches``, so memory is a live enough risk to check:
:func:`check_feature_memory` fails before extraction rather than during the solve, because a
full-batch L-BFGS that OOMs mid-fit surfaces as an allocator error a long way from the
decision that caused it. It is not, however, what sets :data:`DEFAULT_TRAIN_PASSES`.
"""

import psutil
from collections.abc import Iterable, Sequence

import torch as t

from sometria.downstream.classifier_convex import (
    DEFAULT_WEIGHT_DECAYS,
    MotionConvexClassifier,
    score_predictions,
)
from sometria.downstream.segmentation import (
    BOUNDARY_THRESHOLDS,
    best_boundary_f1,
    boundary_f1,
)

#: Crop re-draws before fitting, matching ``classifier_convex.DEFAULT_TRAIN_PASSES``.
#:
#: This was first set to 2 on a memory argument that turned out to be wrong: it assumed ~9k
#: usable training windows, which forgot the ``source_datasets: ["AMASS"]`` filter that both
#: probe configs also apply. Measured on the config as it stands, ``babel_official/train``
#: is 12,994 samples -> 10,015 at >= 240 frames -> 4,903 after the AMASS filter -> 3,488
#: carrying frame-level ``act_cat`` -> 3,282 usable windows. So 2 passes is 196,920 rows and
#: 0.32 GiB, and 20 is 1.88 GiB. Memory does not decide this.
#:
#: What decides it is that there is no saturation point. Passes buy distinct random crops,
#: and the median eligible sample has 278 distinct window offsets -- only 3.6% of samples
#: have 20 or fewer. 2 passes touches 0.7% of that crop space and 20 touches 7.0%: both
#: small, and the curve between them smooth and monotone, so there is no knee a sweep could
#: find. The tie goes to comparability -- this table is read beside the window probe's, and
#: a different passes count would be a second axis of difference between them.
#:
#: The asymmetry worth remembering: one window already yields 30 supervised rows here
#: against the window task's 1, so matching the passes number does not match the
#: supervision. This is a default, not a derivation.
DEFAULT_TRAIN_PASSES = 20

#: Fraction of free memory the feature matrix may claim. The rest is the backbone's own
#: activations during extraction, the transient copy ``t.cat`` makes, and whatever else
#: shares the device.
MEMORY_HEADROOM = 0.8


def feature_matrix_bytes(rows: int, d_model: int, num_labels: int) -> int:
    """Peak float32 bytes one full-batch L-BFGS solve holds for ``rows`` rows.

    The features, plus three logit-sized tensors: ``head(feats)``, its gradient, and the
    sigmoid ``BCEWithLogitsLoss`` computes on the way. Everything else L-BFGS keeps is
    parameter-sized (60 x 256 per history entry) and rounds to nothing beside these.
    """

    return rows * (d_model + 3 * num_labels) * 4


def _available_bytes(device: str) -> int:
    if device.startswith("cuda") and t.cuda.is_available():
        return int(t.cuda.mem_get_info(t.device(device))[0])
    # psutil's `available`, not `free`: on Linux the latter is MemFree, which excludes
    # reclaimable page cache and on a warm box under-reports by an order of magnitude. This
    # guard read 0.40 GiB on a machine with 40 GiB obtainable and refused a 3.74 GiB
    # extraction -- "wrong in the safe direction" is not safe when it blocks a fit that
    # comfortably fits. psutil is already a dependency via Lightning.
    return int(psutil.virtual_memory().available)


def check_feature_memory(rows: int, d_model: int, num_labels: int, device: str) -> str:
    """Raise before extraction if the feature matrix cannot fit; else describe its size.

    Loud and early on purpose. The alternative is discovering it 500 batches into an
    extraction pass, or inside L-BFGS's line search where the traceback names a matmul and
    not the ``train_passes`` that caused it.

    Two demands, not one. The solve holds :func:`feature_matrix_bytes` wherever it runs,
    but extraction always accumulates on the *host* -- a list of per-batch tensors plus the
    copy ``t.cat`` makes -- even when the solve then runs on a GPU. On a box with a large
    card and a busy host it is the host that gives out first, and that failure would
    otherwise look like nothing to do with this.
    """

    solve = feature_matrix_bytes(rows, d_model, num_labels)
    host = 2 * rows * d_model * 4
    demands = {"cpu": max(host, solve) if device == "cpu" else host, device: solve}

    size = f"{rows:,} rows x {d_model} -> {solve / 2**30:.2f} GiB solve, {host / 2**30:.2f} GiB host"
    for where, needed in demands.items():
        free = _available_bytes(where)
        if needed > MEMORY_HEADROOM * free:
            raise MemoryError(
                f"segmentation feature matrix will not fit: {size}, but {where} has "
                f"{free / 2**30:.2f} GiB free and needs {needed / 2**30:.2f} GiB "
                f"({MEMORY_HEADROOM:.0%} of free is usable). Lower training.train_passes "
                f"(default {DEFAULT_TRAIN_PASSES}), or probe fewer cells at once."
            )
    return size


@t.no_grad()
def extract_patch_features(
    backbone,
    pooler,
    norm,
    loader: Iterable[dict],
    device: str,
    num_dofs: int,
    passes: int = 1,
) -> tuple[t.Tensor, t.Tensor]:
    """``(N, TP, d_model)`` features and ``(N, TP, num_labels)`` targets.

    The patch axis is kept rather than flattened here because
    :func:`~sometria.downstream.segmentation.boundary_f1` has to know which rows came from
    the same window and in what order. The fit flattens it; the scoring does not.

    No ``sample_id``: ``classifier_convex.pool_by_sample`` collapses the windows of one
    take to one row, which is the opposite of the axis this task exists to keep.
    """

    feats, labels = [], []
    for _ in range(passes):
        for batch in loader:
            tokens = backbone.embed_tokens(batch["features"].to(device))

            # (B, TP * D, C) -> (B * TP, D, C). The grid flattens as `t * D + d`
            # (PositionalEncoding.get_flat), so this reshape inverts exactly that and the
            # DOF axis is the one the pooler reduces -- same fold as MotionSegmenter.forward.
            batch_size, length, width = tokens.shape
            patches = length // num_dofs
            pooled = norm(pooler(tokens.reshape(batch_size * patches, num_dofs, width)))

            if not feats:
                # Checked once the first batch has told us the patch count and the label
                # width, and before anything has been accumulated. `len(loader)` is batch
                # count for a DataLoader; the last batch may be short, so this over-counts
                # by at most one batch, which is the direction a guard should err in.
                print("  " + check_feature_memory(
                    passes * len(loader) * batch_size * patches,  # type: ignore[arg-type]
                    width,
                    batch["labels"].shape[-1],
                    device,
                ))

            feats.append(pooled.reshape(batch_size, patches, width).cpu())
            labels.append(batch["labels"].float())

    return t.cat(feats), t.cat(labels)


def score_patch_predictions(logits: t.Tensor, targets: t.Tensor) -> dict[str, float]:
    """The window probe's metric set over ``(window, patch)`` rows, plus the boundary gate.

    Metric names are ``scripts/moments_baseline.py``'s segmentation names exactly, and that
    is the point: the dumb baseline has to land in the same table as the probe, and a
    renamed column is a baseline nobody compares against.

    ``macro_map`` is the ranking number. ``boundary_f1s`` is a gate reported beside it: a
    prediction constant across the window can still score on mAP and earns exactly zero
    here, so a cell that beats the moments on mAP and not on boundary F1 has not shown
    temporal structure.
    """

    scores = score_predictions(logits.flatten(end_dim=1), targets.flatten(end_dim=1).int())
    scores["boundary_f1s"] = float(boundary_f1(logits, targets))
    scores["boundary_best_f1s"] = float(best_boundary_f1(logits, targets))
    # best_boundary_f1 returns the maximum and drops the threshold that produced it, while
    # its own docstring asks for that threshold to be reported. Recovered here rather than
    # by changing a function the AdamW head also logs through.
    scores["boundary_best_threshold"] = max(
        BOUNDARY_THRESHOLDS, key=lambda cut: float(boundary_f1(logits, targets, cut))
    )
    return scores


class MotionConvexSegmenter(MotionConvexClassifier):
    """Frozen backbone + DOF-mean pooling + one ``Linear`` per time patch, fit by L-BFGS.

    Inherits the head, the L-BFGS solve and ``fit_features`` from
    :class:`~sometria.downstream.classifier_convex.MotionConvexClassifier`; what it
    replaces is everything that assumed a row is a window.
    """

    def __init__(self, backbone, num_labels: int, weight_decay: float = 1e-2) -> None:
        # `pool` is not exposed. mean is the decision (see the module docstring), and the
        # parent's other convex poolers change out_dim without changing which axis dies.
        super().__init__(backbone, num_labels, pool="mean", weight_decay=weight_decay)
        self.num_dofs = backbone.spec.num_dofs

    def forward(self, features: t.Tensor) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, TP, num_labels)``."""

        with t.no_grad():
            tokens = self.backbone.embed_tokens(features)

        batch, length, width = tokens.shape
        patches = length // self.num_dofs
        pooled = self.norm(self.pooler(tokens.reshape(batch * patches, self.num_dofs, width)))
        return self.head(pooled).reshape(batch, patches, -1)

    def _extract(self, loader: Iterable[dict], device: str, passes: int = 1):
        return extract_patch_features(
            self.backbone, self.pooler, self.norm, loader, device, self.num_dofs, passes
        )

    @t.no_grad()
    def score_features(self, feats: t.Tensor, labels: t.Tensor, device: str) -> dict[str, float]:  # type: ignore[override]
        """Head, then metrics. Narrower than the parent's signature by three arguments:
        take-pooling and its vote-F1 have no meaning for a target that varies inside one
        window, let alone inside one take."""

        return score_patch_predictions(self.head(feats.to(device)).cpu(), labels)

    def fit(  # type: ignore[override]
        self,
        train_loader: Iterable[dict],
        device: str = "cpu",
        max_iter: int = 200,
        tol: float = 1e-7,
        train_passes: int = DEFAULT_TRAIN_PASSES,
    ) -> float:
        self.to(device)
        feats, labels = self._extract(train_loader, device, train_passes)
        return self._fit_features(
            feats.flatten(end_dim=1).to(device), labels.flatten(end_dim=1).to(device), max_iter, tol
        )

    @t.no_grad()
    def evaluate(self, loader: Iterable[dict], device: str = "cpu") -> dict[str, float]:  # type: ignore[override]
        feats, labels = self._extract(loader, device)
        return self.score_features(feats, labels, device)

    def sweep(  # type: ignore[override]
        self,
        train_loader: Iterable[dict],
        val_loader: Iterable[dict],
        weight_decays: Sequence[float] = DEFAULT_WEIGHT_DECAYS,
        device: str = "cpu",
        max_iter: int = 200,
        tol: float = 1e-7,
        train_passes: int = DEFAULT_TRAIN_PASSES,
    ) -> tuple[float, dict[str, float], dict[float, dict[str, float]]]:
        """Refit the head per candidate ``weight_decay``, picking by per-patch ``macro_map``.

        Selection is on mAP and never on boundary F1. Boundary F1 moves with where the
        decision threshold falls as much as with the model, and it is the gate this table
        reports rather than the thing it optimizes -- selecting the regularizer on the gate
        would leave nothing independent holding the ranking honest.

        Same return shape as the parent's sweep, so the runner and ``gen.py`` read both
        trees the same way.
        """

        self.to(device)
        print("train features")
        train_feats, train_labels = self._extract(train_loader, device, train_passes)
        print("val features")
        val_feats, val_labels = self._extract(val_loader, device)

        train_feats = train_feats.flatten(end_dim=1).to(device)
        train_labels = train_labels.flatten(end_dim=1).to(device)

        results: dict[float, dict[str, float]] = {}
        best_wd, best_metrics, best_state = None, None, None
        for wd in weight_decays:
            self.head.reset_parameters()
            self.weight_decay = wd
            self._fit_features(train_feats, train_labels, max_iter, tol)

            metrics = self.score_features(val_feats, val_labels, device)
            results[wd] = metrics

            if best_metrics is None or metrics["macro_map"] > best_metrics["macro_map"]:
                best_wd, best_metrics = wd, metrics
                best_state = {k: v.clone() for k, v in self.head.state_dict().items()}

        assert best_wd is not None and best_state is not None  # weight_decays is never empty
        self.head.load_state_dict(best_state)
        self.weight_decay = best_wd
        return best_wd, best_metrics, results
