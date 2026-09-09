"""Convex alternative to :class:`~sometria.downstream.classifier.MotionLinearClassifier`.

``mean``/``mean_max`` pooling add no parameters, and BCE over one ``Linear`` head is then
a per-label logistic regression -- convex, one global optimum, nothing for AdamW's
lr/warmup/epoch-budget to get wrong. So fit it with L-BFGS to convergence on features
extracted once from the frozen backbone, instead of training it like a neural net.

The attentive poolers are excluded here on purpose: they carry their own weights, which
makes the pooling step itself non-convex, and this class is only useful as long as that
is not true.

The one hyperparameter left is ``weight_decay``, and unlike AdamW's lr/warmup/epochs it
is cheap to sweep: the backbone forward pass dominates cost and does not depend on it, so
:meth:`MotionConvexClassifier.sweep` extracts features once and refits only the head per
candidate.
"""

from collections.abc import Iterable, Sequence

import torch as t
import torch.nn as nn

from sometria.downstream.metrics import MultilabelTopKRecall, WindowMeanAveragePrecision
from sometria.downstream.pooling import get_pooler
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

_CONVEX_POOLS = {"mean", "mean_max"}

# Log-spaced from 1e-1 (visibly over-regularized on the checkpoint matrix, macro mAP
# ~0.20) down to 1e-6. Extends past where the matrix run's optimum landed (1e-5, still
# improving at that grid's edge) so the default sweep finds the turnover instead of
# quietly bottoming out at the last candidate.
DEFAULT_WEIGHT_DECAYS: tuple[float, ...] = (
    1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6,
)

# ponytail: LabelledWindows(tiles=False) draws one random crop per sample per epoch, so
# AdamW's 100-epoch probe effectively trains on ~100 crops/sample; one extraction pass
# gives the convex fit exactly one. This is the augmentation AdamW gets for free that a
# single pass does not -- raise it if the gap to the AdamW probe is still crop-count, not
# regularization.
DEFAULT_TRAIN_PASSES = 20


@t.no_grad()
def _extract(backbone, pooler, norm, loader: Iterable[dict], device: str, passes: int = 1) -> tuple[t.Tensor, t.Tensor]:
    feats, labels = [], []
    for _ in range(passes):
        for batch in loader:
            tokens = backbone.embed_tokens(batch["features"].to(device))
            feats.append(norm(pooler(tokens)).cpu())
            labels.append(batch["labels"].float())
    return t.cat(feats), t.cat(labels)


def _score(logits: t.Tensor, targets: t.Tensor) -> dict[str, float]:
    num_labels = targets.shape[1]
    metrics = {
        "macro_map": WindowMeanAveragePrecision(num_labels=num_labels),
        "micro_map": MultilabelAveragePrecision(num_labels=num_labels, average="micro"),
        "macro_f1s": MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro"),
        "top_1_rec": MultilabelTopKRecall(top_k=1),
        "top_3_rec": MultilabelTopKRecall(top_k=3),
        "top_5_rec": MultilabelTopKRecall(top_k=5),
    }
    return {name: float(metric(logits, targets)) for name, metric in metrics.items()}


class MotionConvexClassifier(nn.Module):
    """Frozen backbone + parameter-free pooler + one ``Linear`` head, fit by L-BFGS."""

    def __init__(
        self,
        backbone,
        num_labels: int,
        pool: str = "mean",
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        if pool not in _CONVEX_POOLS:
            raise ValueError(f"convex head needs a parameter-free pooler, got '{pool}' (use one of {_CONVEX_POOLS})")

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        spec = backbone.spec
        self.weight_decay = weight_decay
        self.pooler = get_pooler(pool, spec.d_model, spec.num_dofs)
        self.norm = nn.LayerNorm(self.pooler.out_dim, elementwise_affine=False, eps=1e-6)  # type: ignore
        self.head = nn.Linear(self.pooler.out_dim, num_labels)  # type: ignore

    def forward(self, features: t.Tensor) -> t.Tensor:
        with t.no_grad():
            tokens = self.backbone.embed_tokens(features)
        return self.head(self.norm(self.pooler(tokens)))

    def _fit_features(self, feats: t.Tensor, labels: t.Tensor, max_iter: int, tol: float) -> float:
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = t.optim.LBFGS(
            self.head.parameters(),
            lr=1.0,
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure() -> t.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(self.head(feats), labels)
            if self.weight_decay:
                loss = loss + self.weight_decay * self.head.weight.pow(2).sum()
            loss.backward()
            return loss

        return float(optimizer.step(closure).detach())  # type: ignore

    def fit(
        self,
        train_loader: Iterable[dict],
        device: str = "cpu",
        max_iter: int = 200,
        tol: float = 1e-7,
        train_passes: int = DEFAULT_TRAIN_PASSES,
    ) -> float:
        """Solve the per-label logistic regression to convergence.

        Full-batch: pooled features are a handful of floats per window, so unlike the
        raw token grid the whole split fits in memory, and there is no stochasticity
        left worth exploiting once the surface is convex. ``train_passes`` re-draws the
        training set's random crop that many times first -- see ``DEFAULT_TRAIN_PASSES``.
        """

        self.to(device)
        feats, labels = _extract(self.backbone, self.pooler, self.norm, train_loader, device, train_passes)
        return self._fit_features(feats.to(device), labels.to(device), max_iter, tol)

    @t.no_grad()
    def evaluate(self, loader: Iterable[dict], device: str = "cpu") -> dict[str, float]:
        """Same six numbers :class:`MotionLinearClassifier` logs under ``val/*``."""

        feats, labels = _extract(self.backbone, self.pooler, self.norm, loader, device)
        logits = self.head(feats.to(device)).cpu()
        return _score(logits, labels.int())

    def sweep(
        self,
        train_loader: Iterable[dict],
        val_loader: Iterable[dict],
        weight_decays: Sequence[float] = DEFAULT_WEIGHT_DECAYS,
        device: str = "cpu",
        max_iter: int = 200,
        tol: float = 1e-7,
        train_passes: int = DEFAULT_TRAIN_PASSES,
    ) -> tuple[float, dict[str, float], dict[float, dict[str, float]]]:
        """Refit the head for each candidate ``weight_decay``, picking by ``macro_map``.

        Returns ``(best_weight_decay, best_metrics, all_metrics)``; leaves ``self.head``
        and ``self.weight_decay`` set to the winner, same as calling :meth:`fit` with it
        directly. ``train_passes`` re-draws the training set's random crop that many
        times, same reasoning as in :meth:`fit`.
        """

        self.to(device)
        train_feats, train_labels = _extract(self.backbone, self.pooler, self.norm, train_loader, device, train_passes)
        val_feats, val_labels = _extract(self.backbone, self.pooler, self.norm, val_loader, device)
        train_feats, train_labels = train_feats.to(device), train_labels.to(device)
        val_targets = val_labels.int()

        results: dict[float, dict[str, float]] = {}
        best_wd, best_metrics, best_state = None, None, None
        for wd in weight_decays:
            self.head.reset_parameters()
            self.weight_decay = wd
            self._fit_features(train_feats, train_labels, max_iter, tol)

            with t.no_grad():
                logits = self.head(val_feats.to(device)).cpu()
            metrics = _score(logits, val_targets)
            results[wd] = metrics

            if best_metrics is None or metrics["macro_map"] > best_metrics["macro_map"]:
                best_wd, best_metrics = wd, metrics
                best_state = {k: v.clone() for k, v in self.head.state_dict().items()}

        assert best_wd is not None and best_state is not None  # weight_decays is never empty
        self.head.load_state_dict(best_state)
        self.weight_decay = best_wd
        return best_wd, best_metrics, results


if __name__ == "__main__":
    # ponytail: shape/convergence smoke test, not a metrics test -- synthetic labels are
    # not separable enough to assert an AP threshold, only that the loss goes down.
    class _IdentityBackbone(nn.Module):
        spec = type("Spec", (), {"d_model": 8, "num_dofs": 2})()

        def embed_tokens(self, features: t.Tensor) -> t.Tensor:
            return features

    def _batches(n_batches: int, batch_size: int = 32) -> list[dict]:
        t.manual_seed(0)
        return [
            {"features": t.randn(batch_size, 6, 8), "labels": (t.rand(batch_size, 6) > 0.7).float()}
            for _ in range(n_batches)
        ]

    clf = MotionConvexClassifier(_IdentityBackbone(), num_labels=6, pool="mean")
    train_batches = _batches(20)
    bce = nn.BCEWithLogitsLoss()
    feats = clf.norm(clf.pooler(clf.backbone.embed_tokens(t.cat([b["features"] for b in train_batches]))))
    targets = t.cat([b["labels"] for b in train_batches])

    with t.no_grad():
        before = bce(clf.head(feats), targets)
    clf.fit(train_batches)
    with t.no_grad():
        after = bce(clf.head(feats), targets)
    assert after < before, f"L-BFGS did not improve loss: {before} -> {after}"

    metrics = clf.evaluate(_batches(5))
    assert set(metrics) == {"macro_map", "micro_map", "macro_f1s", "top_1_rec", "top_3_rec", "top_5_rec"}
    print("ok (fit/evaluate):", metrics)

    best_wd, best_metrics, results = clf.sweep(train_batches, _batches(5), weight_decays=(1e-1, 1e-3, 1e-5))
    assert set(results) == {1e-1, 1e-3, 1e-5}
    assert best_metrics == results[best_wd]
    print("ok (sweep):", best_wd, best_metrics)
