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
def extract_features(
    backbone, pooler, norm, loader: Iterable[dict], device: str, passes: int = 1
) -> tuple[t.Tensor, t.Tensor, list[str]]:
    feats, labels, ids = [], [], []
    for _ in range(passes):
        for batch in loader:
            tokens = backbone.embed_tokens(batch["features"].to(device))
            feats.append(norm(pooler(tokens)).cpu())
            labels.append(batch["labels"].float())
            ids.extend(batch["sample_id"])
    return t.cat(feats), t.cat(labels), ids


def pool_by_sample(
    scores: t.Tensor, targets: t.Tensor, sample_ids: Sequence[str]
) -> tuple[t.Tensor, t.Tensor]:
    """Collapse per-window rows to one row per source motion.

    Off by default because it is wrong for BABEL: there the label varies *within* a take,
    which is the whole point of windowing it, and averaging over a take would score a
    question nobody asked. It is right whenever the label is a property of the take --
    CARE-PD's UPDRS gait score is one number per walk, and the release, the paper and the
    clinic all report per walk. Left per-window, a 40 s take votes ten times and a 4 s one
    votes once, so the metric is length-weighted by an artefact of the capture.

    Scores are averaged and targets unioned. Averaging in probability space rather than
    logit space is the ordinary ensemble mean; the union is a no-op when every window of a
    take carries the same label, and is the only defensible reading when they do not.
    """

    order: dict[str, int] = {}
    for sample_id in sample_ids:
        order.setdefault(sample_id, len(order))
    index = t.tensor([order[sample_id] for sample_id in sample_ids])

    counts = t.zeros(len(order)).index_add_(0, index, t.ones(len(index)))
    pooled = (
        t.zeros(len(order), scores.shape[1]).index_add_(0, index, scores) / counts[:, None]
    )
    unioned = t.zeros(len(order), targets.shape[1]).scatter_reduce_(
        0, index[:, None].expand_as(targets), targets.float(), "amax", include_self=False
    )
    return pooled, unioned


def _macro_f1(true: t.Tensor, pred: t.Tensor, labels: Sequence[int]) -> float:
    """Unweighted mean F1 over ``labels``, from hard single-label predictions.

    Written out rather than pulled from sklearn: it is six lines, and sklearn is not
    otherwise a dependency of this project.
    """

    scores = []
    for label in labels:
        tp = ((pred == label) & (true == label)).sum()
        fp = ((pred == label) & (true != label)).sum()
        fn = ((pred != label) & (true == label)).sum()
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else float(2 * tp / denominator))
    return sum(scores) / len(scores)


def vote_macro_f1(
    probabilities: t.Tensor,
    targets: t.Tensor,
    sample_ids: Sequence[str],
    subsets: Sequence[Sequence[int]],
) -> dict[str, float]:
    """CARE-PD's reported metric: per-take majority vote, macro F1 over a label subset.

    Two details are the paper's, not ours (Adeli et al. 2025, S4.1 and S4.3). The take's
    prediction is the *majority vote* of its windows' argmaxes, not the argmax of their mean
    probability -- so this is computed beside :func:`pool_by_sample` rather than from it.
    And a subset excludes a label from the *averaging only*: samples carrying it stay in the
    evaluation, so a true-3 take predicted as 2 still counts against class 2. That is what
    ``F1_0-2`` means in the paper, and dropping those samples instead would score an easier
    benchmark under the same name.

    Only defined for a single-label target. Returns ``{}`` on anything else -- BABEL's
    multi-hot windows have no argmax worth taking.
    """

    if targets.numel() == 0 or not bool((targets.sum(dim=1) == 1).all()):
        return {}

    order: dict[str, int] = {}
    for sample_id in sample_ids:
        order.setdefault(sample_id, len(order))
    index = t.tensor([order[sample_id] for sample_id in sample_ids])

    num_labels = targets.shape[1]
    # bincount per take over the windows' argmaxes; ties go to the lowest class index
    votes = t.zeros(len(order), num_labels).index_put_(
        (index, probabilities.argmax(dim=1)), t.ones(len(index)), accumulate=True
    )
    predicted = votes.argmax(dim=1)

    truth = t.zeros(len(order), num_labels).scatter_reduce_(
        0, index[:, None].expand_as(targets), targets.float(), "amax", include_self=False
    ).argmax(dim=1)

    # Named by the whole subset, not its endpoints: (0,1,2) and (0,2) share both endpoints
    # and would collide. "macro_f1_012" is the paper's F1_0-2.
    return {
        "macro_f1_" + "".join(str(label) for label in sorted(subset)): _macro_f1(
            truth, predicted, subset
        )
        for subset in subsets
    }


def score_predictions(scores: t.Tensor, targets: t.Tensor) -> dict[str, float]:
    num_labels = targets.shape[1]
    metrics = {
        "macro_map": WindowMeanAveragePrecision(num_labels=num_labels),
        "micro_map": MultilabelAveragePrecision(num_labels=num_labels, average="micro"),
        "macro_f1s": MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro"),
    }
    # A vocabulary can be smaller than the recall ladder: topk(k=5) over a 3-way head is a
    # RuntimeError, and topk(k=3) there is 1.0 by construction. Ask only what the head can
    # answer. BABEL-60 keeps all three, so its numbers are unchanged.
    metrics.update(
        {f"top_{k}_rec": MultilabelTopKRecall(top_k=k) for k in (1, 3, 5) if k < num_labels}
    )
    return {name: float(metric(scores, targets)) for name, metric in metrics.items()}


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

    def fit_features(
        self,
        feats: t.Tensor,
        labels: t.Tensor,
        max_iter: int = 200,
        tol: float = 1e-7,
        weight_decay: float | None = None,
    ) -> float:
        """Refit the head on already-extracted features. The reason cross-validation is cheap.

        The backbone is frozen, so every fold of a CV sees the same features and differs only
        in which rows it keeps. Extract once, call this per fold.
        """

        if weight_decay is not None:
            self.weight_decay = weight_decay
        self.head.reset_parameters()
        return self._fit_features(feats, labels, max_iter, tol)

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
        feats, labels, _ = extract_features(self.backbone, self.pooler, self.norm, train_loader, device, train_passes)
        return self._fit_features(feats.to(device), labels.to(device), max_iter, tol)

    @t.no_grad()
    def score_features(
        self,
        feats: t.Tensor,
        labels: t.Tensor,
        sample_ids: list[str],
        device: str,
        by_sample: bool,
        f1_subsets: Sequence[Sequence[int]] = (),
    ) -> dict[str, float]:
        """Head, optional take-pooling, metrics. Shared by :meth:`evaluate` and :meth:`sweep`.

        When pooling, the two metric families use the aggregation each needs and are reported
        side by side: mAP ranks, so it pools mean probabilities; the paper's macro F1 needs a
        hard label, so it pools by majority vote. Voting first would destroy the ranking mAP
        scores, and averaging first would not be the published protocol.
        """

        logits = self.head(feats.to(device)).cpu()
        if not by_sample:
            return score_predictions(logits, labels.int())

        probabilities = logits.sigmoid()
        scores, targets = pool_by_sample(probabilities, labels, sample_ids)
        return score_predictions(scores, targets.int()) | vote_macro_f1(
            probabilities, labels, sample_ids, f1_subsets
        )

    @t.no_grad()
    def evaluate(
        self,
        loader: Iterable[dict],
        device: str = "cpu",
        by_sample: bool = False,
        f1_subsets: Sequence[Sequence[int]] = (),
    ) -> dict[str, float]:
        """The same numbers :class:`MotionLinearClassifier` logs under ``val/*``."""

        feats, labels, sample_ids = extract_features(self.backbone, self.pooler, self.norm, loader, device)
        return self.score_features(feats, labels, sample_ids, device, by_sample, f1_subsets)

    def sweep(
        self,
        train_loader: Iterable[dict],
        val_loader: Iterable[dict],
        weight_decays: Sequence[float] = DEFAULT_WEIGHT_DECAYS,
        device: str = "cpu",
        max_iter: int = 200,
        tol: float = 1e-7,
        train_passes: int = DEFAULT_TRAIN_PASSES,
        by_sample: bool = False,
        f1_subsets: Sequence[Sequence[int]] = (),
    ) -> tuple[float, dict[str, float], dict[float, dict[str, float]]]:
        """Refit the head for each candidate ``weight_decay``, picking by ``macro_map``.

        Returns ``(best_weight_decay, best_metrics, all_metrics)``; leaves ``self.head``
        and ``self.weight_decay`` set to the winner, same as calling :meth:`fit` with it
        directly. ``train_passes`` re-draws the training set's random crop that many
        times, same reasoning as in :meth:`fit`.

        ``by_sample`` selects the winner on take-pooled validation metrics as well as
        reporting them, which is the point: a weight decay tuned on window-level scores is
        not the one that is best per take.
        """

        self.to(device)
        train_feats, train_labels, _ = extract_features(self.backbone, self.pooler, self.norm, train_loader, device, train_passes)
        val_feats, val_labels, val_ids = extract_features(self.backbone, self.pooler, self.norm, val_loader, device)
        train_feats, train_labels = train_feats.to(device), train_labels.to(device)

        results: dict[float, dict[str, float]] = {}
        best_wd, best_metrics, best_state = None, None, None
        for wd in weight_decays:
            self.head.reset_parameters()
            self.weight_decay = wd
            self._fit_features(train_feats, train_labels, max_iter, tol)

            metrics = self.score_features(val_feats, val_labels, val_ids, device, by_sample, f1_subsets)
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

    def _batches(n_batches: int, batch_size: int = 32, windows_per_take: int = 4) -> list[dict]:
        t.manual_seed(0)
        return [
            {
                "features": t.randn(batch_size, 6, 8),
                "labels": (t.rand(batch_size, 6) > 0.7).float(),
                # contiguous runs, the way tiles of one motion arrive from LabelledWindows
                "sample_id": [
                    f"take{(b * batch_size + i) // windows_per_take}" for i in range(batch_size)
                ],
            }
            for b in range(n_batches)
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

    # take-pooling: four windows per take collapse to one row, scores averaged and
    # targets unioned, and the rows stay in first-seen order
    scores = t.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    targets = t.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    pooled, unioned = pool_by_sample(scores, targets, ["a", "a", "b"])
    assert t.allclose(pooled, t.tensor([[0.5, 0.5], [0.5, 0.5]])), pooled
    assert t.equal(unioned, t.tensor([[1.0, 0.0], [0.0, 1.0]])), unioned

    windowed = clf.evaluate(_batches(5))
    pooled_metrics = clf.evaluate(_batches(5), by_sample=True)
    assert set(windowed) == set(pooled_metrics)
    assert windowed != pooled_metrics, "pooling 4 windows per take changed nothing"
    print("ok (pooling):", pooled_metrics)

    # majority vote, and what excluding a class from the average does.
    #   take a: windows vote 0,0,1 -> 0, true 0   correct
    #   take b: votes 2            -> 2, true 2   correct
    #   take c: votes 0            -> 0, true 1   wrong, and the only class-1 sample
    probabilities = t.tensor(
        [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0], [0.7, 0.3, 0.0]]
    )
    one_hot = t.tensor([[1.0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    ids = ["a", "a", "a", "b", "c"]
    f1 = vote_macro_f1(probabilities, one_hot, ids, [(0, 1, 2), (0, 2)])
    # class 0: 1 tp 1 fp -> 2/3.  class 1: never predicted -> 0.  class 2: perfect -> 1.
    assert abs(f1["macro_f1_012"] - (2 / 3 + 0.0 + 1.0) / 3) < 1e-6, f1
    # Excluding class 1 from the *average* keeps take c in as a false positive for class 0,
    # so class 0 still scores 2/3. That is the paper's F1_0-2, not an easier benchmark:
    # dropping the samples too would give class 0 a clean 1.0 and inflate the number.
    assert abs(f1["macro_f1_02"] - (2 / 3 + 1.0) / 2) < 1e-6, f1
    print("ok (vote macro f1):", f1)

    # multi-hot targets have no argmax worth taking, so the metric declines to answer
    assert vote_macro_f1(probabilities, t.ones(5, 3), ids, [(0, 1)]) == {}

    # a vocabulary smaller than the recall ladder must not ask for topk(5) over 3 columns
    narrow = score_predictions(t.randn(16, 3), (t.rand(16, 3) > 0.5).int())
    assert set(narrow) == {"macro_map", "micro_map", "macro_f1s", "top_1_rec"}, narrow
    print("ok (narrow vocabulary):", narrow)
