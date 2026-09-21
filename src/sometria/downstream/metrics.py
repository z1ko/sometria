"""Metrics over a multi-label window benchmark: mAP, top-k recall, BABEL's Top-1-norm.

``torchmetrics`` computes per-label average precision; the only thing worth writing here
is what to do with a label that never appears. It scores 0, and with a 150-way
frequency-ordered vocabulary over a validation split most of the tail is absent -- macro
averaging would then report the fraction of the vocabulary that happens to occur rather
than how well the model does on what it saw. Labels with no positive are left out.
"""

import torch as t

from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics import Metric


class WindowMeanAveragePrecision(MultilabelAveragePrecision):
    """Per-label AP averaged over the labels that actually occur in the targets."""

    def __init__(self, num_labels: int) -> None:
        super().__init__(num_labels=num_labels, average="none")
        self.add_state(
            "label_presence",
            default=t.zeros(num_labels, dtype=t.bool),
            # "any" is not one of the reductions torchmetrics accepts; max over a bool is
            # the same thing, and a label seen on any rank has to count on all of them.
            dist_reduce_fx="max",
        )

    def update(self, preds: t.Tensor, target: t.Tensor) -> None:
        super().update(preds, target)
        batch_presence = (target > 0).any(dim=0)
        self.label_presence = self.label_presence | batch_presence

    def compute(self) -> t.Tensor:
        per_label = super().compute()
        if not self.label_presence.any():
            return t.tensor(0.0, device=per_label.device)

        # Filter out labels that never appeared in targets (prevents NaNs from zero-support labels)
        valid_ap = per_label[self.label_presence]
        # Mask out any residual NaNs just in case a label had zero positive samples
        valid_ap = valid_ap[~valid_ap.isnan()]

        if valid_ap.numel() == 0:
            return t.tensor(0.0, device=per_label.device)

        return valid_ap.mean()

class MultilabelTop1Norm(Metric):
    """BABEL's ``Top-1-norm``: the unweighted mean, over classes, of the argmax's recall.

    Their implementation accumulates a confusion matrix over single-label samples and
    reports ``mean(confusion.diag() / confusion.sum(1))`` (``train_test.py``). A segment
    carrying *k* categories reaches that loop as *k* duplicated samples sharing one
    prediction, so the same number falls out of a multi-hot target directly: a class's
    support is how many targets carry it, and its hit count is how often the argmax
    *was* it. Duplication changes neither, which is why nothing here duplicates rows.

    Classes with no support are left out, as in :class:`WindowMeanAveragePrecision` and
    for a sharper reason -- theirs divides by a zero row sum and lets the resulting
    ``nan`` swallow the whole metric. BABEL-60 is built so every class occurs; a label
    subset or a small split is not.
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.add_state("hits", default=t.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("support", default=t.zeros(num_labels), dist_reduce_fx="sum")

    def update(self, preds: t.Tensor, target: t.Tensor) -> None:
        positive = target > 0
        self.support += positive.sum(dim=0)

        predicted = preds.argmax(dim=-1)
        # A hit is "the argmax is one of this row's labels", credited to that class only.
        hit = positive.gather(1, predicted[:, None]).squeeze(1) > 0
        self.hits.index_add_(
            0, predicted[hit], t.ones(int(hit.sum()), device=self.hits.device)
        )

    def compute(self) -> t.Tensor:
        seen = self.support > 0
        if not seen.any():
            return t.tensor(0.0, device=self.hits.device)
        return (self.hits[seen] / self.support[seen]).mean()


class MultilabelTopKRecall(Metric):
    """What fraction of the ground-truth positives land in the top ``k`` logits.

    Recall, not accuracy: a window carrying more than ``k`` labels cannot score 1 at
    ``k``, so top-1 over a multi-hot target with ~3 labels a window tops out near 0.33.
    That is the metric behaving, not the model failing.
    """
    def __init__(self, top_k: int = 3) -> None:
        super().__init__()

        self.top_k = top_k
        self.add_state("correct", default=t.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=t.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: t.Tensor, target: t.Tensor) -> None:
        # Get top K predicted class indices per sample: shape [B, K]
        _, top_k_indices = t.topk(preds, k=self.top_k, dim=-1)
        
        # Gather corresponding target values at top K positions: shape [B, K]
        hits = t.gather(target, dim=-1, index=top_k_indices)
        
        # Total positive labels present across samples
        num_positives = target.sum()
        
        if num_positives > 0:
            self.correct += hits.sum()
            self.total += num_positives

    def compute(self) -> t.Tensor:
        if self.total == 0:
            return t.tensor(0.0, device=self.correct.device) # type: ignore
        return self.correct / self.total # type: ignore


def _demo() -> None:
    """Check the two metrics against BABEL's own implementation, run on duplicated rows.

    Their loop scores single-label samples, and a *k*-label segment reaches it as *k*
    duplicates sharing one prediction. Everything in the chunk protocol depends on those
    two formulations agreeing, so this builds the duplicated rows explicitly and compares.
    """

    def babel_top_1_norm(preds: t.Tensor, target: t.Tensor) -> float:
        """train_test.py: mean(confusion.diag() / confusion.sum(1)), over duplicates."""

        num_labels = target.shape[1]
        confusion = t.zeros(num_labels, num_labels)
        for row, labels in enumerate(target):
            for label in labels.nonzero().flatten():
                confusion[label, preds[row].argmax()] += 1
        seen = confusion.sum(dim=1) > 0
        return float((confusion.diag()[seen] / confusion.sum(dim=1)[seen]).mean())

    def babel_top_k(preds: t.Tensor, target: t.Tensor, k: int) -> float:
        """feeder.py: sum(gt in top-k) / n_samples, over duplicates."""

        hits, total = 0, 0
        for row, labels in enumerate(target):
            top_k = preds[row].topk(k).indices
            for label in labels.nonzero().flatten():
                hits += int(label in top_k)
                total += 1
        return hits / total

    # row 0 carries {0,1} and predicts 0; row 1 carries {0} and predicts 1; row 2 carries
    # {1} and predicts 1; row 3 carries {2} and predicts 0. Supports are 2, 2, 1 and hits
    # are 1, 1, 0, so Top-1-norm is (1/2 + 1/2 + 0) / 3.
    preds = t.tensor([[9.0, 1.0, 0.0], [1.0, 9.0, 0.0], [1.0, 9.0, 0.0], [9.0, 1.0, 0.0]])
    target = t.tensor([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    metric = MultilabelTop1Norm(num_labels=3)
    assert abs(float(metric(preds, target)) - 1 / 3) < 1e-6, float(metric(preds, target))
    assert abs(float(metric(preds, target)) - babel_top_1_norm(preds, target)) < 1e-6
    print("ok (top_1_norm, hand-checked):", float(metric(preds, target)))

    # The same agreement on a multi-hot target wide enough to have empty classes.
    t.manual_seed(0)
    preds = t.randn(64, 12)
    target = (t.rand(64, 12) > 0.85).int()

    norm = MultilabelTop1Norm(num_labels=12)
    assert abs(float(norm(preds, target)) - babel_top_1_norm(preds, target)) < 1e-6
    for k in (1, 3, 5):
        ours = float(MultilabelTopKRecall(top_k=k)(preds, target))
        assert abs(ours - babel_top_k(preds, target, k)) < 1e-6, (k, ours)
    print("ok (top_k recall == BABEL Top-k over duplicated rows)")


if __name__ == "__main__":
    _demo()
