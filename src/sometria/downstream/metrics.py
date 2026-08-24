"""mAP over a multi-label window benchmark.

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
