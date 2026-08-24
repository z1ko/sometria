"""mAP over a multi-label window benchmark.

``torchmetrics`` computes per-label average precision; the only thing worth writing here
is what to do with a label that never appears. It scores 0, and with a 150-way
frequency-ordered vocabulary over a validation split most of the tail is absent -- macro
averaging would then report the fraction of the vocabulary that happens to occur rather
than how well the model does on what it saw. Labels with no positive are left out.
"""

import torch as t
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.utilities.data import dim_zero_cat


class WindowMeanAveragePrecision(MultilabelAveragePrecision):
    """Per-label AP averaged over the labels that actually occur in the targets."""

    def __init__(self, num_labels: int) -> None:
        super().__init__(num_labels=num_labels, average="none")

    def compute(self) -> t.Tensor:
        per_label = super().compute()
        present = dim_zero_cat(self.target).sum(dim=0) > 0
        if not present.any():
            return t.zeros((), device=per_label.device)
        return per_label[present].mean()
