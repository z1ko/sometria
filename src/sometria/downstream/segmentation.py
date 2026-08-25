"""Per-patch action segmentation on a frozen backbone.

The sibling of :class:`~sometria.downstream.classifier.MotionLinearClassifier`, and
deliberately the same experiment moved one axis over: frozen backbone, one Linear, the
same BABEL labels. What differs is that the target is resolved in time -- one multi-hot
per time patch rather than one per window -- so the head has to say *when*, not only
*what*.

That difference is the point. A window-level score can be reached by summary statistics
over the window: a linear model on the mean, standard deviation, min and max of each
channel reaches 94% of a pretrained probe on window classification, because "which joints
moved and how much" nearly determines the answer. Those statistics are order-invariant by
construction and cannot produce a target that varies within the window, so a temporal
task measures something a window-level task cannot.

The pooling is over DOFs only. The backbone emits a ``T x D`` grid flattened to
``(B, TP * D, d_model)``; folding the grid back and pooling the DOF axis leaves one
vector per time patch, which is exactly what the poolers in
:mod:`sometria.downstream.pooling` do to any ``(B, S, C)`` -- so they are reused here
against ``S = D`` rather than ``S = TP * D``. ``attentive_factorized`` is not available
for that reason: it pools both axes, and one of them has to survive.

No temporal layer sits between the backbone and the head. A probe measures what the
representation already carries, and a head that smooths its own predictions over time
would be measuring the smoother.
"""

import lightning as L
import torch as t
import torch.nn as nn

from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

from sometria.architecture.encoder import MotionTransformerEncoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.metrics import WindowMeanAveragePrecision
from sometria.downstream.pooling import get_pooler

#: Poolers that reduce one axis and can therefore leave the time axis standing.
DOF_POOLERS = ("mean", "mean_max", "attentive")


class MotionSegmenter(L.LightningModule):
    """Multi-label action segmentation over the time patches of one window."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder,
        num_labels: int,
        pool: str = "attentive",   # "mean" | "mean_max" | "attentive"
        lr: float = 1e-3,
        warmup: float = 0.03,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["backbone"])

        if pool not in DOF_POOLERS:
            raise ValueError(
                f"pool must be one of {DOF_POOLERS} -- {pool!r} reduces the time axis "
                "this task predicts along."
            )

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.pool = pool
        self.warmup = warmup
        self.lr = lr

        spec = backbone.spec
        self.num_dofs = spec.num_dofs
        self.pooler = get_pooler(pool, spec.d_model, spec.num_dofs)

        self.head = nn.Sequential(
            nn.LayerNorm(self.pooler.out_dim, elementwise_affine=False, eps=1e-6),  # type: ignore
            nn.Linear(self.pooler.out_dim, num_labels),  # type: ignore
        )

        # Computed over flattened (window, patch) rows, so a "sample" is one time patch:
        # the same metrics as the classifier, asking the temporal question.
        self.val_macro_map = WindowMeanAveragePrecision(num_labels=num_labels)
        self.val_micro_map = MultilabelAveragePrecision(num_labels=num_labels, average="micro")
        self.val_macro_f1s = MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro")

        self.loss = nn.BCEWithLogitsLoss()

    def train(self, mode: bool = True) -> "MotionSegmenter":
        """Keep the backbone in eval; see the classifier for why Lightning will not."""

        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, features: t.Tensor) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, num_time_patches, num_labels)``."""

        with t.no_grad():
            tokens = self.backbone.embed_tokens(features)

        # (B, TP * D, C) -> (B * TP, D, C): the poolers reduce dim 1, so handing them the
        # DOF axis leaves the time patches standing as independent rows.
        batch, length, width = tokens.shape
        patches = length // self.num_dofs
        pooled = self.pooler(tokens.reshape(batch * patches, self.num_dofs, width))

        return self.head(pooled).reshape(batch, patches, -1)

    def _step(self, batch: dict, stage: str) -> tuple[t.Tensor, t.Tensor]:
        logits = self(batch["features"])
        labels = batch["labels"].float()
        if labels.shape != logits.shape:
            raise ValueError(
                f"targets are {tuple(labels.shape)} but the head predicts {tuple(logits.shape)} -- "
                "set dataloader.label_patches to the number of time patches in a window."
            )

        loss = self.loss(logits, labels)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss, logits

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")[0]

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        loss, logits = self._step(batch, "val")

        # One row per (window, patch): a metric built for multi-label samples does not
        # need to know that some rows came from the same window.
        flat = logits.flatten(end_dim=1)
        target = batch["labels"].flatten(end_dim=1).int()

        for name, metric, prog_bar in (
            ("macro_map", self.val_macro_map, True),
            ("micro_map", self.val_micro_map, False),
            ("macro_f1s", self.val_macro_f1s, False),
        ):
            metric.update(flat, target)
            self.log(f"val/{name}", metric, prog_bar=prog_bar, on_epoch=True, batch_size=flat.shape[0])

        # Both: the fixed one is comparable to anything else logged at 0.5, the swept one
        # is the number that means something.
        self.log("val/boundary_f1s", boundary_f1(logits, batch["labels"]), on_epoch=True, batch_size=logits.shape[0])
        self.log("val/boundary_best_f1s", best_boundary_f1(logits, batch["labels"]), prog_bar=True, on_epoch=True, batch_size=logits.shape[0])
        return loss

    def configure_optimizers(self):  # type: ignore
        trainable = [*self.pooler.parameters(), *self.head.parameters()]
        optimizer = t.optim.AdamW(trainable, lr=self.lr)

        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }


#: Thresholds :func:`best_boundary_f1` sweeps. Dense at the low end, because a target
#: with ~2 of 60 labels per patch puts a calibrated model's scores there.
BOUNDARY_THRESHOLDS = tuple(i / 100 for i in range(2, 61, 2))


def best_boundary_f1(logits: t.Tensor, targets: t.Tensor) -> t.Tensor:
    """:func:`boundary_f1` maximized over :data:`BOUNDARY_THRESHOLDS`.

    A fixed 0.5 cutoff measures calibration, not localization: the median patch carries
    ~2 of 60 labels, so a calibrated model keeps almost every score far below 0.5 and
    predicts almost no change. On this data it costs the probe a third of its score --
    0.0787 at 0.5 against 0.1214 at 0.14 -- and it costs a baseline something different,
    which makes any comparison at 0.5 a comparison of two calibrations.

    Swept, so the number is what the predictions support and two models are read the same
    way. Report the threshold alongside it.
    """

    return max(
        (boundary_f1(logits, targets, threshold=threshold) for threshold in BOUNDARY_THRESHOLDS),
        key=float,
    )


def boundary_f1(logits: t.Tensor, targets: t.Tensor, threshold: float = 0.5) -> t.Tensor:
    """F1 over label *changes* between consecutive patches -- did the model move when the
    annotation moved.

    The metric a window-level score cannot have and the one a summary statistic cannot
    score above chance on: it looks only at where the target changes, so a prediction
    that is constant across the window earns nothing however well it names the action.
    """

    predicted = ((logits.sigmoid() > threshold).float().diff(dim=1) != 0).any(dim=-1)
    actual = (targets.float().diff(dim=1) != 0).any(dim=-1)

    true_positive = (predicted & actual).sum()
    if true_positive == 0:
        return t.zeros((), device=logits.device)

    precision = true_positive / predicted.sum().clamp(min=1)
    recall = true_positive / actual.sum().clamp(min=1)
    return 2 * precision * recall / (precision + recall)
