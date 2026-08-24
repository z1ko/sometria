"""Evaluating a pretrained backbone on labelled windows.

The protocol is three independent knobs, not an enum:

- ``pool``            -- ``"window"`` to ``d_model``, or ``"dof"`` to ``num_dofs * d_model``
- ``head``            -- ``"linear"`` or ``"mlp"``
- ``freeze_backbone`` -- trainable or not

*Linear probe* is ``pool="window", head="linear", freeze_backbone=True``; *finetune* is
``pool="dof", head="mlp", freeze_backbone=False``. Keeping them separate means a frozen
backbone under a deep head stays expressible as the control it is, rather than being an
unrepresentable combination of two named modes.

The target is multi-hot, so the loss is BCE and the metric is mAP -- see
:mod:`sometria.downstream.labels` for why this is not BABEL's official protocol.
"""

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.metrics import WindowMeanAveragePrecision
from sometria.models.masked import MaskedMotionAutoencoder

HEADS = ("linear", "mlp")


class MotionWindowClassifier(L.LightningModule):
    """Multi-label action classification over one window."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | None = None,
        *,
        num_labels: int = 150,
        pool: str = "window",
        head: str = "linear",
        freeze_backbone: bool = True,
        dropout: float = 0.5,
        lr_head: float = 1e-3,
        lr_backbone: float = 1e-5,
        warmup_frac: float = 0.03,
    ) -> None:
        super().__init__()

        if isinstance(backbone, EncoderSpec) or backbone is None:
            backbone = MotionTransformerEncoder(backbone)
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")

        self.save_hyperparameters(
            {
                "backbone": backbone.spec,
                "num_labels": num_labels,
                "pool": pool,
                "head": head,
                "freeze_backbone": freeze_backbone,
                "dropout": dropout,
                "lr_head": lr_head,
                "lr_backbone": lr_backbone,
                "warmup_frac": warmup_frac,
            }
        )

        self.backbone = backbone
        self.pool = pool
        self.freeze_backbone = freeze_backbone
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.warmup_frac = warmup_frac

        width = backbone.spec.pooled_dim(pool)
        if head == "linear":
            # affine=False: a probe that is allowed to scale and shift per feature is
            # measuring the head, not the representation. It still centres, which is what
            # keeps a single Linear trainable over unnormalized pooled features.
            self.head = nn.Sequential(
                nn.BatchNorm1d(width, affine=False, eps=1e-6),
                nn.Linear(width, num_labels),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(width, backbone.spec.d_model),
                nn.BatchNorm1d(backbone.spec.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(backbone.spec.d_model, num_labels),
            )

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.loss = nn.BCEWithLogitsLoss()
        self.val_map = WindowMeanAveragePrecision(num_labels=num_labels)

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs) -> "MotionWindowClassifier":
        """Build a classifier around the backbone of a pretrained objective."""

        pretrained = MaskedMotionAutoencoder.load_from_checkpoint(checkpoint, map_location="cpu")
        return cls(pretrained.backbone, **kwargs)

    def train(self, mode: bool = True) -> "MotionWindowClassifier":
        """Keep a frozen backbone in eval mode; Lightning will not do it for you.

        ``requires_grad_(False)`` stops gradients but not dropout, and not a BatchNorm's
        running statistics -- a frozen backbone that keeps updating its buffers is not
        the fixed feature extractor a linear probe is supposed to measure.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, features: t.Tensor, valid: t.Tensor | None = None) -> t.Tensor:
        if self.freeze_backbone:
            with t.no_grad():
                pooled = self.backbone.embed(features, pool=self.pool, valid=valid)
        else:
            pooled = self.backbone.embed(features, pool=self.pool, valid=valid)
        return self.head(pooled)

    def _step(self, batch: dict, stage: str) -> tuple[t.Tensor, t.Tensor]:
        logits = self(batch["features"], batch.get("valid"))
        loss = self.loss(logits, batch["labels"])
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss, logits

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")[0]

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        loss, logits = self._step(batch, "val")
        self.val_map.update(logits, batch["labels"].int())
        self.log("val/map", self.val_map, prog_bar=True, batch_size=logits.shape[0])
        return loss

    def configure_optimizers(self): # type: ignore
        # Freezing is two things. requires_grad=False stops the gradient; leaving the
        # parameters out of the optimizer stops weight decay, which would otherwise keep
        # shrinking weights that are supposed to be fixed.
        groups = [{"params": self.head.parameters(), "lr": self.lr_head}]
        if not self.freeze_backbone:
            groups.append({"params": self.backbone.parameters(), "lr": self.lr_backbone})

        optimizer = t.optim.AdamW(groups, lr=self.lr_head)
        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup_frac * total_steps)),
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
