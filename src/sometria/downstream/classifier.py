"""Evaluating a pretrained backbone on labelled windows.

The backbone is frozen and in eval; what trains is a pooler and one Linear. The only
knob is how the ``T x D`` token grid collapses to a vector -- see
:mod:`sometria.downstream.pooling` -- because that is the one part of reading a
representation that is a real choice rather than a default.

``mean`` + Linear is the control: it adds no parameters, so its number is the
representation's. The attentive poolers add a small scoring network, which buys a
weighted read of the grid at the cost of no longer being strictly linear -- run both, or
the attentive number has nothing to be better than.

Finetuning is deliberately not here. A trainable backbone is a different experiment with
a different optimizer, and expressing both in one class meant every knob of one was an
implicit default of the other.

The target is multi-hot, so the loss is BCE and the metric is mAP -- see
:mod:`sometria.downstream.labels` for why this is not BABEL's official protocol.
"""

import lightning as L
import torch as t
import torch.nn as nn

from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

from sometria.architecture.encoder import MotionTransformerEncoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.metrics import MultilabelTopKRecall, WindowMeanAveragePrecision
from sometria.downstream.pooling import get_pooler


class MotionLinearClassifier(L.LightningModule):
    """Multi-label action classification over one window, on a frozen backbone."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder,
        num_labels: int,
        pool: str = "attentive_factorized",  # "mean" | "mean_max" | "attentive" | "attentive_factorized"
        lr: float = 1e-3,
        warmup: float = 0.03,
    ) -> None:
        super().__init__()

        # Not the backbone: its weights are an input to this experiment, not a result of
        # it, and they are already in the pretraining checkpoint the config names.
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.pool = pool
        self.warmup = warmup
        self.lr = lr

        spec = backbone.spec
        self.pooler = get_pooler(pool, spec.d_model, spec.num_dofs)

        self.head = nn.Sequential(
            nn.LayerNorm(self.pooler.out_dim, elementwise_affine=False, eps=1e-6),  # type: ignore
            nn.Linear(self.pooler.out_dim, num_labels),  # type: ignore
        )

        # macro-mAP, micro-mAP, macro-F1, top 1/3/5 recall
        self.val_macro_map = WindowMeanAveragePrecision(num_labels=num_labels)
        self.val_micro_map = MultilabelAveragePrecision(num_labels=num_labels, average="micro")
        self.val_macro_f1s = MultilabelF1Score(num_labels=num_labels, threshold=0.5, average="macro")
        self.val_top_1_rec = MultilabelTopKRecall(top_k=1)
        self.val_top_3_rec = MultilabelTopKRecall(top_k=3)
        self.val_top_5_rec = MultilabelTopKRecall(top_k=5)

        self.loss = nn.BCEWithLogitsLoss()

    def train(self, mode: bool = True) -> "MotionLinearClassifier":
        """Keep the backbone in eval mode; Lightning will not do it for you.

        ``requires_grad_(False)`` stops gradients but not dropout -- a frozen backbone
        that still samples a mask is not the fixed feature extractor a probe measures.
        """

        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, features: t.Tensor) -> t.Tensor:
        with t.no_grad():
            tokens = self.backbone.embed_tokens(features)
        return self.head(self.pooler(tokens))

    def _step(self, batch: dict, stage: str) -> tuple[t.Tensor, t.Tensor]:
        logits = self(batch["features"])
        loss = self.loss(logits, batch["labels"].float())
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss, logits

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")[0]

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        loss, logits = self._step(batch, "val")
        targets = batch["labels"].int()
        batch_size = logits.shape[0]

        # Logging the metric object rather than its value: Lightning updates it, computes
        # it once at epoch end, and resets it, so no epoch can inherit another's state.
        for name, metric, prog_bar in (
            ("macro_map", self.val_macro_map, True),
            ("micro_map", self.val_micro_map, False),
            ("macro_f1s", self.val_macro_f1s, False),
            ("top_1_rec", self.val_top_1_rec, False),
            ("top_3_rec", self.val_top_3_rec, False),
            ("top_5_rec", self.val_top_5_rec, False),
        ):
            metric.update(logits, targets)
            self.log(f"val/{name}", metric, prog_bar=prog_bar, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):  # type: ignore
        # Freezing is two things. requires_grad=False stops the gradient; leaving the
        # backbone out of the optimizer stops weight decay, which would otherwise keep
        # shrinking weights that are supposed to be fixed.
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
