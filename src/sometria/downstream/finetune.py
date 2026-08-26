"""The same readout as the linear probe, with the backbone training under it.

A probe asks what a representation already contains; a finetune asks what it is a good
*starting point* for. They share the head, the loss and the metrics, so this is
:class:`MotionLinearClassifier` with the three things that make it a probe removed:
the backbone stays in train mode, the gradient reaches it, and it is in the optimizer.

Two knobs, and both are the same knob: the backbone moves at ``backbone_lr`` while the
fresh head moves at ``lr``. Pretrained weights are close to where they should be and a
random head is not, so one rate for both either scrambles the backbone or starves the
head. Ten to a hundred times lower is the usual range.

Not here: layer-wise learning-rate decay. It is the next thing to try if a lower
``backbone_lr`` alone does not stop the early epochs from undoing the pretraining, but it
is a per-block optimizer group for a gain this codebase has not measured yet.
"""

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import MotionTransformerEncoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.classifier import MotionLinearClassifier


def param_groups(module: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    """Split ``module`` into a decayed and an undecayed group at one learning rate.

    Biases, LayerNorm gains and the positional table are 1-D and are excluded: decay on
    them is not regularization, it is a constant pull of the normalization statistics
    toward zero, and on a pretrained backbone it erases the scales pretraining set.
    """

    decay = [p for p in module.parameters() if p.requires_grad and p.ndim >= 2]
    plain = [p for p in module.parameters() if p.requires_grad and p.ndim < 2]
    # Empty groups dropped: a parameterless pooler ("mean") would otherwise contribute two
    # of them, and every lr log line then carries a rate that steers nothing.
    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for params, wd in ((decay, weight_decay), (plain, 0.0))
        if params
    ]


class MotionFinetuneClassifier(MotionLinearClassifier):
    """Multi-label action classification over one window, backbone included."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder,
        num_labels: int,
        pool: str = "attentive_factorized",
        min_lr_frac: float = 0.01,
        lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup: float = 0.05,
    ) -> None:
        super().__init__(
            backbone,
            num_labels,
            pool=pool,
            min_lr_frac=min_lr_frac,
            lr=lr,
            warmup=warmup,
        )

        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay

        # Undo the probe. The parent froze and eval'd the backbone in its own __init__,
        # which is correct there and is exactly what this class exists not to do.
        self.backbone.requires_grad_(True)
        self.backbone.train()

    def train(self, mode: bool = True) -> "MotionFinetuneClassifier":
        """Follow the parent module, unlike the probe, which pins the backbone to eval.

        Skipping one level in the MRO rather than calling ``super()``: the method being
        skipped is the freeze itself.
        """

        L.LightningModule.train(self, mode)
        return self

    def forward(self, features: t.Tensor) -> t.Tensor:
        # No torch.no_grad, which is the whole difference: the probe's forward severs the
        # graph at the backbone, so a gradient that reached it would have nowhere to go.
        return self.head(self.pooler(self.backbone.embed_tokens(features)))

    def configure_optimizers(self):  # type: ignore
        groups = [
            *param_groups(self.backbone, self.backbone_lr, self.weight_decay),
            *param_groups(self.pooler, self.lr, self.weight_decay),
            *param_groups(self.head, self.lr, self.weight_decay),
        ]
        # lr comes from each group; passing it here only sets a default for groups that
        # omit it, and none of these do.
        optimizer = t.optim.AdamW(groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # A factor schedule, not an absolute floor: it scales each group by its
                # own base lr, so the backbone at 1e-4 and the head at 1e-3 decay in
                # step instead of the backbone annealing toward a rate above its own.
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    min_factor=self.min_lr_frac,
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
