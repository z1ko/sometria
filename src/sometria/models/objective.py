"""The training loop every masked pretext objective sits inside.

A **Pretext objective** is what is hidden, what is predicted, and what the loss is. The
first of those comes from a :class:`~sometria.masking.MaskSpec` and the last two are the
objective's own -- but the loop around them is identical whichever target it picked, and
was written out once per objective until this module existed.

The downstream classifier deliberately stays out. CONTEXT.md calls it a **Protocol**
rather than a **Pretext objective**, and its optimizer agrees: two parameter groups at
different rates, AdamW's own betas and weight decay, no ``min_lr_frac``. Expressing that
here would mean three more knobs on this class for exactly one call site, and every one
of them is an implicit default today -- getting one wrong would silently move every probe
number ever measured.
"""

from dataclasses import asdict

import lightning as L
import torch as t

from sometria.architecture.encoder import MotionTransformerEncoder, backbone_hparam
from sometria.architecture.scheduler import lr_schedule
from sometria.masking import MaskSpec, as_mask_spec


class PretextObjective(L.LightningModule):
    """Optimizer, schedule and steps for a self-supervised objective over a backbone.

    A subclass implements :meth:`step` and, if it does not train every parameter it owns,
    :meth:`trainable_parameters`. Everything else about training is here.
    """

    #: What this objective masks when a config does not say. JEPA holds out a quarter of
    #: the window where masked reconstruction holds out nine tenths.
    DEFAULT_MASK: MaskSpec = MaskSpec()

    #: Which of :meth:`step`'s metrics reach the progress bar. The loss always does.
    PROG_BAR: tuple[str, ...] = ()

    def __init__(
        self,
        mask: MaskSpec | dict | None = None,
        *,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()

        self.mask = as_mask_spec(mask, self.DEFAULT_MASK)
        self.lr = lr
        self.min_lr_frac = min_lr_frac
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac

    def save_objective_hyperparameters(
        self,
        backbone: MotionTransformerEncoder,
        **objective,
    ) -> None:
        """Store the specs as plain fields, plus whatever the objective adds.

        Neither spec goes in as its dataclass: Lightning refuses to log a frozen one
        ("A frozen dataclass was passed to `apply_to_collection`") and ``torch.load``
        defaults to ``weights_only=True``, which rejects any class it has not been told
        about. A dict travels through both, which is what lets
        ``load_from_checkpoint(path)`` rebuild an objective unaided.
        """

        self.save_hyperparameters(
            {
                "backbone": backbone_hparam(backbone),
                "mask": asdict(self.mask),
                "lr": self.lr,
                "min_lr_frac": self.min_lr_frac,
                "weight_decay": self.weight_decay,
                "warmup_frac": self.warmup_frac,
                **objective,
            }
        )

    def step(self, batch: dict) -> tuple[t.Tensor, dict[str, t.Tensor | float]]:
        """Return this objective's loss, and whatever else is worth logging beside it.

        The only thing a subclass has to say about training. Metric names are bare here
        and get their ``train/`` or ``val/`` prefix from :meth:`_step`, so one objective
        cannot log a series the other stage does not have.
        """

        raise NotImplementedError

    def trainable_parameters(self):
        """What the optimizer is given. Overridden by an objective holding frozen parts."""

        return self.parameters()

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        loss, metrics = self.step(batch)
        batch_size = batch["features"].shape[0]

        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, prog_bar=name in self.PROG_BAR, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self): # type: ignore
        # betas and weight decay follow the reference's AdamW; its cosine decays to
        # min_lr = lr / 2 rather than to zero, which min_lr_frac carries.
        optimizer = t.optim.AdamW(
            self.trainable_parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup_frac * total_steps)),
                    total_steps=total_steps,
                    min_factor=self.min_lr_frac,
                ),
                "interval": "step",
            },
        }
