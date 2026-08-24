"""Joint-Embedding Predictive Architecture over motion tokens.

The student reads only context tokens. A shallow predictor receives those contextualized
context tokens plus learned target slots, and predicts the teacher's full-grid embedding
at the target indices. The teacher is an EMA copy of the student and is never optimized
directly.

Splitting the window into **Context** and **Target** belongs to
:mod:`sometria.models.window`; what lives here is the two backbones, the predictor, and
the EMA.
"""

from copy import deepcopy
from dataclasses import asdict

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import (
    EncoderSpec,
    MotionTransformerEncoder,
    as_backbone,
    backbone_hparam,
    transformer_stack,
)
from sometria.architecture.pos_embed import PositionalEncoding
from sometria.architecture.scheduler import lr_schedule
from sometria.masking import MaskSpec, as_mask_spec
from sometria.models.window import MaskedWindow, mask_window, masked_token_mse

# JEPA holds out a quarter of the window where masked reconstruction holds out nine
# tenths: its target is an embedding the teacher computed from the whole grid, not a
# patch the decoder has to invent.
DEFAULT_MASK = MaskSpec(mask_ratio=0.25, tau=0.80, score_channels=(2,))


class MotionPredictor(nn.Module):
    """Predict target embeddings from context embeddings and target positions."""

    def __init__(self, spec: EncoderSpec, *, depth: int = 4) -> None:
        super().__init__()

        time_patches, num_dofs = spec.grid_shape
        self.spec = spec
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.position = PositionalEncoding(time_patches, num_dofs, spec.d_model)
        self.blocks = transformer_stack(spec, depth)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context: t.Tensor,
        *,
        context_idx: t.Tensor,
        targets_idx: t.Tensor,
        num_time_patches: int,
        target_valid: t.Tensor | None = None,
    ) -> t.Tensor:
        batch, target_count = targets_idx.shape

        context = context + self.position.gather(context_idx, num_time_patches)
        targets = self.mask_token.expand(batch, target_count, -1)
        targets = targets + self.position.gather(targets_idx, num_time_patches)

        padding = None
        if target_valid is not None:
            padding = t.cat(
                [
                    t.zeros(context_idx.shape, dtype=t.bool, device=context.device),
                    ~target_valid.to(context.device),
                ],
                dim=1,
            )

        predicted = self.blocks(t.cat([context, targets], dim=1), src_key_padding_mask=padding)
        return predicted[:, -target_count:]


class MotionJEPA(L.LightningModule):
    """Predict an EMA teacher's target-token embeddings from student context tokens."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | dict | None = None,
        mask: MaskSpec | dict | None = None,
        *,
        predictor_depth: int = 4,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()

        backbone = as_backbone(backbone)
        mask = as_mask_spec(mask, DEFAULT_MASK)
        if not 0.0 <= ema_start <= ema_end <= 1.0:
            raise ValueError("EMA values must satisfy 0 <= ema_start <= ema_end <= 1.")

        self.save_hyperparameters(
            {
                "backbone": backbone_hparam(backbone),
                "mask": asdict(mask),
                "predictor_depth": predictor_depth,
                "ema_start": ema_start,
                "ema_end": ema_end,
                "lr": lr,
                "min_lr_frac": min_lr_frac,
                "weight_decay": weight_decay,
                "warmup_frac": warmup_frac,
            }
        )

        self.student = backbone
        self.teacher = deepcopy(backbone)
        self.teacher.requires_grad_(False)
        self.predictor = MotionPredictor(backbone.spec, depth=predictor_depth)

        self.mask = mask
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.lr = lr
        self.min_lr_frac = min_lr_frac
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac

    def forward(
        self,
        features: t.Tensor,
        valid: t.Tensor | None = None,
        generator: t.Generator | None = None,
    ) -> tuple[t.Tensor, t.Tensor, MaskedWindow]:
        """Return ``(prediction, teacher_target, window)`` over target tokens."""

        window = mask_window(
            features, self.student.spec, self.mask, valid=valid, generator=generator
        )
        # ponytail: mask_ratio=0.25 gives 322 target slots on a 1290-token window; a
        # padded window with more invalid tokens than that would spill padding into
        # context, but current pretraining excludes short windows and passes valid=None.

        context = self.student.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )
        prediction = self.predictor(
            context,
            context_idx=window.mask.context,
            targets_idx=window.mask.targets,
            num_time_patches=window.num_time_patches,
            target_valid=window.target_valid,
        )

        with t.no_grad():
            teacher_tokens = self.teacher.embed_values(
                window.values, window.num_time_patches, valid=valid
            )
            target = window.mask.targets_of(teacher_tokens)

        self._last_teacher_embed_std = teacher_tokens.std(dim=0, unbiased=False).mean()

        return prediction, target, window

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        prediction, target, window = self(batch["features"], batch.get("valid"))
        loss = masked_token_mse(prediction, target, window.target_valid)
        batch_size = batch["features"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/context_tokens", float(window.mask.context.shape[1]), batch_size=batch_size)
        self.log(
            f"{stage}/embed_std",
            self._last_teacher_embed_std,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "val")

    @t.no_grad()
    def update_teacher(self, ema: float) -> None:
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters(), strict=True
        ):
            teacher_param.data.mul_(ema).add_(student_param.data, alpha=1.0 - ema)
        for student_buffer, teacher_buffer in zip(
            self.student.buffers(), self.teacher.buffers(), strict=True
        ):
            teacher_buffer.copy_(student_buffer)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None: # type: ignore
        total = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, float(self.trainer.global_step) / float(total))
        ema = self.ema_start + (self.ema_end - self.ema_start) * progress
        self.update_teacher(ema)
        self.log("train/ema", ema, batch_size=batch["features"].shape[0])

    def configure_optimizers(self): # type: ignore
        optimizer = t.optim.AdamW(
            list(self.student.parameters()) + list(self.predictor.parameters()),
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
