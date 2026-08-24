"""Joint-Embedding Predictive Architecture over motion tokens.

The student reads only context tokens. A shallow predictor receives those contextualized
context tokens plus learned target slots, and predicts the teacher's full-grid embedding
at the target indices. The teacher is an EMA copy of the student and is never optimized
directly.
"""

from copy import deepcopy

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import (
    EncoderSpec,
    MotionTransformerEncoder,
    as_backbone,
    backbone_hparam,
)
from sometria.architecture.pos_embed import PositionalEncoding
from sometria.architecture.scheduler import lr_schedule
from sometria.masking import MaskIndices, motion_aware_mask, patchify, token_validity


class MotionPredictor(nn.Module):
    """Predict target embeddings from context embeddings and target positions."""

    def __init__(self, spec: EncoderSpec, *, depth: int = 4) -> None:
        super().__init__()

        time_patches, num_dofs = spec.grid_shape
        self.spec = spec
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.position = PositionalEncoding(time_patches, num_dofs, spec.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.num_heads,
            dim_feedforward=int(spec.d_model * spec.mlp_ratio),
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(spec.d_model), enable_nested_tensor=False
        )
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
        *,
        predictor_depth: int = 4,
        mask_ratio: float = 0.25,
        tau: float = 0.80,
        score_channels: tuple[int, ...] = (2,),
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()

        backbone = as_backbone(backbone)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1 for JEPA.")
        if tau > 0 and not score_channels:
            raise ValueError("motion-aware masking needs at least one score channel.")
        if not 0.0 <= ema_start <= ema_end <= 1.0:
            raise ValueError("EMA values must satisfy 0 <= ema_start <= ema_end <= 1.")

        self.save_hyperparameters(
            {
                "backbone": backbone_hparam(backbone),
                "predictor_depth": predictor_depth,
                "mask_ratio": mask_ratio,
                "tau": tau,
                "score_channels": tuple(score_channels),
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

        self.mask_ratio = mask_ratio
        self.tau = tau
        self.score_channels = tuple(score_channels)
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
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, MaskIndices]:
        """Return ``(prediction, teacher_target, target_valid, mask)`` over target tokens."""

        spec = self.student.spec
        patches = self.student.time_patches(features.shape[1])
        tokens_in = patchify(features, spec.patch_size)
        values = tokens_in.flatten(start_dim=-2)

        mask = motion_aware_mask(
            tokens_in,
            score_channels=self.score_channels,
            mask_ratio=self.mask_ratio,
            tau=self.tau,
            valid=valid,
            generator=generator,
        )
        target_valid = (
            t.ones(mask.targets.shape, dtype=t.bool, device=values.device)
            if valid is None else _gather_tokens(
                token_validity(valid, spec.patch_size, values.shape[1]).to(values.device),
                mask.targets,
            )
        )
        # ponytail: mask_ratio=0.25 gives 322 target slots on a 1290-token window; a
        # padded window with more invalid tokens than that would spill padding into
        # context, but current pretraining excludes short windows and passes valid=None.

        context = self.student.embed_values(values, patches, index=mask.context)
        prediction = self.predictor(
            context,
            context_idx=mask.context,
            targets_idx=mask.targets,
            num_time_patches=patches,
            target_valid=target_valid,
        )

        with t.no_grad():
            teacher_tokens = self.teacher.embed_values(values, patches, valid=valid)
            target = _gather_tokens(teacher_tokens, mask.targets)

        self._last_teacher_embed_std = teacher_tokens.std(dim=0, unbiased=False).mean()

        return prediction, target, target_valid, mask

    def prediction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        """MSE per target token, ignoring padded targets."""

        if not target_valid.any():
            return (prediction - target).square().mean() * 0.0

        per_token = (prediction - target).square().mean(dim=-1)
        return (per_token * target_valid).sum() / target_valid.sum()

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        prediction, target, target_valid, mask = self(batch["features"], batch.get("valid"))
        loss = self.prediction_loss(prediction, target, target_valid)
        batch_size = batch["features"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/context_tokens", float(mask.context.shape[1]), batch_size=batch_size)
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


def _gather_tokens(x: t.Tensor, idx: t.Tensor) -> t.Tensor:
    if x.ndim == 2:
        return x.gather(dim=1, index=idx)
    return x.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
