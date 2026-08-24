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

import torch as t
import torch.nn as nn

from sometria.architecture.encoder import (
    EncoderSpec,
    MotionTransformerEncoder,
    as_backbone,
    transformer_stack,
)
from sometria.architecture.pos_embed import PositionalEncoding
from sometria.masking import MaskSpec
from sometria.models.objective import PretextObjective
from sometria.models.window import MaskedWindow, mask_window, masked_token_mse


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
    ) -> t.Tensor:
        batch, target_count = targets_idx.shape

        context = context + self.position.gather(context_idx, num_time_patches)
        targets = self.mask_token.expand(batch, target_count, -1)
        targets = targets + self.position.gather(targets_idx, num_time_patches)

        predicted = self.blocks(t.cat([context, targets], dim=1))
        return predicted[:, -target_count:]


class MotionJEPA(PretextObjective):
    """Predict an EMA teacher's target-token embeddings from student context tokens."""

    # JEPA holds out a quarter of the window where masked reconstruction holds out nine
    # tenths: its target is an embedding the teacher computed from the whole grid, not a
    # patch the decoder has to invent.
    DEFAULT_MASK = MaskSpec(mask_ratio=0.25, tau=0.80, score_channels=(2,))

    # The collapse guard belongs on the bar: a JEPA whose embeddings stop varying drives
    # its own loss to zero, so val/loss alone cannot say whether training is working.
    PROG_BAR = ("embed_std",)

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
        super().__init__(
            mask,
            lr=lr,
            min_lr_frac=min_lr_frac,
            weight_decay=weight_decay,
            warmup_frac=warmup_frac,
        )

        backbone = as_backbone(backbone)
        if not 0.0 <= ema_start <= ema_end <= 1.0:
            raise ValueError("EMA values must satisfy 0 <= ema_start <= ema_end <= 1.")

        self.save_objective_hyperparameters(
            backbone,
            predictor_depth=predictor_depth,
            ema_start=ema_start,
            ema_end=ema_end,
        )

        self.student = backbone
        self.teacher = deepcopy(backbone)
        self.teacher.requires_grad_(False)
        self.predictor = MotionPredictor(backbone.spec, depth=predictor_depth)

        self.ema_start = ema_start
        self.ema_end = ema_end

    def forward(
        self,
        features: t.Tensor,
        generator: t.Generator | None = None,
    ) -> tuple[t.Tensor, t.Tensor, MaskedWindow]:
        """Return ``(prediction, teacher_target, window)`` over target tokens."""

        window = mask_window(features, self.student.spec, self.mask, generator=generator)

        context = self.student.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )
        prediction = self.predictor(
            context,
            context_idx=window.mask.context,
            targets_idx=window.mask.targets,
            num_time_patches=window.num_time_patches,
        )

        with t.no_grad():
            teacher_tokens = self.teacher.embed_values(window.values, window.num_time_patches)
            target = window.mask.targets_of(teacher_tokens)

        return prediction, target, window

    def step(self, batch: dict) -> tuple[t.Tensor, dict[str, t.Tensor | float]]:
        prediction, target, window = self(batch["features"])
        loss = masked_token_mse(prediction, target)
        return loss, {
            "context_tokens": float(window.mask.context.shape[1]),
            # Spread of the teacher's target vectors across the batch. Measured on the
            # target tokens rather than the whole grid because those are the ones the
            # loss scores: if these collapse to a constant, the loss reaches zero while
            # the backbone has learned nothing.
            "embed_std": target.std(dim=0, unbiased=False).mean(),
        }

    def trainable_parameters(self):
        """The teacher is an EMA copy, not an optimized module."""

        return list(self.student.parameters()) + list(self.predictor.parameters())

    def train(self, mode: bool = True) -> "MotionJEPA":
        """Keep the teacher in eval mode; Lightning will not do it for you.

        ``requires_grad_(False)`` stops gradients but not dropout, and not a BatchNorm's
        running statistics. A teacher that drops units while computing the target makes
        the target stochastic, so the student is asked to predict a different vector each
        time it sees the same window -- and the EMA copies the student's buffers over
        anyway, so any statistic the teacher gathered would be discarded.
        """

        super().train(mode)
        self.teacher.eval()
        return self

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
