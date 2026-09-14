"""JEPA over motion tokens: predict a teacher's embeddings, not the input's values.

Deliberately one axis away from :class:`~sometria.models.baseline.MAE`. Same uniform
masking at the same ratio, same encoder, same predictor shape, same MSE, same optimizer.
The single difference is what the loss is scored against: MAE regresses the raw patch it
hid, JEPA regresses the embedding an EMA teacher computed for that position. Anything
else that differed would confound the comparison the two models exist to make.

Shape of the thing, following S-JEPA (Abdelfattah and Alahi, ECCV 2024, "S-JEPA: A Joint
Embedding Predictive Architecture for Skeletal Action Recognition"):

- The **student** sees the context tokens only -- 129 of 1290 at ``mask_ratio=0.90``.
- The **teacher** sees the *whole* grid, and the mask is applied to its **output**. Their
  Table 6 prices this: masking the teacher's input instead costs 7.2 points on NTU60,
  because a target computed with full self-attention carries the context of the whole
  window, while one computed from an isolated patch does not.
- The **predictor** is :class:`~sometria.models.baseline.Decoder` with its reconstruction
  head swapped for the encoder width, which is the same module S-JEPA describes: insert
  mask tokens at the held-out positions, add positional encodings, run the blocks. Their
  Table 7 sweeps its depth over 4-7 and width over 64-512 and lands on 5 and 256, which
  are already this repo's ``dec_depth`` and ``dim`` defaults, so nothing is tuned here.
- At probe and fine-tune time the **teacher** is the encoder that gets read, which is what
  the paper does ("At fine-tuning and test times, only the target encoder weights are
  used"). :meth:`encode` is the teacher for that reason.

What this deliberately leaves out. S-JEPA additionally centres its targets, sharpens them
and scores them with a cross-entropy, and feeds its two encoders geometrically augmented
views. None of that is here:

- **Views** are not expressible. The paper rotates and translates 3D skeletons; this
  representation excludes all six pelvis DOFs (see ``config/human.yaml``), so global
  orientation and translation are already normalized out and both transforms are no-ops.
  Their Table 5 prices the whole component at 0.7 points.
- **Centering and cross-entropy** are priced at 0.8 and 1.6 points, and are left out to
  keep the one-axis comparison. Note the risk that buys: their Table 5 measures MSE *with*
  centering and no-centering *with* cross-entropy, never both removed at once, and the
  paper's "Stabilizing training" paragraph describes exactly this configuration as
  unstable. That is what ``embed_std`` and ``loss_over_null`` are logged to catch. If
  ``loss_over_null`` settles at 1.0 the fallbacks, cheapest first, are target centering
  and then the cross-entropy with sharpening.

Nothing anchors these embeddings to the input, so a student and teacher that agree on a
constant drive the loss to zero having learned nothing. ``val/loss`` is therefore not a
quality signal and must not select checkpoints -- see ``loss_over_null`` below, and
``config/pretrain_jepa.yaml``, which monitors it instead.
"""

import math
from copy import deepcopy

import lightning as L
import torch
import torch.nn.functional as F

from sometria.architecture.scheduler import lr_schedule
from sometria.models.baseline import Decoder, Encoder, gather_tokens, random_mask


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance per token, across the feature axis.

    The same operation :class:`~sometria.models.baseline.MAE` applies to its
    reconstruction targets, applied here to the teacher's embeddings, so the two
    objectives still differ on one axis and not two.

    It also happens to be the target normalization I-JEPA and Brain-JEPA both apply in
    their released code while omitting it from their papers. With centering declined it is
    the only thing standing between the loss and a teacher whose scale runs away, though
    it constrains scale only: a teacher whose embeddings all collapse to the same
    *direction* still normalizes to the same unit vector, and only ``embed_std`` sees it.
    """

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1e-6).sqrt()


class JEPA(L.LightningModule):
    def __init__(
        self,
        num_dofs: int = 43,
        num_frames_in_patch: int = 8,
        num_frames: int = 240,
        enc_depth: int = 8,
        dec_depth: int = 5,
        mask_ratio: float = 0.90,
        min_lr_ratio: float = 0.5,
        warmup: float = 0.05,
        weight_decay: float = 0.05,
        mlp_ratio: int = 4,
        # No `channels_output`: nothing here reconstructs a channel. The predictor's head
        # is the encoder width, so the output channel set has nowhere to apply.
        channels_input: tuple[int, ...] = (0, 1, 2, 3, 4),
        num_heads: int = 8,
        dim: int = 256,
        dec_dim: int | None = None,
        # S-JEPA ramps 0.9999 -> 1.0, tuned for a 1200-epoch run. An EMA at rate `lambda`
        # averages over roughly 1/(1 - lambda) steps, so 0.9999 spans 10,000 -- about 5%
        # of their run, and about 74% of a 100-epoch run here, which logs ~13,600 steps.
        # A teacher averaged over three quarters of training barely leaves its
        # initialization. 0.998 spans ~500 steps, which is the same 5% at this scale.
        # Recompute it if the run length changes; do not inherit the constant.
        ema_start: float = 0.998,
        ema_stop: float = 1.000,
        lr: float = 1e-3,
    ) -> None:

        super().__init__()
        if not 0.0 <= ema_start <= ema_stop <= 1.0:
            # A swapped pair ramps the teacher *down*, which does not fail loudly -- it
            # trains, and quietly walks the teacher toward the collapse the EMA exists to
            # prevent. Cheaper to reject here than to read it off a loss curve later.
            raise ValueError(
                f"EMA must satisfy 0 <= ema_start <= ema_stop <= 1; got {ema_start} and {ema_stop}."
            )
        self.save_hyperparameters()

        self.mask_ratio = mask_ratio
        self.num_frames_in_patch = num_frames_in_patch
        self.channels_input = channels_input
        self.weight_decay = weight_decay
        self.min_lr_ratio = min_lr_ratio
        self.warmup = warmup
        self.ema_start = ema_start
        self.ema_stop = ema_stop
        self.lr = lr

        Te = num_frames // num_frames_in_patch
        self.num_tokens = Te * num_dofs

        self.encoder_student = Encoder(
            num_dofs, num_frames_in_patch, dim, enc_depth, num_heads, channels_input, mlp_ratio, Te
        )
        # Teacher starts as an exact copy, then only ever moves by EMA. `requires_grad_`
        # is not decoration: without it the teacher's parameters sit in `self.parameters()`
        # looking trainable, and the next person to write `AdamW(self.parameters())` here
        # gets weight decay applied to a module that is supposed to be a moving average.
        self.encoder_teacher = deepcopy(self.encoder_student)
        self.encoder_teacher.requires_grad_(False)

        self.predictor = Decoder(
            num_dofs=num_dofs,
            num_frames_in_patch=num_frames_in_patch,
            max_te=Te,
            # Unused: `out_dim` below takes over the head, so the channel count that would
            # have sized it is never read.
            channels_output=(),
            dim=dim,
            depth=dec_depth,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            dec_dim=dec_dim,
            # The predictor lands in the teacher's embedding space, not in patch space.
            out_dim=dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, C) with full channels -> (B, N, E). The teacher, for probes."""
        return self.encoder_teacher(x[..., self.channels_input], None, None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, V, C) -> (prediction, target), both (B, N_masked, dim)."""

        keep_idx, mask_idx = random_mask(self.num_tokens, self.mask_ratio, x.size(0), x.device)
        values = x[..., self.channels_input]

        h = self.encoder_student(values, keep_idx, None)
        prediction = gather_tokens(self.predictor(h, keep_idx), mask_idx)

        with torch.no_grad():
            # `keep=None`: the teacher reads every token. Masking its output rather than
            # its input is the 7.2-point decision from the paper's Table 6.
            target = self.encoder_teacher(values, None, None)
        target = standardize(gather_tokens(target, mask_idx))

        return prediction, target

    def step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction, target = self.forward(x)
        loss = F.mse_loss(prediction, target)

        # Both metrics below are spreads *across the batch*, so both are identically zero
        # for a single window -- not because anything collapsed, but because one sample has
        # no spread. `loss_over_null` would then read as the loss over a clamped 1e-8,
        # order 1e8. That matters more than it looks: this is the metric
        # `config/pretrain_jepa.yaml` selects checkpoints by, and a validation loader does
        # not drop its short trailing batch, so a val set of size 200k+1 would let one
        # window swamp the epoch mean and hand selection back to whichever epoch had the
        # smallest val/loss -- the collapse-selecting behaviour monitoring this instead of
        # val/loss exists to avoid. An undefined metric is reported as absent, not as a
        # number, so Lightning averages only the batches that could actually measure it.
        if x.size(0) < 2:
            return loss, {}

        # What a predictor that ignored its input entirely would score: emit the batch mean
        # for every window, and the squared error left over is the targets' own variance.
        null_loss = target.var(dim=0, unbiased=False).mean()

        return loss, {
            # Spread of the teacher's target vectors across the batch, on the tokens the
            # loss actually scores. If these stop varying, the loss reaches zero while the
            # encoder has learned nothing.
            "embed_std": target.std(dim=0, unbiased=False).mean(),
            # The number to read, because under collapse both the loss and the null
            # baseline fall and only their ratio says which is happening. 1.0 means the
            # predictor is doing exactly as well as emitting the mean -- learning nothing,
            # however small the loss has become. Below 1 is real prediction.
            "loss_over_null": loss / null_loss.clamp(min=1e-8),
        }

    def training_step(self, x: dict, _):
        loss, metrics = self.step(x["features"])
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True)
        return loss

    def validation_step(self, x: dict, _):
        loss, metrics = self.step(x["features"])
        self.log("val/loss", loss, prog_bar=True)
        # `val/loss_over_null` is the monitored metric, so it has to be logged on every
        # validation epoch or ModelCheckpoint has nothing to rank.
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)

    def trainable_parameters(self):
        """The teacher is an EMA copy, not an optimized module."""
        return [*self.encoder_student.parameters(), *self.predictor.parameters()]

    def train(self, mode: bool = True) -> "JEPA":
        """Keep the teacher in eval mode; Lightning will not do it for you.

        ``requires_grad_(False)`` stops gradients but not dropout, and not a norm layer's
        running statistics. A teacher that drops units while computing the target makes the
        target stochastic, so the student is asked to predict a different vector each time
        it sees the same window.
        """

        super().train(mode)
        self.encoder_teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, ema: float) -> None:
        params = zip(
            self.encoder_student.parameters(), self.encoder_teacher.parameters(), strict=True
        )
        for student_param, teacher_param in params:
            teacher_param.data.mul_(ema).add_(student_param.data, alpha=1.0 - ema)
        buffers = zip(self.encoder_student.buffers(), self.encoder_teacher.buffers(), strict=True)
        for student_buffer, teacher_buffer in buffers:
            teacher_buffer.copy_(student_buffer)

    def ema_at(self, progress: float) -> float:
        """Cosine ramp from ``ema_start`` to ``ema_stop`` over ``progress`` in [0, 1].

        Cosine rather than linear because that is the schedule S-JEPA uses; only the
        endpoints were rescaled for this repo's run length.
        """

        phase = (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress)))) / 2.0
        return self.ema_stop - (self.ema_stop - self.ema_start) * phase

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        total = max(1, int(self.trainer.estimated_stepping_batches))
        ema = self.ema_at(float(self.trainer.global_step) / float(total))
        self.update_teacher(ema)
        self.log("train/ema", ema, batch_size=batch["features"].shape[0])

    def configure_optimizers(self):  # type: ignore
        total_steps = int(self.trainer.estimated_stepping_batches)
        optimizer = torch.optim.AdamW(
            # Not `self.parameters()`: that sweeps the teacher in, and AdamW's decoupled
            # weight decay would shrink a module whose whole contract is to be an average
            # of the student.
            self.trainable_parameters(),
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            lr=self.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    min_factor=self.min_lr_ratio,
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
