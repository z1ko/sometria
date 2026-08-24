"""Masked reconstruction over motion tokens: MAE and MAMP as one objective.

The encoder sees context tokens only. A shallow decoder receives the encoded context
plus a learned mask token at every target position and predicts the held-out patch
values. Everything architectural belongs to the backbone, and splitting the window into
**Context** and **Target** belongs to :mod:`sometria.models.window`; what lives here is
what is predicted and what the loss is.

MAE and MAMP are not two models. They are this one at two configurations:

- ``target="values"``, ``MaskSpec(tau=0.0)``  -- uniform masking, reconstruct the input: MAE.
- ``target="motion"``, ``MaskSpec(tau=0.8)``  -- motion-aware masking, predict the temporal
  difference of the input: MAMP.

Both follow the reference (Mao et al., https://github.com/maoyunyao/MAMP) in taking the
motion target by differencing *the input the encoder sees*, and in normalizing each
target token to zero mean and unit variance before the loss (``norm_skes_loss`` there,
``standardize_tokens`` here). An earlier version selected the stored ``vel`` channel
instead, on the theory that a stored velocity makes the difference a channel selection
rather than a computation. It does not: that channel is signed-log compressed and
normalized per DOF, so its per-frame values are near-unpredictable from context, and the
objective explained 7.7% of its target's variance in 10 epochs while the pose objective
explained 98.8%.
"""

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
from sometria.masking import MaskSpec, as_mask_spec, extract_motion, patchify
from sometria.models.window import (
    MaskedWindow,
    mask_window,
    masked_token_mse,
    standardize_tokens,
)

TARGETS = ("values", "motion")

# The reference's MAMP masking. MAE is the same at tau <= 0; both baselines set every
# knob in their config, so this default is what an unconfigured model gets, not a policy.
DEFAULT_MASK = MaskSpec(mask_ratio=0.90, tau=0.80, score_channels=(2,))


class MaskedMotionAutoencoder(L.LightningModule):
    """Predict the values of masked motion tokens from the ones left visible."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | dict | None = None,
        mask: MaskSpec | dict | None = None,
        *,
        decoder_depth: int = 5,
        # The defaults are the MAMP configuration, at the reference's values.
        target: str = "motion",
        motion_stride: int = 1,
        loss_channels: tuple[int, ...] = (0, 1),     # representation.indices("sin", "cos")
        norm_targets: bool = True,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()

        backbone = as_backbone(backbone)
        mask = as_mask_spec(mask, DEFAULT_MASK)
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        if not loss_channels:
            raise ValueError("the loss needs at least one channel to score.")

        # The specs, never the modules: hparams have to survive a checkpoint round trip,
        # and this is what lets load_from_checkpoint(path) rebuild them unaided.
        self.save_hyperparameters(
            {
                "backbone": backbone_hparam(backbone),
                "mask": asdict(mask),
                "decoder_depth": decoder_depth,
                "target": target,
                "motion_stride": motion_stride,
                "loss_channels": tuple(loss_channels),
                "norm_targets": norm_targets,
                "lr": lr,
                "min_lr_frac": min_lr_frac,
                "weight_decay": weight_decay,
                "warmup_frac": warmup_frac,
            }
        )

        self.backbone = backbone
        self.mask = mask
        spec = backbone.spec
        self.target = target
        self.motion_stride = motion_stride
        self.norm_targets = norm_targets
        self.lr = lr
        self.min_lr_frac = min_lr_frac
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac

        time_patches, num_dofs = spec.grid_shape
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.decoder_position = PositionalEncoding(time_patches, num_dofs, spec.d_model)
        self.decoder = transformer_stack(spec, decoder_depth)

        # The head is exactly as wide as the target, as in the reference: predicting
        # channels no loss scores would be parameters that never receive a gradient.
        self.loss_channels = tuple(loss_channels)
        self.register_buffer("loss_channel_index", t.tensor(self.loss_channels, dtype=t.long))
        self.prediction = nn.Linear(spec.d_model, spec.patch_size * len(self.loss_channels))

        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        features: t.Tensor,
        valid: t.Tensor | None = None,
        generator: t.Generator | None = None,
    ) -> tuple[t.Tensor, t.Tensor, MaskedWindow]:
        """Return ``(prediction, target, window)``, the first two over target tokens only."""

        spec = self.backbone.spec
        window = mask_window(features, spec, self.mask, valid=valid, generator=generator)

        # The encoder always reads the input; only what the decoder is asked for changes.
        # extract_motion runs on the window, not on the patches, so the difference at a
        # patch boundary is a real one rather than being zeroed at every 8th frame.
        target_source = (
            window.patches
            if self.target == "values"
            else patchify(extract_motion(features, self.motion_stride), spec.patch_size)
        )

        encoded = self.backbone.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )

        batch, length, _ = window.values.shape
        tokens = self.mask_token.expand(batch, length, -1).clone()
        tokens.scatter_(
            dim=1,
            index=window.mask.context.unsqueeze(-1).expand(-1, -1, spec.d_model),
            src=encoded,
        )
        decoded = self.decoder(tokens + self.decoder_position.get_flat(window.num_time_patches))
        prediction = self.prediction(decoded)

        target = target_source[..., self.loss_channel_index].flatten(start_dim=-2)
        return window.mask.targets_of(prediction), window.mask.targets_of(target), window

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        """``prediction`` and ``target`` are both ``(B, N, patch_size * len(loss_channels))``."""

        if self.norm_targets:
            target = standardize_tokens(target)
        return masked_token_mse(prediction, target, target_valid)

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        prediction, target, window = self(batch["features"], batch.get("valid"))
        loss = self.reconstruction_loss(prediction, target, window.target_valid)
        batch_size = batch["features"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/context_tokens", float(window.mask.context.shape[1]), batch_size=batch_size)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self): # type: ignore
        # betas and weight decay follow the reference's AdamW; its cosine decays to
        # min_lr = lr / 2 rather than to zero, which min_lr_frac carries.
        optimizer = t.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay
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
