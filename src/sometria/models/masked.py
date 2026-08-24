"""Masked reconstruction over motion tokens: MAE and MAMP as one objective.

The encoder sees context tokens only. A shallow decoder receives the encoded context
plus a learned mask token at every target position and predicts the held-out patch
values. Everything architectural belongs to the backbone; what lives here is what is
hidden, what is predicted, and what the loss is.

MAE and MAMP are not two models. They are this one at two configurations:

- ``target="values"``, ``tau <= 0``  -- uniform masking, reconstruct the input: MAE.
- ``target="motion"``, ``tau > 0``   -- motion-aware masking, predict the temporal
  difference of the input: MAMP.

Both follow the reference (Mao et al., https://github.com/maoyunyao/MAMP) in taking the
motion target by differencing *the input the encoder sees*, and in normalizing each
target token to zero mean and unit variance before the loss (``norm_skes_loss`` there,
``norm_targets`` here). An earlier version selected the stored ``vel`` channel instead,
on the theory that a stored velocity makes the difference a channel selection rather than
a computation. It does not: that channel is signed-log compressed and normalized per DOF,
so its per-frame values are near-unpredictable from context, and the objective explained
7.7% of its target's variance in 10 epochs while the pose objective explained 98.8%.
"""

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
from sometria.masking import (
    MaskIndices,
    extract_motion,
    motion_aware_mask,
    patchify,
    token_validity,
)

TARGETS = ("values", "motion")


class MaskedMotionAutoencoder(L.LightningModule):
    """Predict the values of masked motion tokens from the ones left visible."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | dict | None = None,
        *,
        decoder_depth: int = 5,
        mask_ratio: float = 0.90,
        # The defaults are the MAMP configuration, at the reference's values. MAE is
        # target="values" with tau <= 0; both baselines set every knob in their config.
        tau: float = 0.80,
        target: str = "motion",
        motion_stride: int = 1,
        score_channels: tuple[int, ...] = (2,),      # representation.indices("vel")
        loss_channels: tuple[int, ...] = (0, 1),     # representation.indices("sin", "cos")
        norm_targets: bool = True,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()

        backbone = as_backbone(backbone)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1 for masked reconstruction.")
        if tau > 0 and not score_channels:
            raise ValueError("motion-aware masking needs at least one score channel.")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        if not loss_channels:
            raise ValueError("the loss needs at least one channel to score.")

        # The spec, never the module: hparams have to survive a checkpoint round trip, and
        # this is what lets load_from_checkpoint(path) rebuild the backbone unaided.
        self.save_hyperparameters(
            {
                "backbone": backbone_hparam(backbone),
                "decoder_depth": decoder_depth,
                "mask_ratio": mask_ratio,
                "tau": tau,
                "target": target,
                "motion_stride": motion_stride,
                "score_channels": tuple(score_channels),
                "loss_channels": tuple(loss_channels),
                "norm_targets": norm_targets,
                "lr": lr,
                "min_lr_frac": min_lr_frac,
                "weight_decay": weight_decay,
                "warmup_frac": warmup_frac,
            }
        )

        self.backbone = backbone
        spec = backbone.spec
        self.mask_ratio = mask_ratio
        self.tau = tau
        self.target = target
        self.motion_stride = motion_stride
        self.score_channels = tuple(score_channels)
        self.norm_targets = norm_targets
        self.lr = lr
        self.min_lr_frac = min_lr_frac
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac

        time_patches, num_dofs = spec.grid_shape
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.decoder_position = PositionalEncoding(time_patches, num_dofs, spec.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.num_heads,
            dim_feedforward=int(spec.d_model * spec.mlp_ratio),
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=decoder_depth, norm=nn.LayerNorm(spec.d_model), enable_nested_tensor=False
        )
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
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, MaskIndices]:
        """Return ``(prediction, target, target_valid, mask)`` over target tokens only."""

        spec = self.backbone.spec
        patches = self.backbone.time_patches(features.shape[1])
        # One patchify feeds the mask, which scores it, and the encoder, which projects
        # it. A motion target needs its own, over a differenced copy of the window.
        tokens_in = patchify(features, spec.patch_size)
        values = tokens_in.flatten(start_dim=-2)
        batch, length, _ = values.shape

        # The encoder always reads the input; only what the decoder is asked for changes.
        # extract_motion runs on the window, not on the patches, so the difference at a
        # patch boundary is a real one rather than being zeroed at every 8th frame.
        target_source = (
            tokens_in
            if self.target == "values"
            else patchify(extract_motion(features, self.motion_stride), spec.patch_size)
        )

        mask = motion_aware_mask(
            tokens_in,
            score_channels=self.score_channels,
            mask_ratio=self.mask_ratio,
            tau=self.tau,
            valid=valid,
            generator=generator,
        )

        encoded = self.backbone.embed_values(values, patches, index=mask.context)

        tokens = self.mask_token.expand(batch, length, -1).clone()
        tokens.scatter_(
            dim=1,
            index=mask.context.unsqueeze(-1).expand(-1, -1, spec.d_model),
            src=encoded,
        )
        decoded = self.decoder(tokens + self.decoder_position.get_flat(patches))
        prediction = self.prediction(decoded)

        target_valid = (
            t.ones(mask.targets.shape, dtype=t.bool, device=values.device)
            if valid is None
            else _gather(token_validity(valid, spec.patch_size, length).to(values.device), mask.targets)
        )
        target = target_source[..., self.loss_channel_index].flatten(start_dim=-2)
        return _gather(prediction, mask.targets), _gather(target, mask.targets), target_valid, mask

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        """``prediction`` and ``target`` are both ``(B, N, patch_size * len(loss_channels))``."""

        if self.norm_targets:
            # MAMP's norm_skes_loss. Each target token is standardized over its own
            # values, so the loss asks for the *shape* of the patch rather than its
            # magnitude. Without it a squared error on motion is dominated by the few
            # fastest tokens -- which motion-aware masking has deliberately selected for.
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if not target_valid.any():
            return (prediction - target).square().mean() * 0.0

        # Mean per token, then mean over target tokens: a token counts once whatever its
        # width, which is what makes the number comparable across loss_channels.
        per_token = (prediction - target).square().mean(dim=-1)
        return (per_token * target_valid).sum() / target_valid.sum()

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        prediction, target, target_valid, mask = self(batch["features"], batch.get("valid"))
        loss = self.reconstruction_loss(prediction, target, target_valid)
        batch_size = batch["features"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/context_tokens", float(mask.context.shape[1]), batch_size=batch_size)
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


def _gather(x: t.Tensor, idx: t.Tensor) -> t.Tensor:
    """Select tokens by flat index, whether or not they carry a feature axis."""

    if x.ndim == 2:
        return x.gather(dim=1, index=idx)
    return x.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
