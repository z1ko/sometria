"""Masked reconstruction over motion tokens: MAE and MAMP as one objective.

The encoder sees context tokens only. A shallow decoder receives the encoded context
plus a learned mask token at every target position and predicts the held-out patch
values. Everything architectural belongs to the backbone; what lives here is what is
hidden, what is predicted, and what the loss is.

MAE and MAMP are not two models. They are this one at two configurations:

- ``tau <= 0``  -- uniform random masking, reconstruction of pose channels: MAE.
- ``tau > 0``   -- motion-aware masking, loss on ``vel``: MAMP.

The reference differences raw joint coordinates to build a motion target because motion
is absent from its input. Here ``vel`` is a stored channel, so the difference is a
channel selection (``loss_channels``) rather than a computation -- and differencing the
normalized, signed-log-compressed values would not reproduce it anyway.
"""

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.architecture.pos_embed import PositionalEncoding
from sometria.architecture.scheduler import lr_schedule
from sometria.masking import MaskIndices, motion_aware_mask, patchify, token_validity


class MaskedMotionAutoencoder(L.LightningModule):
    """Predict the values of masked motion tokens from the ones left visible."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | None = None,
        *,
        decoder_depth: int = 2,
        mask_ratio: float = 0.80,
        # The defaults are the MAMP configuration. MAE is tau <= 0 with the pose
        # channels as its target; both baselines set all three explicitly in their config.
        tau: float = 0.25,
        score_channels: tuple[int, ...] = (2,),      # representation.indices("vel")
        loss_channels: tuple[int, ...] = (2,),       # representation.indices("vel")
        lr: float = 1e-3,
        warmup_frac: float = 0.03,
    ) -> None:
        super().__init__()

        if isinstance(backbone, EncoderSpec) or backbone is None:
            backbone = MotionTransformerEncoder(backbone)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1 for masked reconstruction.")
        if tau > 0 and not score_channels:
            raise ValueError("motion-aware masking needs at least one score channel.")

        # The spec, never the module: hparams have to survive a checkpoint round trip, and
        # this is what lets load_from_checkpoint(path) rebuild the backbone unaided.
        self.save_hyperparameters(
            {
                "backbone": backbone.spec,
                "decoder_depth": decoder_depth,
                "mask_ratio": mask_ratio,
                "tau": tau,
                "score_channels": tuple(score_channels),
                "loss_channels": tuple(loss_channels),
                "lr": lr,
                "warmup_frac": warmup_frac,
            }
        )

        self.backbone = backbone
        spec = backbone.spec
        self.mask_ratio = mask_ratio
        self.tau = tau
        self.score_channels = tuple(score_channels)
        self.lr = lr
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
        self.prediction = nn.Linear(spec.d_model, spec.token_dim)

        loss_mask = t.zeros(spec.num_features, dtype=t.bool)
        loss_mask[list(loss_channels)] = True
        self.register_buffer("loss_channel_mask", loss_mask)

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
        # One patchify feeds all three consumers: the mask scores it, the encoder
        # projects it, and the loss compares against it.
        tokens_in = patchify(features, spec.patch_size)
        values = tokens_in.flatten(start_dim=-2)
        batch, length, _ = values.shape

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
        return _gather(prediction, mask.targets), _gather(values, mask.targets), target_valid, mask

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        spec = self.backbone.spec
        shape = (*prediction.shape[:2], spec.patch_size, spec.num_features)
        prediction = prediction.reshape(shape)
        target = target.reshape(shape)

        mask = target_valid[:, :, None, None] & self.loss_channel_mask[None, None, None, :]
        squared_error = (prediction - target).square()
        if not mask.any():
            return squared_error.mean() * 0.0
        return squared_error.masked_select(mask).mean()

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
        optimizer = t.optim.AdamW(self.parameters(), lr=self.lr)
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


def _gather(x: t.Tensor, idx: t.Tensor) -> t.Tensor:
    """Select tokens by flat index, whether or not they carry a feature axis."""

    if x.ndim == 2:
        return x.gather(dim=1, index=idx)
    return x.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
