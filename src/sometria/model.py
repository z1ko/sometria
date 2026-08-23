"""Small baseline models for validating the motion training pipeline."""

from typing import Any

import lightning as L
import torch as t
import torch.nn as nn


def lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup into cosine decay, stepped per optimizer step."""

    return t.optim.lr_scheduler.SequentialLR(
        optimizer,
        milestones=[warmup_steps],
        schedulers=[
            t.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            ),
            t.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5
            ),
        ],
    )


class MotionConvAutoencoder(L.LightningModule):
    """A minimal temporal Conv1D autoencoder over normalized motion windows.

    Input and output tensors have shape ``(batch, time, dofs, features)``. The
    model flattens ``dofs * features`` into Conv1D channels and predicts a
    same-shaped reconstruction. This is deliberately simple: its job is to prove
    that loading, cropping, normalization, batching, and optimization all work.
    """

    def __init__(
        self,
        num_dofs: int = 43,
        num_features: int = 5,
        hidden_channels: int = 256,
        latent_channels: int | None = None,
        depth: int = 3,
        kernel_size: int = 5,
        lr: float = 1e-3,
        warmup_frac: float = 0.03,
        norm: bool = False,
        loss_channels: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_dofs = num_dofs
        self.num_features = num_features
        self.lr = lr
        self.warmup_frac = warmup_frac
        self.latent_channels = hidden_channels if latent_channels is None else latent_channels
        in_channels = num_dofs * num_features
        padding = kernel_size // 2

        # Off by default: BatchNorm does what it promises -- without it the biases drift until
        # 87% of preactivations sit in GELU's linear tail, with it 77-99% land in [-2, 2] where
        # the nonlinearity is real -- but it measured ~10% worse while the decoder was linear.
        # Worth retesting now that the decoder has depth.
        def block(in_dim: int) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv1d(in_dim, hidden_channels, kernel_size, padding=padding)
            ]
            if norm:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.GELU())
            return layers

        # The decoder has to be nonlinear. With a single 1x1 conv reading the code, every
        # reconstruction lands in a latent_channels-dimensional affine subspace of the input
        # space regardless of how clever the encoder is -- exactly the constraint PCA solves
        # optimally, so the model could match that baseline and provably never beat it.
        encoder = block(in_channels)
        for _ in range(depth - 1):
            encoder += block(hidden_channels)
        encoder.append(nn.Conv1d(hidden_channels, self.latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*encoder)

        decoder = block(self.latent_channels)
        for _ in range(depth - 1):
            decoder += block(hidden_channels)
        decoder.append(nn.Conv1d(hidden_channels, in_channels, kernel_size=1))
        self.decoder = nn.Sequential(*decoder)

        loss_mask = t.zeros(num_features, dtype=t.bool)
        loss_mask[list(loss_channels)] = True
        self.register_buffer("loss_channel_mask", loss_mask)

    def forward(self, features: t.Tensor) -> t.Tensor:
        batch, frames, dofs, channels = features.shape
        if dofs != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} DoFs, got {dofs}.")
        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {channels}.")

        x = features.reshape(batch, frames, dofs * channels).transpose(1, 2)
        y = self.decoder(self.encode(x))
        return y.transpose(1, 2).reshape(batch, frames, dofs, channels)

    def encode(self, x: t.Tensor) -> t.Tensor:
        """Return the ``(batch, latent_channels, frames)`` code for flattened input."""

        return self.encoder(x)

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        valid: t.Tensor,
    ) -> t.Tensor:
        frame_mask = valid[:, :, None, None]
        channel_mask = self.loss_channel_mask[None, None, None, :]
        mask = frame_mask & channel_mask
        squared_error = (prediction - target).square()
        return squared_error.masked_select(mask).mean()

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        features = batch["features"]
        prediction = self(features)
        loss = self.reconstruction_loss(prediction, features, batch["valid"])
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=features.shape[0])
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
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


def mhsa_encoder(
    d_model: int, 
    num_layers: int, 
    num_heads: int, 
    mlp_ratio: float, 
    dropout: float
) -> nn.Module:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        ),
        num_layers=num_layers
    )
