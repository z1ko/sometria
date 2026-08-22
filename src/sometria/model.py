"""Small baseline models for validating the motion training pipeline."""

import lightning as L
import torch as t
import torch.nn as nn


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
        depth: int = 3,
        kernel_size: int = 5,
        lr: float = 1e-3,
        loss_channels: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_dofs = num_dofs
        self.num_features = num_features
        self.lr = lr
        in_channels = num_dofs * num_features
        padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                nn.GELU(),
            ]
        layers.append(nn.Conv1d(hidden_channels, in_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

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
        y = self.net(x)
        return y.transpose(1, 2).reshape(batch, frames, dofs, channels)

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
        return t.optim.AdamW(self.parameters(), lr=self.lr)
