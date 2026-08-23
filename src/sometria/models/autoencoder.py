
import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.scheduler import lr_schedule
from sometria.masking import MaskIndices, motion_aware_mask, patchify

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


class MotionMaskedAutoencoder(L.LightningModule):
    """Masked patch reconstruction over motion tokens.

    The encoder sees only context tokens. A shallow decoder receives encoded context
    tokens plus learned mask tokens at target positions and reconstructs the held-out
    patch values. This is intentionally a reconstruction objective, not JEPA: it keeps
    the transformer/masking path small enough to debug before adding a target encoder.
    """

    def __init__(
        self,
        num_dofs: int = 43,
        num_features: int = 5,
        patch_size: int = 8,
        max_time_patches: int = 64,
        d_model: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mask_ratio: float = 0.80,
        mask_strategy: str = "random",
        tau: float = 0.25,
        score_channels: tuple[int, ...] = (2,),
        loss_channels: tuple[int, ...] = (2, 3, 4),
        lr: float = 1e-3,
        warmup_frac: float = 0.03,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1 for masked reconstruction.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if mask_strategy not in {"random", "mamp"}:
            raise ValueError("mask_strategy must be either 'random' or 'mamp'.")
        if mask_strategy == "mamp" and len(score_channels) == 0:
            raise ValueError("mamp masking requires at least one score channel.")

        self.num_dofs = num_dofs
        self.num_features = num_features
        self.patch_size = patch_size
        self.max_time_patches = max_time_patches
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.tau = tau
        self.score_channels = tuple(score_channels)
        self.lr = lr
        self.warmup_frac = warmup_frac

        token_dim = patch_size * num_features
        hidden_dim = int(d_model * mlp_ratio)

        self.patch_projection = nn.Linear(token_dim, d_model)
        self.encoder_time_encoding = nn.Parameter(t.zeros(1, max_time_patches, 1, d_model))
        self.encoder_dof_encoding = nn.Parameter(t.zeros(1, 1, num_dofs, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.mask_token = nn.Parameter(t.zeros(1, 1, d_model))
        self.decoder_time_encoding = nn.Parameter(t.zeros(1, max_time_patches, 1, d_model))
        self.decoder_dof_encoding = nn.Parameter(t.zeros(1, 1, num_dofs, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.prediction = nn.Linear(d_model, token_dim)

        loss_mask = t.zeros(num_features, dtype=t.bool)
        loss_mask[list(loss_channels)] = True
        self.register_buffer("loss_channel_mask", loss_mask)

        nn.init.trunc_normal_(self.encoder_time_encoding, std=0.02)
        nn.init.trunc_normal_(self.encoder_dof_encoding, std=0.02)
        nn.init.trunc_normal_(self.decoder_time_encoding, std=0.02)
        nn.init.trunc_normal_(self.decoder_dof_encoding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        features: t.Tensor,
        valid: t.Tensor | None = None,
        generator: t.Generator | None = None,
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, MaskIndices]:
        batch, frames, dofs, channels = features.shape
        if dofs != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} DoFs, got {dofs}.")
        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {channels}.")

        patches = patchify(features, self.patch_size)
        values = patches.flatten(start_dim=-2)
        num_time_patches = frames // self.patch_size
        if num_time_patches > self.max_time_patches:
            raise ValueError(
                f"window needs {num_time_patches} time patches, "
                f"but max_time_patches is {self.max_time_patches}."
            )

        mask = self._mask(patches, valid=valid, generator=generator)
        encoder_pos = self._flat_pos(
            num_time_patches,
            self.encoder_time_encoding,
            self.encoder_dof_encoding,
        )
        decoder_pos = self._flat_pos(
            num_time_patches,
            self.decoder_time_encoding,
            self.decoder_dof_encoding,
        )

        tokens = self.patch_projection(values) + encoder_pos
        context_tokens = self._gather(tokens, mask.context)
        encoded_context = self.encoder(context_tokens)

        decoder_tokens = self.mask_token.expand(batch, values.shape[1], -1).clone()
        decoder_tokens.scatter_(
            dim=1,
            index=mask.context.unsqueeze(-1).expand(-1, -1, self.d_model),
            src=encoded_context,
        )
        decoded = self.decoder(decoder_tokens + decoder_pos)
        prediction = self.prediction(decoded)

        target_prediction = self._gather(prediction, mask.targets)
        target_values = self._gather(values, mask.targets)
        target_valid = self._target_valid(valid, num_time_patches, mask.targets, values.device)
        return target_prediction, target_values, target_valid, mask

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        prediction = prediction.reshape(*prediction.shape[:2], self.patch_size, self.num_features)
        target = target.reshape(*target.shape[:2], self.patch_size, self.num_features)

        token_mask = target_valid[:, :, None, None]
        channel_mask = self.loss_channel_mask[None, None, None, :]
        mask = token_mask & channel_mask
        squared_error = (prediction - target).square()
        if not mask.any():
            return squared_error.mean() * 0.0
        return squared_error.masked_select(mask).mean()

    def _step(self, batch: dict, stage: str) -> t.Tensor:
        prediction, target, target_valid, mask = self(batch["features"], batch.get("valid"))
        loss = self.reconstruction_loss(prediction, target, target_valid)
        batch_size = batch["features"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/context_tokens", mask.context.shape[1], batch_size=batch_size)
        self.log(f"{stage}/target_tokens", mask.targets.shape[1], batch_size=batch_size)
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

    def _mask(
        self,
        patches: t.Tensor,
        *,
        valid: t.Tensor | None,
        generator: t.Generator | None,
    ) -> MaskIndices:
        if self.mask_strategy == "mamp":
            return motion_aware_mask(
                patches,
                score_channels=self.score_channels,
                mask_ratio=self.mask_ratio,
                tau=self.tau,
                valid=valid,
                generator=generator,
            )

        batch, length = patches.shape[:2]
        noise = t.rand(batch, length, device=patches.device, generator=generator)
        if valid is not None:
            usable = self._usable_tokens(valid, patches.shape[2], length).to(patches.device)
            noise = noise.masked_fill(~usable, float("inf"))
        order = noise.argsort(dim=-1)
        len_keep = round(length * (1.0 - self.mask_ratio))
        return MaskIndices(context=order[:, :len_keep], targets=order[:, len_keep:])

    def _flat_pos(
        self,
        num_time_patches: int,
        time_encoding: t.Tensor,
        dof_encoding: t.Tensor,
    ) -> t.Tensor:
        return (time_encoding[:, :num_time_patches] + dof_encoding).flatten(1, 2)

    def _target_valid(
        self,
        valid: t.Tensor | None,
        num_time_patches: int,
        target_idx: t.Tensor,
        device: t.device,
    ) -> t.Tensor:
        if valid is None:
            return t.ones(target_idx.shape, dtype=t.bool, device=device)
        usable = self._usable_tokens(valid, self.patch_size, num_time_patches * self.num_dofs)
        return self._gather(usable.to(device), target_idx)

    def _usable_tokens(self, valid: t.Tensor, patch_size: int, length: int) -> t.Tensor:
        if valid.shape[1] % patch_size != 0:
            raise ValueError(
                f"valid covers {valid.shape[1]} frames, not divisible by patch_size {patch_size}"
            )
        usable = valid.reshape(valid.shape[0], -1, patch_size).all(dim=-1)
        return usable.repeat_interleave(length // usable.shape[1], dim=1)

    def _gather(self, x: t.Tensor, idx: t.Tensor) -> t.Tensor:
        if x.ndim == 2:
            return x.gather(dim=1, index=idx)
        idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return x.gather(dim=1, index=idx)
