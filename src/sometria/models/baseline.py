"""
Standard masked autoencoder for biomechanical DOF sequences.
 
Random masking, mask-token decoder with full self-attention, MSE on masked
tokens.
 
Defaults: 43 DOFs at 60Hz, T=240 frames (4.0s), l=8 frames per patch (133ms)
-> Te=30, N=1290 tokens. C=5 channels laid out as [sin, cos, vel, acc, tau].
 
channels_input / channels_output pick encoder input and reconstruction target from the
channel stack independently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from sometria.architecture.scheduler import lr_schedule


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, T, V, C) -> (B, T//p, V, p*C). Groups p consecutive frames per DOF."""
    B, T, V, C = x.shape
    assert T % p == 0, f"T={T} not divisible by patch length {p}"
    return x.view(B, T // p, p, V, C).permute(0, 1, 3, 2, 4).reshape(B, T // p, V, p * C)


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, N, D) gathered along N by idx (B, K) -> (B, K, D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def random_mask(num_tokens: int, mask_ratio: float, batch: int, device: torch.device):
    """Uniform random masking. Returns (keep_idx, mask_idx)."""
    n_keep = int(num_tokens * (1.0 - mask_ratio))
    order = torch.rand(batch, num_tokens, device=device).argsort(dim=-1)
    return order[:, :n_keep], order[:, n_keep:]


def _encode_layers(dim: int, heads: int, mlp_ratio: int, depth: int) -> nn.Module:
    return nn.TransformerEncoder(
        num_layers=depth,
        norm=nn.LayerNorm(dim),
        enable_nested_tensor=False,
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=(dim * mlp_ratio), 
            dropout=0.0, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
    )

class Encoder(nn.Module):
    def __init__(
        self,
        num_dofs: int,
        num_frames_in_patch: int,
        dim: int,
        depth: int,
        heads: int,
        channels_input: tuple[int, ...],
        mlp_ratio: int,
        max_te: int = 30,
    ) -> None:
        super().__init__()

        self.num_dofs = num_dofs
        self.num_frames_in_patch = num_frames_in_patch
        self.dim = dim

        self.proj = nn.Linear(num_frames_in_patch * len(channels_input), dim)
        self.blocks = _encode_layers(dim, heads, mlp_ratio, depth)

        # Positional embeddings
        self.pos_s = nn.Parameter(torch.zeros(1, 1, num_dofs, dim))
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        self.pos_t = nn.Parameter(torch.zeros(1, max_te, 1, dim))
        nn.init.trunc_normal_(self.pos_t, std=0.02)

    def pad(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.unsqueeze(-1).expand(-1, -1, self.num_dofs).flatten(1)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None, padding: torch.Tensor | None) -> torch.Tensor:

        e = self.proj(patchify(x, self.num_frames_in_patch))
        e = e + self.pos_s + self.pos_t[:, :e.size(1)]
        e = e.flatten(1, 2)

        token_pad = self.pad(padding) if padding is not None else None
        if keep is not None:
            e = gather_tokens(e, keep)
            if token_pad is not None:
                token_pad = torch.gather(token_pad, 1, keep)

        return self.blocks(e, src_key_padding_mask=token_pad)


class Decoder(nn.Module):
    def __init__(
        self,
        num_dofs: int,
        num_frames_in_patch: int,
        max_te: int,
        channels_output: tuple[int, ...],
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()

        self.num_dofs = num_dofs

        self.proj = nn.Linear(dim, num_frames_in_patch * len(channels_output))
        self.blocks = _encode_layers(dim, heads, mlp_ratio, depth)

        # Target tokens to generate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings
        self.pos_s = nn.Parameter(torch.zeros(1, 1, num_dofs, dim))
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        self.pos_t = nn.Parameter(torch.zeros(1, max_te, 1, dim))
        nn.init.trunc_normal_(self.pos_t, std=0.02)

    def forward(self, h: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:

        B, _, D = h.shape
        pos = (self.pos_s + self.pos_t).flatten(1, 2)

        x = self.mask_token.expand(B, pos.size(1), D).clone()
        x = x.scatter(1, keep.unsqueeze(-1).expand(-1, -1, D), h)

        return self.proj(self.blocks(x + pos))


class MAE(L.LightningModule):
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
        channels_input: tuple[int, ...] = (0, 1, 2, 3, 4),
        channels_output: tuple[int, ...] = (0, 1, 2, 3, 4),
        num_heads: int = 8,
        dim: int = 256,
        lr: float = 1e-3,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        self.mask_ratio = mask_ratio
        self.num_frames_in_patch = num_frames_in_patch
        self.channels_output = channels_output
        self.channels_input = channels_input
        self.weight_decay = weight_decay
        self.min_lr_ratio = min_lr_ratio
        self.warmup = warmup
        self.lr = lr

        Te = num_frames // num_frames_in_patch
        self.num_tokens = Te * num_dofs

        self.encoder = Encoder(num_dofs, num_frames_in_patch, dim, enc_depth, num_heads, channels_input, mlp_ratio, Te)
        self.decoder = Decoder(num_dofs, num_frames_in_patch, Te, channels_output, dim, dec_depth, num_heads, mlp_ratio)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, C) with full channels -> (B, N, E)"""
        return self.encoder(x[..., self.channels_input], None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, C) with full channels -> scalar loss"""

        keep_idx, mask_idx = random_mask(self.num_tokens, self.mask_ratio, x.size(0), x.device)

        h = self.encoder(x[..., self.channels_input], keep_idx, None)
        prediction = self.decoder(h, keep_idx)

        target = patchify(x[..., self.channels_output], self.num_frames_in_patch).flatten(1, 2)

        # Per-token normalization: the loss asks for the shape of a patch,
        # not its magnitude. Without this, high-variance channels (acc, tau)
        # dominate the gradient and the model learns noise.
        target_mean = target.mean(dim=-1, keepdim=True)
        target_var = target.var(dim=-1, keepdim=True)
        target_norm = (target - target_mean) / (target_var + 1e-6).sqrt()
        return F.mse_loss(
            gather_tokens(prediction, mask_idx), 
            gather_tokens(target_norm, mask_idx)
        )

    def training_step(self, x: dict, _):
        loss = self.forward(x["features"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, x: dict, _):
        self.log("val/loss", self.forward(x["features"]), prog_bar=True)

    def configure_optimizers(self): # type: ignore
        total_steps = int(self.trainer.estimated_stepping_batches)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.weight_decay, 
            betas=(0.9, 0.95),
            lr=self.lr
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
        
