"""The backbone: a window in, tokens out, and nothing else.

Patch projection, positional encoding, transformer blocks, final norm. No masking
strategy, no decoder, no mask token, no prediction head, no Lightning -- those belong to
whichever pretext objective owns this module. Every objective owns one backbone (JEPA
will own two, which is why "the encoder" is not a usable name here).

The architecture travels as an :class:`EncoderSpec`, not as a module. That is what a
YAML ``encoder:`` block maps to and what a checkpoint's hparams carry, so a downstream
classifier reloads without being told the architecture a second time.
"""

from dataclasses import asdict, dataclass

import torch as t
import torch.nn as nn

from sometria.architecture.pos_embed import PositionalEncoding
from sometria.masking import patchify, token_validity

POOLINGS = ("window", "dof")


@dataclass(frozen=True)
class EncoderSpec:
    """Everything needed to rebuild a :class:`MotionTransformerEncoder`.

    Frozen and picklable on purpose: it is stored in a ``LightningModule``'s
    hyperparameters, so ``load_from_checkpoint(path)`` works with no extra arguments.
    """

    num_dofs: int = 43
    num_features: int = 5
    patch_size: int = 8
    window_frames: int = 240
    d_model: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.window_frames % self.patch_size != 0:
            raise ValueError(
                f"window of {self.window_frames} frames is not divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(time patches, DOFs)`` for a full-length window."""

        return (self.window_frames // self.patch_size, self.num_dofs)

    @property
    def token_dim(self) -> int:
        """Values per token: one DOF over ``patch_size`` frames, all channels."""

        return self.patch_size * self.num_features

    def pooled_dim(self, pool: str) -> int:
        """Width of :meth:`MotionTransformerEncoder.embed` under one pooling."""

        if pool == "window":
            return self.d_model
        if pool == "dof":
            return self.num_dofs * self.d_model
        raise ValueError(f"pool must be one of {POOLINGS}, got {pool!r}")


def as_backbone(
    backbone: "MotionTransformerEncoder | EncoderSpec | dict | None",
) -> "MotionTransformerEncoder":
    """Accept a built backbone, a spec, or a spec's fields, and return a backbone.

    The dict form is what comes back out of a checkpoint. Hyperparameters are stored as
    plain fields rather than as the dataclass itself: Lightning refuses to log a frozen
    dataclass ("A frozen dataclass was passed to `apply_to_collection`"), and
    ``torch.load`` defaults to ``weights_only=True``, which rejects any class it has not
    been told about. A dict travels through both, and through a YAML ``encoder:`` block.
    """

    if isinstance(backbone, MotionTransformerEncoder):
        return backbone
    if isinstance(backbone, dict):
        return MotionTransformerEncoder(EncoderSpec(**backbone))
    return MotionTransformerEncoder(backbone)


def backbone_hparam(backbone: "MotionTransformerEncoder") -> dict:
    """The backbone's spec as plain fields, ready for ``save_hyperparameters``."""

    return asdict(backbone.spec)


class MotionTransformerEncoder(nn.Module):
    """Tokenize a window and contextualize its tokens."""

    def __init__(self, spec: EncoderSpec | None = None) -> None:
        super().__init__()

        self.spec = spec or EncoderSpec()
        time_patches, num_dofs = self.spec.grid_shape

        self.projection = nn.Linear(self.spec.token_dim, self.spec.d_model)
        self.position = PositionalEncoding(time_patches, num_dofs, self.spec.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.spec.d_model,
            nhead=self.spec.num_heads,
            dim_feedforward=int(self.spec.d_model * self.spec.mlp_ratio),
            dropout=self.spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first blocks leave the residual stream unnormalized, so the final norm is
        # not optional -- nn.TransformerEncoder only applies one if it is given one.
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=self.spec.depth, norm=nn.LayerNorm(self.spec.d_model), enable_nested_tensor=False
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.spec.grid_shape

    def time_patches(self, frames: int) -> int:
        """How many time patches a window of ``frames`` frames occupies."""

        if frames % self.spec.patch_size != 0:
            raise ValueError(
                f"window of {frames} frames is not divisible by patch_size {self.spec.patch_size}"
            )
        patches = frames // self.spec.patch_size
        if patches > self.spec.grid_shape[0]:
            raise ValueError(
                f"window needs {patches} time patches, but the encoder holds "
                f"{self.spec.grid_shape[0]}"
            )
        return patches

    def token_values(self, features: t.Tensor) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, L, patch_size * C)``, the raw values of every token.

        The token layout in one place: a masked objective's reconstruction target is
        these same numbers, gathered at the target indices.
        """

        _, _, dofs, channels = features.shape
        if dofs != self.spec.num_dofs:
            raise ValueError(f"Expected {self.spec.num_dofs} DOFs, got {dofs}.")
        if channels != self.spec.num_features:
            raise ValueError(f"Expected {self.spec.num_features} features, got {channels}.")
        return patchify(features, self.spec.patch_size).flatten(start_dim=-2)

    def embed_tokens(
        self,
        features: t.Tensor,
        valid: t.Tensor | None = None,
        index: t.Tensor | None = None,
    ) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, L, d_model)``, one contextualized vector per token.

        ``index`` selects a subset of the flat grid before the blocks run -- a masked
        objective passes its context indices here, so the encoder never attends to a
        token it is supposed to be predicting. ``valid`` is the frame mask; invalid
        tokens are hidden from attention. Callers that exclude short samples up front
        (``MotionViewSpec.min_frames``) pass neither.
        """

        return self.embed_values(
            self.token_values(features),
            self.time_patches(features.shape[1]),
            valid=valid,
            index=index,
        )

    def embed_values(
        self,
        values: t.Tensor,
        num_time_patches: int,
        *,
        valid: t.Tensor | None = None,
        index: t.Tensor | None = None,
    ) -> t.Tensor:
        """:meth:`embed_tokens` for a caller that has already patchified.

        A masked objective needs the patches before it can pick a mask, and patchifying
        a 1290-token window twice per step is a whole discarded copy of the batch.
        """

        x = self.projection(values)
        if index is None:
            x = x + self.position.get_flat(num_time_patches)
            padding = None if valid is None else ~self._token_valid(valid, values.shape[1])
        else:
            x = x.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, self.spec.d_model))
            x = x + self.position.gather(index, num_time_patches)
            # A masker forces invalid tokens into its targets, so a context set is
            # already free of padding.
            padding = None

        return self.blocks(x, src_key_padding_mask=padding)

    def embed(
        self,
        features: t.Tensor,
        *,
        pool: str = "window",
        valid: t.Tensor | None = None,
    ) -> t.Tensor:
        """Pooled convenience over :meth:`embed_tokens`.

        ``pool="window"`` means over the whole grid to ``d_model``; ``pool="dof"`` means
        over time only, so the per-joint axis survives into ``num_dofs * d_model``. Both
        average over valid tokens only -- a padded token is not a quiet one.
        """

        if pool not in POOLINGS:
            raise ValueError(f"pool must be one of {POOLINGS}, got {pool!r}")

        tokens = self.embed_tokens(features, valid=valid)
        batch, length, width = tokens.shape
        patches = length // self.spec.num_dofs

        weight = (
            t.ones(batch, length, 1, device=tokens.device, dtype=tokens.dtype)
            if valid is None
            else self._token_valid(valid, length).unsqueeze(-1).to(tokens.dtype)
        )
        tokens = tokens * weight

        if pool == "window":
            return tokens.sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

        grid = tokens.reshape(batch, patches, self.spec.num_dofs, width)
        counts = weight.reshape(batch, patches, self.spec.num_dofs, 1).sum(dim=1).clamp(min=1.0)
        return (grid.sum(dim=1) / counts).flatten(start_dim=1)

    def _token_valid(self, valid: t.Tensor, length: int) -> t.Tensor:
        return token_validity(valid, self.spec.patch_size, length).to(self.position.time_encoding.device)
