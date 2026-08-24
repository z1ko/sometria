"""The backbone: a window in, tokens out, and nothing else.

Patch projection, positional encoding, transformer blocks, final norm. No masking
strategy, no decoder, no mask token, no prediction head, no Lightning -- those belong to
whichever pretext objective owns this module. Every objective owns one backbone (JEPA
owns two, which is why "the encoder" is not a usable name here).

The architecture travels as an :class:`EncoderSpec`, not as a module. That is what a
YAML ``encoder:`` block maps to and what a checkpoint's hparams carry, so a downstream
classifier reloads without being told the architecture a second time.
"""

from dataclasses import asdict, dataclass

import torch as t
import torch.nn as nn

from sometria.architecture.pos_embed import PositionalEncoding
from sometria.masking import gather_tokens, patchify


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

    def time_patches(self, frames: int) -> int:
        """How many time patches a window of ``frames`` frames occupies.

        On the spec rather than on the module: it reads nothing else, and a masked
        window has to answer it before any backbone is involved.
        """

        if frames % self.patch_size != 0:
            raise ValueError(
                f"window of {frames} frames is not divisible by patch_size {self.patch_size}"
            )
        patches = frames // self.patch_size
        if patches > self.grid_shape[0]:
            raise ValueError(
                f"window needs {patches} time patches, but the encoder holds "
                f"{self.grid_shape[0]}"
            )
        return patches

    def check_features(self, features: t.Tensor) -> None:
        """Raise unless ``(B, T, D, C)`` carries the DOFs and channels this spec expects.

        Checked wherever a window first enters the model -- the backbone's own
        tokenization and a masked window both -- so a mis-shaped feature tensor is named
        rather than surfacing as a matmul error inside the projection.
        """

        _, _, dofs, channels = features.shape
        if dofs != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} DOFs, got {dofs}.")
        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {channels}.")

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


def transformer_stack(spec: EncoderSpec, depth: int) -> nn.TransformerEncoder:
    """``depth`` pre-norm blocks at this spec's width, plus the final norm.

    The backbone, a masked objective's decoder and JEPA's predictor all want the same
    block at the same width and differ only in depth, so the two non-obvious keyword
    arguments are set -- and explained -- in one place.
    """

    layer = nn.TransformerEncoderLayer(
        d_model=spec.d_model,
        nhead=spec.num_heads,
        dim_feedforward=int(spec.d_model * spec.mlp_ratio),
        dropout=spec.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    # norm_first blocks leave the residual stream unnormalized, so the final norm is
    # not optional -- nn.TransformerEncoder only applies one if it is given one.
    # enable_nested_tensor would silently drop to a fast path that ignores norm_first.
    return nn.TransformerEncoder(
        layer, num_layers=depth, norm=nn.LayerNorm(spec.d_model), enable_nested_tensor=False
    )


class MotionTransformerEncoder(nn.Module):
    """Tokenize a window and contextualize its tokens."""

    def __init__(self, spec: EncoderSpec | None = None) -> None:
        super().__init__()

        self.spec = spec or EncoderSpec()
        time_patches, num_dofs = self.spec.grid_shape

        self.projection = nn.Linear(self.spec.token_dim, self.spec.d_model)
        self.position = PositionalEncoding(time_patches, num_dofs, self.spec.d_model)
        self.blocks = transformer_stack(self.spec, self.spec.depth)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.spec.grid_shape

    def token_values(self, features: t.Tensor) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, L, patch_size * C)``, the raw values of every token.

        The token layout in one place: a masked objective's reconstruction target is
        these same numbers, gathered at the target indices.
        """

        self.spec.check_features(features)
        patches = patchify(features, self.spec.patch_size)
        return patches.flatten(start_dim=-2)

    def embed_tokens(
        self,
        features: t.Tensor,
        index: t.Tensor | None = None,
    ) -> t.Tensor:
        """``(B, T, D, C)`` -> ``(B, L, d_model)``, one contextualized vector per token.

        ``index`` selects a subset of the flat grid before the blocks run -- a masked
        objective passes its context indices here, so the encoder never attends to a
        token it is supposed to be predicting.
        """

        return self.embed_values(
            self.token_values(features),
            self.spec.time_patches(features.shape[1]),
            index=index,
        )

    def embed_values(
        self,
        values: t.Tensor,
        num_time_patches: int,
        *,
        index: t.Tensor | None = None,
    ) -> t.Tensor:
        """:meth:`embed_tokens` for a caller that has already patchified.

        A masked objective needs the patches before it can pick a mask, and patchifying
        a 1290-token window twice per step is a whole discarded copy of the batch.
        """

        x = self.projection(values)
        if index is None:
            x = x + self.position.get_flat(num_time_patches)
        else:
            x = gather_tokens(x, index)
            x = x + self.position.gather(index, num_time_patches)

        # No src_key_padding_mask: every token is real, and handing nn.TransformerEncoder
        # an all-False mask costs ~35% of a full-grid encode by disabling its fused path.
        return self.blocks(x)


ENCODER_PREFIXES = {
    "teacher": ("teacher.", "backbone."),
    "student": ("student.", "backbone."),
    "backbone": ("backbone.",),
}


def load_encoder(checkpoint: str, encoder: str = "teacher") -> MotionTransformerEncoder:
    """The backbone of a pretrained objective, rebuilt from its checkpoint alone.

    The spec travels in the objective's hparams, so the architecture is read back rather
    than restated. Which submodule holds the weights depends on the objective -- JEPA has
    a teacher and a student, a masked objective has one backbone -- and everything else
    in the checkpoint (decoder, predictor, mask token, prediction head) is scaffolding
    that existed to train these weights and is dropped here.
    """

    if encoder not in ENCODER_PREFIXES:
        raise ValueError(f"encoder must be one of {tuple(ENCODER_PREFIXES)}, got {encoder!r}")

    loaded = t.load(checkpoint, map_location="cpu")
    backbone = MotionTransformerEncoder(EncoderSpec(**loaded["hyper_parameters"]["backbone"]))

    state = loaded["state_dict"]
    for prefix in ENCODER_PREFIXES[encoder]:
        weights = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if weights:
            backbone.load_state_dict(weights)
            return backbone

    raise ValueError(f"{checkpoint} holds no weights for encoder {encoder!r}.")
