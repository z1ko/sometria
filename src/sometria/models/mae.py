"""Plain MAE over motion tokens: hide most of the window, reconstruct what was hidden.

The sibling of :class:`~sometria.models.mamp.MaskedMotionPredictor`, and the simpler of
the two: the target is always the input's own patch values, the mask is always uniform,
and the loss scores every channel. Nothing branches, because nothing here is optional.

The encoder is built here from an
:class:`~sometria.architecture.encoder.EncoderSpec` rather than being handed in, because
nothing outside chooses it. It stays a
:class:`~sometria.architecture.encoder.MotionTransformerEncoder` all the same: it is the
one part of this model that outlives it, and a downstream probe reads it back out of the
checkpoint by its ``backbone.`` prefix. The decoder, the mask token and the prediction
head are scaffolding that existed to train those weights.

The encoder sees the context tokens only. The decoder receives the encoded context, a
learned mask token at every held-out position, and its own positional encoding, and
predicts the raw values of the held-out patches.
"""

from dataclasses import asdict

import torch as t
import torch.nn as nn

from sometria.architecture.encoder import (
    EncoderSpec,
    MotionTransformerEncoder,
    transformer_stack,
)
from sometria.architecture.pos_embed import PositionalEncoding
from sometria.masking import MaskSpec
from sometria.models.objective import PretextObjective
from sometria.models.window import (
    MaskedWindow,
    mask_window,
    masked_token_mse,
    standardize_tokens,
)


class MaskedAutoencoder(PretextObjective):
    """Reconstruct the values of masked motion tokens from the ones left visible."""

    #: Uniform masking: ``tau = 0`` skips scoring entirely, which is what makes this MAE
    #: rather than MAMP. With no scoring there is no channel to score, hence the empty set.
    DEFAULT_MASK = MaskSpec(mask_ratio=0.90, tau=0.0, score_channels=())

    def __init__(
        self,
        spec: EncoderSpec | dict | None = None,
        mask: MaskSpec | dict | None = None,
        *,
        decoder_depth: int = 5,
        norm_targets: bool = True,
        lr: float = 1e-3,
        min_lr_frac: float = 0.5,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__(
            mask,
            lr=lr,
            min_lr_frac=min_lr_frac,
            weight_decay=weight_decay,
            warmup_frac=warmup_frac,
        )

        # The dict form is what comes back out of a checkpoint, and what a YAML
        # ``encoder:`` block maps to.
        if isinstance(spec, dict):
            spec = EncoderSpec(**spec)
        self.spec = spec = spec or EncoderSpec()

        # Saved here rather than through ``save_objective_hyperparameters``: that helper
        # stores the architecture under ``backbone``, and a hyperparameter only rebuilds
        # the model if its name is this constructor's argument name.
        self.save_hyperparameters(
            {
                "spec": asdict(spec),
                "mask": asdict(self.mask),
                "lr": lr,
                "min_lr_frac": min_lr_frac,
                "weight_decay": weight_decay,
                "warmup_frac": warmup_frac,
                "decoder_depth": decoder_depth,
                "norm_targets": norm_targets,
            }
        )

        self.norm_targets = norm_targets
        time_patches, num_dofs = spec.grid_shape

        # The encoder is a module rather than three loose fields because it is the one
        # part of this model that outlives it: a probe loads `backbone.*` out of the
        # checkpoint and drops everything below. Decoder and mask token are scaffolding.
        self.backbone = MotionTransformerEncoder(spec)

        # Decoder: the whole grid, context tokens where they were, mask tokens elsewhere.
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.decoder_position = PositionalEncoding(time_patches, num_dofs, spec.d_model)
        self.decoder = transformer_stack(spec, decoder_depth)
        self.prediction = nn.Linear(spec.d_model, spec.token_dim)

        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, window: MaskedWindow) -> t.Tensor:
        """``(B, L_keep, d_model)`` -- the context tokens, and nothing the decoder predicts."""

        return self.backbone.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )

    def forward(
        self,
        features: t.Tensor,
        generator: t.Generator | None = None,
    ) -> tuple[t.Tensor, t.Tensor, MaskedWindow]:
        """Return ``(prediction, target, window)``, the first two over target tokens only."""

        window = mask_window(features, self.spec, self.mask, generator=generator)
        encoded = self.encode(window)

        # A full grid of mask tokens with the encoded context written back into its own
        # positions -- scatter_ rather than two gathers, so the decoder sees the tokens
        # in grid order and its positional encoding needs no reordering.
        batch, length, _ = window.values.shape
        tokens = self.mask_token.expand(batch, length, -1).clone()
        tokens.scatter_(
            dim=1,
            index=window.mask.context.unsqueeze(-1).expand(-1, -1, self.spec.d_model),
            src=encoded,
        )
        decoded = self.decoder(tokens + self.decoder_position.get_flat(window.num_time_patches))
        prediction = self.prediction(decoded)

        return window.mask.targets_of(prediction), window.mask.targets_of(window.values), window

    def reconstruction_loss(self, prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
        """``prediction`` and ``target`` are both ``(B, N, patch_size * num_features)``."""

        if self.norm_targets:
            target = standardize_tokens(target)
        return masked_token_mse(prediction, target)

    def step(self, batch: dict) -> tuple[t.Tensor, dict[str, t.Tensor | float]]:
        prediction, target, window = self(batch["features"])
        loss = self.reconstruction_loss(prediction, target)
        return loss, {"context_tokens": float(window.mask.context.shape[1])}
