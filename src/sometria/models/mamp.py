"""MAMP over motion tokens: hide the parts that move, predict how they move.

The sibling of :class:`~sometria.models.mae.MaskedAutoencoder`. Two things separate it
from that one, and they are the whole of MAMP (Mao et al.,
https://github.com/maoyunyao/MAMP):

- **The mask is motion-aware.** Tokens are drawn as targets in proportion to how much
  they move, so what is held out is the informative part of the window rather than an
  arbitrary tenth of it.
- **The target is motion, not values.** The decoder predicts ``x[t + stride] - x[t]``,
  taken by differencing *the input the encoder sees* before it is patchified, so the
  difference at a patch boundary is a real one rather than zero at every 8th frame.

The loss scores a subset of the channels and the head is exactly as wide as that subset,
as in the reference: predicting channels no loss scores would be parameters that never
receive a gradient. Each target token is standardized before the loss (the reference's
``norm_skes_loss``), which asks for the *shape* of a patch rather than its magnitude --
without it a squared error on motion is dominated by the few fastest tokens, which
motion-aware masking has deliberately selected for.

An earlier version took the target from the stored ``vel`` channel instead, on the theory
that a stored velocity makes the difference a channel selection rather than a
computation. It does not: that channel is signed-log compressed and normalized per DOF,
so its per-frame values are near-unpredictable from context, and the objective explained
7.7% of its target's variance in 10 epochs where the pose objective explained 98.8%.

Like the MAE sibling, the encoder is built here from an
:class:`~sometria.architecture.encoder.EncoderSpec` rather than handed in -- and stays a
:class:`~sometria.architecture.encoder.MotionTransformerEncoder`, because it is the part
a downstream probe loads back out of the checkpoint.
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
from sometria.masking import MaskSpec, extract_motion, patchify
from sometria.models.objective import PretextObjective
from sometria.models.window import (
    MaskedWindow,
    mask_window,
    masked_token_mse,
    standardize_tokens,
)


class MaskedMotionPredictor(PretextObjective):
    """Predict the motion of masked tokens from the ones left visible."""

    #: The reference's masking: nine tenths held out, drawn from the velocity channel at
    #: tau = 0.8. Dropping tau to 0 would make this MAE's uniform mask, which is what
    #: :class:`~sometria.models.mae.MaskedAutoencoder` is for.
    DEFAULT_MASK = MaskSpec(mask_ratio=0.90, tau=0.80, score_channels=(2,))

    def __init__(
        self,
        spec: EncoderSpec | dict | None = None,
        mask: MaskSpec | dict | None = None,
        *,
        decoder_depth: int = 5,
        # The reference's values.
        motion_stride: int = 1,
        loss_channels: tuple[int, ...] = (0, 1),     # representation.indices("sin", "cos")
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

        if not loss_channels:
            raise ValueError("the loss needs at least one channel to score.")

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
                "motion_stride": motion_stride,
                "loss_channels": tuple(loss_channels),
                "norm_targets": norm_targets,
            }
        )

        self.motion_stride = motion_stride
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

        self.loss_channels = tuple(loss_channels)
        self.register_buffer("loss_channel_index", t.tensor(self.loss_channels, dtype=t.long))
        self.prediction = nn.Linear(spec.d_model, spec.patch_size * len(self.loss_channels))

        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, window: MaskedWindow) -> t.Tensor:
        """``(B, L_keep, d_model)`` -- the context tokens, and nothing the decoder predicts."""

        return self.backbone.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )

    def motion_target(self, features: t.Tensor) -> t.Tensor:
        """``(B, L, patch_size * len(loss_channels))`` -- the window's own temporal difference.

        Differenced over the whole window and patchified afterwards, so a patch boundary
        carries a real difference; the last ``stride`` frames have no successor and stay
        at zero, exactly as the reference leaves them.
        """

        motion = patchify(extract_motion(features, self.motion_stride), self.spec.patch_size)
        return motion[..., self.loss_channel_index].flatten(start_dim=-2) # type: ignore

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

        target = self.motion_target(features)
        return window.mask.targets_of(prediction), window.mask.targets_of(target), window

    def reconstruction_loss(self, prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
        """``prediction`` and ``target`` are both ``(B, N, patch_size * len(loss_channels))``."""

        if self.norm_targets:
            target = standardize_tokens(target)
        return masked_token_mse(prediction, target)

    def step(self, batch: dict) -> tuple[t.Tensor, dict[str, t.Tensor | float]]:
        prediction, target, window = self(batch["features"])
        loss = self.reconstruction_loss(prediction, target)
        return loss, {"context_tokens": float(window.mask.context.shape[1])}
