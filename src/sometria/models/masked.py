"""Masked reconstruction over motion tokens: MAE and MAMP as one objective.

The encoder sees context tokens only. A shallow decoder receives the encoded context
plus a learned mask token at every target position and predicts the held-out patch
values. Everything architectural belongs to the backbone, and splitting the window into
**Context** and **Target** belongs to :mod:`sometria.models.window`; what lives here is
what is predicted and what the loss is.

MAE and MAMP are not two models. They are this one at two configurations:

- ``target="values"``, ``MaskSpec(tau=0.0)``  -- uniform masking, reconstruct the input: MAE.
- ``target="motion"``, ``MaskSpec(tau=0.8)``  -- motion-aware masking, predict the temporal
  difference of the input: MAMP.

Both follow the reference (Mao et al., https://github.com/maoyunyao/MAMP) in taking the
motion target by differencing *the input the encoder sees*, and in normalizing each
target token to zero mean and unit variance before the loss (``norm_skes_loss`` there,
``standardize_tokens`` here). An earlier version selected the stored ``vel`` channel
instead, on the theory that a stored velocity makes the difference a channel selection
rather than a computation. It does not: that channel is signed-log compressed and
normalized per DOF, so its per-frame values are near-unpredictable from context, and the
objective explained 7.7% of its target's variance in 10 epochs while the pose objective
explained 98.8%.
"""

import torch as t
import torch.nn as nn

from sometria.architecture.encoder import (
    EncoderSpec,
    MotionTransformerEncoder,
    as_backbone,
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

TARGETS = ("values", "motion")


class MaskedMotionAutoencoder(PretextObjective):
    """Predict the values of masked motion tokens from the ones left visible."""

    # The reference's MAMP masking. MAE is the same at tau <= 0; both baselines set every
    # knob in their config, so this is what an unconfigured model gets, not a policy.
    DEFAULT_MASK = MaskSpec(mask_ratio=0.90, tau=0.80, score_channels=(2,))

    def __init__(
        self,
        backbone: MotionTransformerEncoder | EncoderSpec | dict | None = None,
        mask: MaskSpec | dict | None = None,
        *,
        decoder_depth: int = 5,
        # The defaults are the MAMP configuration, at the reference's values.
        target: str = "motion",
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

        backbone = as_backbone(backbone)
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        if not loss_channels:
            raise ValueError("the loss needs at least one channel to score.")

        self.save_objective_hyperparameters(
            backbone,
            decoder_depth=decoder_depth,
            target=target,
            motion_stride=motion_stride,
            loss_channels=tuple(loss_channels),
            norm_targets=norm_targets,
        )

        self.backbone = backbone
        spec = backbone.spec
        self.target = target
        self.motion_stride = motion_stride
        self.norm_targets = norm_targets

        time_patches, num_dofs = spec.grid_shape
        self.mask_token = nn.Parameter(t.zeros(1, 1, spec.d_model))
        self.decoder_position = PositionalEncoding(time_patches, num_dofs, spec.d_model)
        self.decoder = transformer_stack(spec, decoder_depth)

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
    ) -> tuple[t.Tensor, t.Tensor, MaskedWindow]:
        """Return ``(prediction, target, window)``, the first two over target tokens only."""

        spec = self.backbone.spec
        window = mask_window(features, spec, self.mask, valid=valid, generator=generator)

        # The encoder always reads the input; only what the decoder is asked for changes.
        # extract_motion runs on the window, not on the patches, so the difference at a
        # patch boundary is a real one rather than being zeroed at every 8th frame.
        target_source = (
            window.patches
            if self.target == "values"
            else patchify(extract_motion(features, self.motion_stride), spec.patch_size)
        )

        encoded = self.backbone.embed_values(
            window.values, window.num_time_patches, index=window.mask.context
        )

        batch, length, _ = window.values.shape
        tokens = self.mask_token.expand(batch, length, -1).clone()
        tokens.scatter_(
            dim=1,
            index=window.mask.context.unsqueeze(-1).expand(-1, -1, spec.d_model),
            src=encoded,
        )
        decoded = self.decoder(tokens + self.decoder_position.get_flat(window.num_time_patches))
        prediction = self.prediction(decoded)

        target = target_source[..., self.loss_channel_index].flatten(start_dim=-2)
        return window.mask.targets_of(prediction), window.mask.targets_of(target), window

    def reconstruction_loss(
        self,
        prediction: t.Tensor,
        target: t.Tensor,
        target_valid: t.Tensor,
    ) -> t.Tensor:
        """``prediction`` and ``target`` are both ``(B, N, patch_size * len(loss_channels))``."""

        if self.norm_targets:
            target = standardize_tokens(target)
        return masked_token_mse(prediction, target, target_valid)

    def step(self, batch: dict) -> tuple[t.Tensor, dict[str, t.Tensor | float]]:
        prediction, target, window = self(batch["features"], batch.get("valid"))
        loss = self.reconstruction_loss(prediction, target, window.target_valid)
        return loss, {"context_tokens": float(window.mask.context.shape[1])}
