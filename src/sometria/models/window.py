"""One window, already split into what is seen and what is predicted.

Every masked pretext objective opens the same way: patchify the window, score it, draw a
:class:`~sometria.masking.MaskIndices` over the token grid, and work out which targets
are real. What differs between them starts after that -- MAE reconstructs the input's
patch values, MAMP the input's temporal difference, JEPA the teacher's embedding.

This module owns the part they share. It sits above :mod:`sometria.masking`, which stays
a leaf so a mask can still be computed and plotted with no model in sight, and above
:class:`~sometria.architecture.encoder.EncoderSpec`, which answers the questions about
the token grid that need no built backbone.
"""

from dataclasses import dataclass

import torch as t

from sometria.architecture.encoder import EncoderSpec
from sometria.masking import MaskIndices, MaskSpec, motion_aware_mask, patchify


@dataclass(frozen=True)
class MaskedWindow:
    """A window already split into **Context** and **Target**.

    Holds no parameters and builds no module: it is the window's tokens plus the two
    index sets over them, handed to whichever **Pretext objective** asked for it.
    """

    spec: EncoderSpec
    patches: t.Tensor          # (B, L, patch_size, C)  one DOF over patch_size frames
    mask: MaskIndices
    num_time_patches: int      # this window's, which a short window makes smaller

    @property
    def values(self) -> t.Tensor:
        """``(B, L, patch_size * C)``, what the backbone projects.

        A view, not a copy: ``patchify`` returns a contiguous tensor, so flattening the
        last two axes costs nothing. Kept as a property rather than a field so the
        patch-shaped tensor stays the one source -- an objective selecting target
        channels needs the axes apart.
        """

        return self.patches.flatten(start_dim=-2)


def mask_window(
    features: t.Tensor,
    spec: EncoderSpec,
    mask: MaskSpec,
    *,
    generator: t.Generator | None = None,
) -> MaskedWindow:
    """Split ``(B, T, D, C)`` into context and targets under one :class:`MaskSpec`.

    One ``patchify`` serves both the masker, which scores the patches, and the backbone,
    which projects them -- patchifying a 1290-token window twice per step is a whole
    discarded copy of the batch.

    Every token is real. A **Window** is a crop of a motion long enough to fill it:
    ``MotionViewSpec.min_frames`` drops the shorter ones on every split, and
    ``WindowCollate`` refuses one if it ever gets that far.
    """

    spec.check_features(features)
    patches = patchify(features, spec.patch_size)

    return MaskedWindow(
        spec=spec,
        patches=patches,
        mask=motion_aware_mask(
            patches,
            score_channels=mask.score_channels,
            mask_ratio=mask.mask_ratio,
            tau=mask.tau,
            generator=generator,
        ),
        num_time_patches=spec.time_patches(features.shape[1]),
    )


def masked_token_mse(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """Squared error over target tokens.

    Named rather than inlined for the empty-set guard: a ``mask_ratio`` small enough to
    round every token into the context is a legal :class:`~sometria.masking.MaskSpec`,
    and the mean of an empty tensor is NaN. Summing rather than replacing with a constant
    keeps a graph the backward pass can be walked through.
    """

    if prediction.numel() == 0:
        return prediction.sum() * 0.0

    return (prediction - target).square().mean()


def standardize_tokens(target: t.Tensor, eps: float = 1.0e-6) -> t.Tensor:
    """Zero mean and unit variance per token, over its own values.

    MAMP's ``norm_skes_loss``. The loss then asks for the *shape* of a patch rather than
    its magnitude -- without it a squared error on motion is dominated by the few fastest
    tokens, which motion-aware masking has deliberately selected for.
    """

    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    return (target - mean) / (var + eps) ** 0.5
