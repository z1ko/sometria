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
from sometria.masking import (
    MaskIndices,
    MaskSpec,
    motion_aware_mask,
    patchify,
    token_validity,
)


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
    target_valid: t.Tensor     # (B, L - L_keep) bool   which targets are real frames

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
    valid: t.Tensor | None = None,
    generator: t.Generator | None = None,
) -> MaskedWindow:
    """Split ``(B, T, D, C)`` into context and targets under one :class:`MaskSpec`.

    One ``patchify`` serves both the masker, which scores the patches, and the backbone,
    which projects them -- patchifying a 1290-token window twice per step is a whole
    discarded copy of the batch.

    ``valid`` is an optional ``(B, T)`` frame mask. Tokens covering any padded frame sort
    last, so they fill the targets before any real token does -- but only up to the target
    budget: a window with more invalid tokens than ``mask_ratio * L`` spills the remainder
    into the context, where the backbone's index path does not mask them. Callers that
    exclude short samples up front (``MotionViewSpec.min_frames``) pass ``None`` and never
    meet this; anything that starts padding windows has to check the budget covers them.
    """

    spec.check_features(features)
    num_time_patches = spec.time_patches(features.shape[1])
    patches = patchify(features, spec.patch_size)
    length = patches.shape[1]

    indices = motion_aware_mask(
        patches,
        score_channels=mask.score_channels,
        mask_ratio=mask.mask_ratio,
        tau=mask.tau,
        valid=valid,
        generator=generator,
    )

    target_valid = (
        t.ones(indices.targets.shape, dtype=t.bool, device=features.device)
        if valid is None
        else indices.targets_of(token_validity(valid, spec.patch_size, length).to(features.device))
    )

    return MaskedWindow(
        spec=spec,
        patches=patches,
        mask=indices,
        num_time_patches=num_time_patches,
        target_valid=target_valid,
    )


def masked_token_mse(
    prediction: t.Tensor,
    target: t.Tensor,
    target_valid: t.Tensor,
) -> t.Tensor:
    """Squared error over target tokens, ignoring the ones that are padding.

    Mean per token, then mean over valid target tokens: a token counts once whatever its
    width, which is what makes the number comparable between an objective predicting two
    channels of a patch and one predicting a ``d_model`` embedding.
    """

    if not target_valid.any():
        # Multiplied rather than replaced by a constant: a window with no real target
        # still has to hand back something the graph can be walked back through. Summed
        # rather than meaned because the target set can be empty outright -- a mask_ratio
        # small enough to round every token into the context is a legal MaskSpec -- and
        # the mean of an empty tensor is NaN.
        return prediction.sum() * 0.0

    per_token = (prediction - target).square().mean(dim=-1)
    return (per_token * target_valid).sum() / target_valid.sum()


def standardize_tokens(target: t.Tensor, eps: float = 1.0e-6) -> t.Tensor:
    """Zero mean and unit variance per token, over its own values.

    MAMP's ``norm_skes_loss``. The loss then asks for the *shape* of a patch rather than
    its magnitude -- without it a squared error on motion is dominated by the few fastest
    tokens, which motion-aware masking has deliberately selected for.
    """

    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    return (target - mean) / (var + eps) ** 0.5
