"""Splitting a window into context and targets.

Masked prediction needs two index sets over the token grid: the **context** an encoder
sees, and the **targets** a decoder must reconstruct. This module produces them and
nothing else -- it holds no parameters, builds no modules, and never touches an
embedding. Selection is index arithmetic; gathering belongs to the caller.

Following MAMP (Mao et al., "Masked Motion Predictors are Strong 3D Action Learners"),
targets are drawn in proportion to how much a patch moves, so the informative parts of a
window are the ones held out. The reference has to approximate motion by differencing
raw joint coordinates; we read ``vel`` straight off the feature channels instead, which
is why nothing here patches up an undefined first frame.

Kept deliberately separate from ``sometria.architecture`` so a mask can be computed,
plotted and tested without instantiating a model.
"""

from dataclasses import dataclass

import torch as t

# Guards the log and the division by a max that can legitimately be zero (a window whose
# velocity channel is flat everywhere).
EPS: float = 1e-10


@dataclass(frozen=True)
class MaskIndices:
    """The two halves of a masked window, as indices into the flattened token grid.

    Disjoint and exhaustive: together they name every token exactly once. Both are
    ``(batch, n)`` integer tensors suitable for ``torch.gather`` after an ``unsqueeze``.
    """

    context: t.Tensor    # (B, L_keep)        what the encoder sees
    targets: t.Tensor    # (B, L - L_keep)    what the decoder must predict


def patchify(features: t.Tensor, patch_size: int) -> t.Tensor:
    """``(B, T, D, C)`` -> ``(B, TP * D, patch_size, C)``.

    One token is one DOF over ``patch_size`` consecutive frames. Tokens are laid out so
    that flat index ``t * D + d`` addresses time patch ``t`` of DOF ``d`` -- the ordering
    ``PositionalEncoding.get_flat`` already produces.

    ``patch_size`` and ``C`` stay separate axes rather than being flattened together, so
    a caller can select feature channels by index without being told the channel count.
    An embedding that wants a single vector per token flattens the last two axes itself.
    """

    B, T, D, C = features.shape
    if T % patch_size != 0:
        raise ValueError(f"window of {T} frames is not divisible by patch_size {patch_size}")

    patches = features.reshape(B, T // patch_size, patch_size, D, C)
    patches = patches.permute(0, 1, 3, 2, 4) # (B, TP, D, patch_size, C)
    return patches.reshape(B, -1, patch_size, C)


def token_validity(valid: t.Tensor, patch_size: int, length: int) -> t.Tensor:
    """``(B, T)`` frame mask -> ``(B, length)`` token mask.

    A token is valid only if *every* frame it covers is real: a patch that is half
    padding is not half usable. The same reduction is needed by the masker (to keep
    padding out of the context set), by the encoder's pooling (so a mean is taken over
    real tokens only) and by the reconstruction loss, so it lives here rather than being
    written out three times.

    ``length`` is the flat token count ``TP * D``; the per-time-patch mask is repeated
    across the DOF axis, which is what ``t * D + d`` ordering asks for.
    """

    if valid.shape[1] % patch_size != 0:
        raise ValueError(
            f"valid covers {valid.shape[1]} frames, not divisible by patch_size {patch_size}"
        )
    usable = valid.reshape(valid.shape[0], -1, patch_size).all(dim=-1)          # (B, TP)
    if length % usable.shape[1] != 0:
        raise ValueError(f"{length} tokens is not a whole number of {usable.shape[1]} time patches")
    return usable.repeat_interleave(length // usable.shape[1], dim=1)


def motion_aware_mask(
    patches: t.Tensor,
    *,
    score_channels: tuple[int, ...],
    mask_ratio: float = 0.80,
    tau: float = 0.75,
    valid: t.Tensor | None = None,
    generator: t.Generator | None = None,
) -> MaskIndices:
    """Split a patchified window into context and targets, biased toward motion.

    ``patches`` is ``(B, L, patch_size, C)`` as returned by :func:`patchify`.
    ``score_channels`` names the channels motion is read from -- pass
    ``representation.indices("vel")`` rather than a literal, so channel meaning stays
    owned by :class:`~sometria.representation.Representation`.

    Each token scores as the mean absolute value of those channels over its frames. The
    scores become a distribution through ``softmax(score / (max * tau))`` and are drawn
    without replacement by Gumbel top-k, with the *highest*-scoring tokens becoming
    targets. ``tau`` interpolates the whole range: small sharpens toward a deterministic
    top-k, large flattens toward uniform, and ``tau <= 0`` skips scoring altogether and
    masks uniformly at random -- the ablation baseline, same function, no second path.

    Note that scores are max-normalized, so the logit spread is capped at ``1 / tau``
    regardless of how extreme the underlying motion is. At the default ``tau = 0.75``
    that is ~1.33, close to one standard deviation of Gumbel noise, which makes the
    default a mild preference rather than a hard selection. Lower ``tau`` if you want it
    to bite.

    ``valid`` is an optional ``(B, T)`` frame mask; tokens covering any invalid frame are
    forced into ``targets`` so a padded window never spends its context budget on
    padding. Callers that exclude short samples up front (``MotionViewSpec.min_frames``)
    pass ``None``.
    """

    B, L, patch_size, _ = patches.shape
    device = patches.device
    # round, not truncate: 1290 * (1.0 - 0.80) is 257.99999... in binary floating point,
    # which would silently hand back one fewer context token than the ratio asks for.
    len_keep = round(L * (1.0 - mask_ratio))

    if tau > 0:
        score = patches[..., list(score_channels)].abs().mean(dim=(-2, -1))          # (B, L)
        logits = score / (score.amax(dim=-1, keepdim=True) * tau + EPS)
        # Gumbel top-k: perturbing log-probabilities by Gumbel noise and sorting draws
        # without replacement from the softmax. log_softmax rather than log(softmax) so
        # a sharp tau cannot underflow.
        noise = t.log_softmax(logits, dim=-1) + _gumbel((B, L), device, generator)
    else:
        noise = t.rand(B, L, device=device, generator=generator)

    if valid is not None:
        # A token is usable only if every frame it covers is real. +inf sorts it last,
        # which puts it in targets; masking the noise afterwards keeps +inf out of the
        # softmax, where it would produce NaN.
        usable = token_validity(valid, patch_size, L).to(device)
        noise = noise.masked_fill(~usable, float("inf"))

    order = noise.argsort(dim=-1)          # ascending: quiet and real first, loud last
    return MaskIndices(context=order[:, :len_keep], targets=order[:, len_keep:])


def _gumbel(shape: tuple[int, ...], device, generator: t.Generator | None) -> t.Tensor:
    """Standard Gumbel(0, 1) noise."""

    u = t.rand(shape, device=device, generator=generator)
    return -t.log(-t.log(u + EPS) + EPS)


def extract_motion(features: t.Tensor, stride: int = 1) -> t.Tensor:
    """``(B, T, D, C)`` -> the same shape, holding ``x[t + stride] - x[t]``.

    MAMP's ``extract_motion``. The motion is taken over the *whole window* before it is
    patchified, so a token still carries ``patch_size`` values per channel and the
    prediction head keeps its shape; the last ``stride`` frames have no successor and are
    left at zero, exactly as the reference leaves them.

    Differencing here rather than reading the stored ``vel`` channel is deliberate. The
    stored channel is signed-log compressed and normalized per DOF, which is a different
    quantity from a plain temporal difference of the input the encoder actually sees --
    and the compression is what made the velocity target unlearnable.
    """

    if stride < 1:
        raise ValueError(f"motion stride must be at least 1, got {stride}")
    if stride >= features.shape[1]:
        raise ValueError(
            f"motion stride {stride} needs a window longer than {features.shape[1]} frames"
        )

    motion = t.zeros_like(features)
    motion[:, :-stride] = features[:, stride:] - features[:, :-stride]
    return motion
