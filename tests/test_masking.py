"""Run with: python tests/test_masking.py"""

import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.masking import MaskIndices, motion_aware_mask, patchify
from sometria.representation import Representation

REP = Representation.from_config(Path(__file__).resolve().parents[1] / "config/human.yaml")

VEL = REP.indices("vel")
PATCH = 8
D = len(REP.dofs)
C = len(REP.channels)


def _features(frames=240, batch=2, seed=0):
    """A normalized-looking window: (B, T, D, C)."""

    g = t.Generator().manual_seed(seed)
    return t.randn(batch, frames, D, C, generator=g)


LOUD_DOFS = 10


def _loud_dofs(batch=1, frames=240, seed=1):
    """A window where the first LOUD_DOFS DOFs move hard and the rest barely move."""

    g = t.Generator().manual_seed(seed)
    x = t.randn(batch, frames, D, C, generator=g) * 0.01
    x[:, :, :LOUD_DOFS, VEL[0]] = 10.0
    return x


def _loud_tokens(L):
    """Which flat token indices _loud_dofs made loud. Flat index is t * D + d."""

    is_loud = t.zeros(L, dtype=t.bool)
    is_loud.reshape(-1, D)[:, :LOUD_DOFS] = True
    return is_loud


def _mask_frequency(patches, draws, seed=0, **kwargs):
    """How often each token lands in ``targets`` over repeated draws."""

    g = t.Generator().manual_seed(seed)
    hits = t.zeros(patches.shape[1])
    for _ in range(draws):
        hits[motion_aware_mask(patches, score_channels=VEL, generator=g, **kwargs).targets[0]] += 1
    return hits


def test_patchify_places_a_known_cell_at_t_times_dofs_plus_d():
    x = t.zeros(1, 240, D, C)
    x[0, 3 * PATCH + 2, 11, 4] = 7.0                       # time patch 3, frame 2 within it

    patches = patchify(x, PATCH)

    assert patches.shape == (1, (240 // PATCH) * D, PATCH, C)
    assert patches[0, 3 * D + 11, 2, 4] == 7.0             # flat index is t * D + d
    assert patches.sum() == 7.0                            # and nothing landed anywhere else


def test_patchify_preserves_every_value():
    x = _features()
    patches = patchify(x, PATCH)
    assert patches.numel() == x.numel()
    assert t.allclose(patches.sum(), x.sum())


def test_patchify_rejects_a_window_that_does_not_divide():
    try:
        patchify(_features(frames=250), PATCH)
    except ValueError:
        pass
    else:
        raise AssertionError("a window not divisible by patch_size should raise")


def test_context_and_targets_partition_the_grid():
    patches = patchify(_features(), PATCH)
    B, L = patches.shape[0], patches.shape[1]

    m = motion_aware_mask(patches, score_channels=VEL, mask_ratio=0.80)

    assert isinstance(m, MaskIndices)
    assert m.context.shape == (B, int(L * 0.20))
    assert m.targets.shape == (B, L - int(L * 0.20))
    for b in range(B):
        both = t.cat([m.context[b], m.targets[b]])
        assert t.equal(both.sort().values, t.arange(L))     # disjoint and exhaustive


def test_a_fixed_generator_reproduces_the_same_split():
    patches = patchify(_features(), PATCH)

    first  = motion_aware_mask(patches, score_channels=VEL, generator=t.Generator().manual_seed(7))
    second = motion_aware_mask(patches, score_channels=VEL, generator=t.Generator().manual_seed(7))
    other  = motion_aware_mask(patches, score_channels=VEL, generator=t.Generator().manual_seed(8))

    assert t.equal(first.context, second.context)
    assert not t.equal(first.context, other.context)


def test_each_sample_in_a_batch_gets_its_own_draw():
    patches = patchify(_features(batch=4), PATCH)
    m = motion_aware_mask(patches, score_channels=VEL, generator=t.Generator().manual_seed(3))
    assert not t.equal(m.context[0], m.context[1])


def test_small_tau_holds_out_every_loud_patch():
    patches = patchify(_loud_dofs(), PATCH)
    loud = _loud_tokens(patches.shape[1]).nonzero().flatten()

    m = motion_aware_mask(patches, score_channels=VEL, mask_ratio=0.5, tau=1e-4)

    # tau -> 0 sharpens the distribution until Gumbel noise cannot lift a loud patch back
    # into the context. Only the loud/quiet split is asserted: the quiet tokens are within
    # noise of each other by construction, so their relative order stays random.
    assert set(loud.tolist()) <= set(m.targets[0].tolist())


def test_loud_patches_are_masked_more_often_than_quiet_ones():
    patches = patchify(_loud_dofs(), PATCH)
    loud = _loud_tokens(patches.shape[1])
    quiet = ~loud

    draws = 200
    hits = _mask_frequency(patches, draws, mask_ratio=0.5)

    assert hits[loud].mean() > 1.4 * hits[quiet].mean()


def test_lowering_tau_sharpens_the_preference():
    patches = patchify(_loud_dofs(), PATCH)
    loud = _loud_tokens(patches.shape[1])

    draws = 200
    mild = _mask_frequency(patches, draws, mask_ratio=0.5, tau=0.75)
    sharp = _mask_frequency(patches, draws, mask_ratio=0.5, tau=0.1)

    assert sharp[loud].mean() > mild[loud].mean()


def test_nonpositive_tau_ignores_motion_and_masks_uniformly():
    patches = patchify(_loud_dofs(), PATCH)
    loud = _loud_tokens(patches.shape[1])
    quiet = ~loud

    draws = 400
    hits = _mask_frequency(patches, draws, mask_ratio=0.5, tau=0.0)

    # Every token should be held out about half the time regardless of how loud it is.
    assert abs(hits[loud].mean() - hits[quiet].mean()) < 0.05 * draws
    assert abs(hits.mean() - 0.5 * draws) < 0.05 * draws


def test_padding_is_never_spent_on_context():
    x = _features(batch=1)
    valid = t.ones(1, x.shape[1], dtype=t.bool)
    valid[0, 120:] = False                                   # second half is collate padding
    x[0, 120:] = 0.0

    m = motion_aware_mask(patchify(x, PATCH), score_channels=VEL, valid=valid, mask_ratio=0.5)

    padded = set(range((120 // PATCH) * D, (240 // PATCH) * D))
    assert not set(m.context[0].tolist()) & padded


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
