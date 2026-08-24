"""What a feature channel means.

One module owns the whole path between physical OpenSim channels and the tensors
the model consumes:

    raw motion  --encode-->  features  --to_model-->  model input
    raw motion  <--decode--  features  <--to_physical--

``Representation`` is a value: it carries the channel layout (which channels
exist, which DOFs survive, which slots get compressed) and nothing else.
Normalization statistics are passed in as an argument, never held. The layout
changes when the features are redesigned (v2 -> v3); the statistics change when
the training split changes. Different clocks, so they are not fused.

This module exists because the two halves used to be scattered: the signed-log
mask was written and documented but never called, so the compression it gated
silently never ran until a residual analysis found 0.6% of frames carrying 43%
of the loss. And there was no inverse path at all -- every call site that wanted
physical units hand-assembled denormalize + signed_exp.
"""

from pathlib import Path

import numpy as np
import torch as t
import yaml

# The current, and only, representation on disk. Descriptive, not enforced: it names
# what the tensors under data/processed/motions/ contain.
NAME = "opensim_sincos_log_vel_acc_tau_v2"


# Raw velocity, acceleration and torque are heavy-tailed: knee torque runs three orders of
# magnitude above wrist torque, and rare impact frames reach tens of sigma after normalization,
# where they dominate any mean-squared loss. signed_log compresses that range while staying
# smooth, sign-preserving, and near-identity for small values (d/dx = 1 at 0).
def signed_log(x: np.ndarray) -> np.ndarray:
    """Compress magnitude while preserving sign: ``sign(x) * log1p(|x|)``."""

    return np.sign(x) * np.log1p(np.abs(x))


def signed_exp(x: np.ndarray) -> np.ndarray:
    """Invert :func:`signed_log`."""

    return np.sign(x) * np.expm1(np.abs(x))


def _load_human_definition(path: str | Path) -> dict:
    """Load the YAML description of the OpenSim human model columns."""

    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _dof_indices(human: dict, key: str) -> list[int]:
    """Resolve DOF names from the human config into integer axis indices."""

    index = {dof: i for i, dof in enumerate(human["dofs"])}
    names = human.get(key) or []
    if isinstance(names, str):
        names = [names]
    unknown = [n for n in names if n not in index]
    if unknown:
        raise KeyError(f"{key}: not in dofs: {unknown}")
    return [index[n] for n in names]


def channel_names(indices: tuple[int, ...] | None = None) -> tuple[str, ...]:
    """Feature-channel names, by index into :attr:`Representation.channels`.

    A model logging a per-channel breakdown wants the names without needing the DOF list
    a full :class:`Representation` is built from.
    """

    if indices is None:
        return Representation.channels
    return tuple(Representation.channels[i] for i in indices)


class Representation:
    """The feature layout: what the five channels mean and how to move between them."""

    # The feature channels live here, not in human.yaml. human.yaml describes the *raw*
    # OpenSim columns (pos, vel, acc, tau); how those become model features is a modelling
    # decision, and modelling decisions belong in code.
    channels: tuple[str, ...] = ("sin", "cos", "vel", "acc", "tau")

    def __init__(self, dofs: tuple[str, ...], excluded: tuple[int, ...]) -> None:
        self.name = NAME
        self.dofs = dofs
        self._excluded = excluded

        # Which slots carry a compressed, normalizable quantity. sin/cos are already
        # bounded in [-1, 1] and encode circular geometry, so they are left alone by both
        # signed_log and mean/std normalization. One mask, one concept: what used to be
        # feature_log_mask and feature_normalization_mask were byte-identical.
        self._mask = np.ones((len(dofs), len(self.channels)), dtype=bool)
        self._mask[:, :2] = False

    @classmethod
    def from_config(cls, path: str | Path) -> "Representation":
        """Build the representation from the human model definition on disk."""

        return cls.from_human(_load_human_definition(path))

    @classmethod
    def from_human(cls, human: dict) -> "Representation":
        """Build the representation from an already-loaded human model definition."""

        excluded = set(_dof_indices(human, "excluded_dofs"))
        translations = set(_dof_indices(human, "root_position_dofs"))

        kept = [(i, dof) for i, dof in enumerate(human["dofs"]) if i not in excluded]
        still_there = [dof for i, dof in kept if i in translations]
        if still_there:
            raise ValueError(
                f"root_position_dofs are distances, not angles, so encode cannot represent them "
                f"as sin/cos: {still_there}. Either add them to excluded_dofs, or give them their "
                f"own branch in encode."
            )

        return cls(tuple(dof for _, dof in kept), tuple(sorted(excluded)))

    def __repr__(self) -> str:
        return f"Representation({self.name!r}, {len(self.dofs)} dofs, {len(self.channels)} channels)"

    def indices(self, *names: str) -> tuple[int, ...]:
        """Return feature-channel indices by name, e.g. ``indices('vel', 'acc', 'tau')``."""

        unknown = [n for n in names if n not in self.channels]
        if unknown:
            raise KeyError(f"not feature channels: {unknown}; have {list(self.channels)}")
        return tuple(self.channels.index(n) for n in names)

    def encode(self, motion: np.ndarray) -> np.ndarray:
        """Raw OpenSim channels ``(T, dofs, 4)`` -> features ``(T, kept_dofs, 5)``.

        Input channels are ``position, velocity, acceleration, torque``; output channels
        are ``sin(position), cos(position), velocity, acceleration, torque`` with the
        three derivative channels passed through :func:`signed_log`. Every kept DOF is an
        angle, so all of them get the same slots and the DOF axis stays a clean per-joint
        token space. Root translations are excluded because they are distances.
        """

        kept = np.delete(motion, self._excluded, axis=1)

        out = np.zeros(kept.shape[:2] + (len(self.channels),), dtype=kept.dtype)
        out[:, :, 0] = np.sin(kept[:, :, 0])
        out[:, :, 1] = np.cos(kept[:, :, 0])
        out[:, :, 2:] = kept[:, :, 1:]
        return np.where(self._mask, signed_log(out), out)

    def decode(self, features: np.ndarray) -> np.ndarray:
        """Features ``(T, kept_dofs, 5)`` -> raw channels ``(T, kept_dofs, 4)``.

        The inverse of :meth:`encode` for the DOFs it kept; the excluded ones are gone
        for good. Position comes back through ``atan2(sin, cos)``, so it is recovered
        wrapped to ``(-pi, pi]`` -- exact only for inputs already in that range.
        """

        out = np.zeros(features.shape[:2] + (4,), dtype=features.dtype)
        out[:, :, 0] = np.arctan2(features[:, :, 0], features[:, :, 1])
        out[:, :, 1:] = signed_exp(features[:, :, 2:])
        return out

    def to_model(self, features: t.Tensor, stats: dict) -> t.Tensor:
        """Features -> model input: mean/std normalization on the compressed channels."""

        mask = t.as_tensor(self._mask, device=features.device)
        return t.where(mask, (features - stats["mean"]) / stats["std"], features)

    def to_physical(self, x: t.Tensor, stats: dict) -> t.Tensor:
        """Model input (or output) -> features. The inverse of :meth:`to_model`."""

        mask = t.as_tensor(self._mask, device=x.device)
        return t.where(mask, x * stats["std"] + stats["mean"], x)
