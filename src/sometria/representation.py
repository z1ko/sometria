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
from scipy.spatial.transform import Rotation
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
    # A model logging channels of another representation (SmplRepresentation has 18) would
    # otherwise index off the end of this tuple; an unnamed channel logs as its index.
    return tuple(
        Representation.channels[i] if i < len(Representation.channels) else f"c{i}"
        for i in indices
    )


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

        # Which slots carry a compressed, normalizable quantity. The pose channels are
        # already bounded in [-1, 1] and encode rotation geometry, so they are left alone
        # by both signed_log and mean/std normalization. One mask, one concept: what used
        # to be feature_log_mask and feature_normalization_mask were byte-identical.
        # Read off the channel names so a different layout does not need a new rule.
        pose = tuple(c.startswith(("sin", "cos", "rot")) for c in self.channels)
        self._mask = np.tile(np.logical_not(pose), (len(dofs), 1))

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


# The SMPL feature layout. Named separately from NAME because statistics, tensors and
# checkpoints computed under one layout are meaningless under the other.
SMPL_NAME = "smpl_rot6d_log_vel_acc_v1"


class SmplRepresentation(Representation):
    """The SMPL feature layout: per joint, a 6D rotation and its first two derivatives.

    Exists to answer one question -- whether the OpenSim conversion earns its keep --
    so it copies every choice the OpenSim layout makes that is not about OpenSim: one
    token per joint, pose channels left uncompressed, derivatives through signed_log,
    the take's global orientation excluded.

    Two differences are the experiment. A SMPL joint is a ball joint, so its pose is a
    rotation rather than a scalar angle: it is stored as the first two columns of the
    rotation matrix (continuous everywhere, unlike axis-angle, which jumps at +-pi)
    rather than as sin/cos of one angle. And there is no ``tau``: torque comes out of
    inverse dynamics on a scaled musculoskeletal model, which is exactly what the
    OpenSim pipeline adds and a SMPL file cannot carry.

    Velocity and acceleration are differences per *frame*, not per second. Everything is
    resampled to one rate before encoding, so the two differ by a constant, and a
    constant is absorbed by normalization.
    """

    channels: tuple[str, ...] = tuple(
        f"{kind}{i}" for kind in ("rot", "vel", "acc") for i in range(6)
    )

    def __init__(self, dofs: tuple[str, ...], excluded: tuple[int, ...]) -> None:
        super().__init__(dofs, excluded)
        self.name = SMPL_NAME

    @classmethod
    def from_human(cls, human: dict) -> "SmplRepresentation":
        """Build the representation from a SMPL joint definition (``config/smpl.yaml``)."""

        joints = human["joints"]
        index = {joint: i for i, joint in enumerate(joints)}
        unknown = [j for j in human.get("excluded_joints") or [] if j not in index]
        if unknown:
            raise KeyError(f"excluded_joints: not in joints: {unknown}")

        excluded = {index[j] for j in human.get("excluded_joints") or []}
        kept = tuple(j for i, j in enumerate(joints) if i not in excluded)
        return cls(kept, tuple(sorted(excluded)))

    def encode(self, motion: np.ndarray) -> np.ndarray:
        """Axis-angle joint rotations ``(T, joints, 3)`` -> features ``(T, kept, 18)``."""

        kept = np.delete(motion, self._excluded, axis=1)
        frames, joints, _ = kept.shape
        if frames < 2:
            raise ValueError("Need at least two frames to difference; got one.")

        rotation = Rotation.from_rotvec(kept.reshape(-1, 3)).as_matrix()
        rot6d = rotation[:, :, :2].reshape(frames, joints, 6)

        vel = np.gradient(rot6d, axis=0)
        acc = np.gradient(vel, axis=0)

        out = np.concatenate([rot6d, vel, acc], axis=-1)
        return np.where(self._mask, signed_log(out), out)

    def decode(self, features: np.ndarray) -> np.ndarray:
        """Features ``(T, kept, 18)`` -> axis-angle joint rotations ``(T, kept, 3)``.

        Only the pose channels are inverted: velocity and acceleration are derived from
        them, so returning them would be returning the same information twice.
        """

        frames, joints, _ = features.shape
        # encode flattened a (3, 2) block row-major, so this is its exact inverse.
        columns = features[:, :, :6].reshape(-1, 3, 2)

        a, b = columns[:, :, 0], columns[:, :, 1]
        x = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b - (x * b).sum(-1, keepdims=True) * x
        y = b / np.linalg.norm(b, axis=-1, keepdims=True)
        matrix = np.stack([x, y, np.cross(x, y)], axis=-1)

        return Rotation.from_matrix(matrix).as_rotvec().reshape(frames, joints, 3)


# Which layout a config file describes is the config's business, not the caller's: the
# datamodule is handed one path and must not know which experiment arm it is running.
def build_representation(path: str | Path) -> Representation:
    """Build the representation the definition at ``path`` describes."""

    human = _load_human_definition(path)
    return (SmplRepresentation if human.get("kind") == "smpl" else Representation).from_human(human)
