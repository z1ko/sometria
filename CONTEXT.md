# Sometria

Self-supervised representation learning on human motion: OpenSim biomechanical
takes imported from AMASS, encoded into per-joint feature tensors, and modelled.

## Language

**Representation**:
The meaning of a feature channel — which channels exist, which DOFs survive, and how
raw physical units and model units convert into each other. It is a value, carries its
own name, and holds no statistics.
_Avoid_: feature spec, encoding config, feature builder

**Raw motion**:
Kinematic and dynamic channels as OpenSim produced them, in physical units:
position, velocity, acceleration, torque. Quality is judged here, never after encoding.
_Avoid_: raw features, signal

**Features**:
What a Representation produces from raw motion and what is stored on disk:
`sin, cos, vel, acc, tau` per kept DOF, with the derivative channels signed-log compressed.
_Avoid_: encoded motion, tensors

**Stats**:
Per-DOF, per-channel mean and std measured over one training split. Passed to a
Representation as an argument, never held by it: the layout and the statistics change
on different clocks.
_Avoid_: normalization, norm params

**DOF**:
One degree of freedom of the human model, a joint angle. The DOF axis of a feature
tensor is a per-joint token space.
_Avoid_: joint, channel, column

**Sample**:
One motion take: a stored feature tensor plus its catalog row (provenance, timing,
quality). Identified by `sample_id`.
_Avoid_: clip, sequence, motion file

**Split set**:
A named partitioning of samples (`babel_official`, `pretrain_v1`); a **split** is one
part of it (`train`, `val`, `test`). Split membership and labels are separate tables:
a sample can belong to a split with no labels published for it.
_Avoid_: dataset, fold
