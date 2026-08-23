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
on different clocks. Computed by `normalization.py` over a whole sample view — corpus
work, unlike ingest which is per file.
_Avoid_: normalization, norm params

**DOF**:
One degree of freedom of the human model, a joint angle. The DOF axis of a feature
tensor is a per-joint token space.
_Avoid_: joint, channel, column

**Sample**:
One motion take: a stored feature tensor plus its catalog row (provenance, timing,
quality). Identified by `sample_id`, derived from the source path so re-importing lands
on the same id.
_Avoid_: clip, sequence, motion file

**Split set**:
A named partitioning of samples (`babel_official`, `pretrain_v1`); a **split** is one
part of it (`train`, `val`, `test`). Split membership and labels are separate tables:
a sample can belong to a split with no labels published for it.
_Avoid_: dataset, fold

**Processed layout**:
The on-disk contract for `data/processed/`: tensors under `motions/`, parquet tables
under `tables/`, normalization artifacts under `stats/`. Owned by `catalog.py` in both
directions — reads and writes. Nothing else builds a path into the processed root or
writes a table; callers name *what* they want, never *where* it lives.
_Avoid_: storage layer, data dir, repository

**Motion ingest**:
Turning one raw file into one stored sample: load, resample to `TARGET_HZ`, measure
quality on raw motion, encode, write tensor and catalog row. Owned by `preprocess.py`.
There is exactly one ingest path because every corpus reaches us as OpenSim CSV — the
conversion happens upstream. `import_opensim_csv_dataset` is specific about *format*,
not about *corpus*.
_Avoid_: loader, ETL, AMASS importer

**Label source**:
An external annotation corpus: what a segment of motion *is*, plus how that corpus
identifies its sequences. BABEL is the only one today, owned by `babel.py`. It is
responsible for three things: mapping its sequence identifiers onto our `sample_id`s,
importing split membership, and importing labels.

Where the next seam goes: when a second label source arrives, the part that genuinely
varies is the **key mapping**. Everything else is table writing that `catalog.py`
already owns. Do not introduce an importer protocol before then — one adapter is a
hypothetical seam, two is a real one.
_Avoid_: annotation provider, label adapter

**Ontology**:
One of several parallel vocabularies a label source attaches to the same segment.
BABEL carries three: `raw` (annotator free text), `proc` (its processed form),
`act_cat` (entries from the action taxonomy). Stored as rows under an `ontology`
column, not as columns, so a downstream task can ask for exactly one.
_Avoid_: label type, taxonomy, namespace

**Vocabulary**:
The fixed set of labels a benchmark scores, and the integer each maps to. Annotations
say what a sample *is*; a vocabulary says which answers are *admissible* and in what
order. Labels present in the data but absent from the vocabulary are out of scope for
that benchmark, not errors.
_Avoid_: class list, label map, classes

**Corpus policy**:
Which samples constitute a named view, as a decision rather than a mechanism —
`pretrain_v1` is "BABEL train, plus every sample BABEL never saw". Lives in `catalog.py`
beside `build_motion_view`: persisting a view and materializing one are the same
question asked in two directions. Deliberately not in `babel.py` — the policy is "hold
out whatever an evaluation benchmark reserved", which stops being BABEL-shaped as soon
as a second label source exists.
_Avoid_: split strategy, sampling policy
