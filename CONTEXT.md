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
that benchmark, not errors. BABEL ships one frequency-ordered list of 150;
`babel_action_60` and `babel_action_120` are its index prefixes, materialized as their own
`label_set` rows so a benchmark is named rather than carried around as a cutoff. `transition`
belongs to none of them — `action_label_2_idx.json` gives it no index at all, despite its being
19% of `act_cat` rows. The three sets cover 70.3%, 75.3% and 76.0% of annotation rows, so the
90 labels between 60 and 150 are worth 5.7 points of coverage between them.
_Avoid_: class list, label map, classes

**Corpus policy**:
Which samples constitute a named view, as a decision rather than a mechanism —
`pretrain_v1` is "BABEL train, plus every sample BABEL never saw". Lives in `catalog.py`
beside `build_motion_view`: persisting a view and materializing one are the same
question asked in two directions. Deliberately not in `babel.py` — the policy is "hold
out whatever an evaluation benchmark reserved", which stops being BABEL-shaped as soon
as a second label source exists.
_Avoid_: split strategy, sampling policy

**Window**:
The fixed-length crop one batch item carries: `window_frames` frames taken from one sample
by `WindowCollate`. A window is what the model sees; a
patch is how that window is tokenized. Samples shorter than a window are excluded by
`MotionViewSpec.min_frames`, not padded — 22.8% of `pretrain_v1/train` samples but only
8.0% of its frames, and a zero-padded window would otherwise fill the context set with
tokens that score as motionless. Offsets are random while training and
deterministic while evaluating: a random validation crop moves the metric for reasons
unrelated to the model, and `ModelCheckpoint` then selects on crop luck.
_Avoid_: clip, crop, patch

**Patch**:
One token: a single DOF over `patch_size` consecutive frames, so `patch_size * 5` values.
The grid is `TP x 43`, flattened as `t * 43 + d` — the ordering `PositionalEncoding.get_flat()`
already produces. One DOF wide on purpose: 43 is prime, so no uniform spatial grouping
exists, and the DOF axis is already a per-joint token space. `patch_size = 8` at 60 Hz is
133 ms, the same physical duration as MAMP's 4 frames at 30 Hz.
_Avoid_: window, chunk, segment, token grid cell

**Context** and **Target**:
The two halves of a masked window, held as index sets into the flattened token grid:
context is what the encoder sees, target is what the decoder must predict. Disjoint,
exhaustive, and fixed-size once a mask ratio is chosen. Indices rather than a binary
mask, so the loss gathers its targets instead of multiplying across every token, and so
a mask can be plotted without instantiating a model.
_Avoid_: visible/masked, keep/remove, unmasked

**Motion-aware masking**:
Choosing targets in proportion to how much a patch moves, so the informative parts of a
window are the ones held out. The score is `|vel|` meaned over the patch's frames — read
through the channel `Representation` names, never a literal index — turned into a
distribution by `softmax(score / (max * tau))` and drawn without replacement by Gumbel
top-k. `tau` interpolates: small sharpens toward deterministic top-k, large flattens
toward uniform, and `<= 0` is plain random masking, which makes the ablation baseline
the same function rather than a second code path.
_Avoid_: importance sampling, saliency masking, hard mining

**Backbone**:
The `nn.Module` that turns a window into tokens and nothing else: patch projection,
positional encoding, transformer blocks, final norm. Built from an `EncoderSpec` — what a
YAML `encoder:` block maps to, and what travels in a checkpoint's hparams so a classifier
reloads without being told the architecture twice. Every objective owns one (JEPA will own
two, which is why "the encoder" is not a usable name); no objective's decoder, mask token or
prediction head belongs to it. Exposes `embed_tokens` (full grid), `embed` (pooled) and
`grid_shape`. Lives under `architecture/`, which holds no Lightning and no training.
_Avoid_: encoder, feature extractor, trunk

**Student** and **Teacher**:
JEPA's two backbone instances. Student embeds only context tokens and is optimized by
gradient descent; teacher embeds the full token grid, is frozen to gradients, and tracks
student by EMA. Downstream loads teacher by default because probe input is a full
window, while the student only trained on context subsets.
_Avoid_: online encoder, target encoder, momentum encoder

**Predictor**:
JEPA module that receives student context embeddings plus learned target slots and
outputs vectors in the backbone embedding space at target indices. Same width and
heads as the backbone, shallower depth only; no bottleneck projections, no decoder
to patch values.
_Avoid_: decoder, head, projection MLP

**Pretext objective**:
A self-supervised training task over a backbone: what is hidden, what is predicted, what the
loss is. One `LightningModule` per objective under `models/` — masked reconstruction today,
JEPA later. MAE and MAMP are *not* separate objectives: they are one
`MaskedMotionAutoencoder` at two configurations, differing in `tau` (uniform vs motion-aware
masking) and `target` (reconstruct the input vs predict its temporal difference). Both follow
the reference in taking the motion target by differencing *the input the encoder sees*
(`extract_motion`, over the window rather than per patch), and in standardizing every target
token before the loss (`norm_targets`, the reference's `norm_skes_loss`).

The tempting shortcut — select the stored `vel` channel and call that the difference — was
tried and does not work. That channel is signed-log compressed and normalized per DOF, so its
per-frame values are near-unpredictable from context: the objective explained 7.7% of its
target's variance in 10 epochs where pose reconstruction explained 98.8%, and the backbone it
produced probed *below* random initialization. Standardizing per token matters for the same
reason — an unnormalized squared error on motion is carried by the few fastest tokens, which
motion-aware masking has deliberately selected for.
_Avoid_: model, task, head

**Label coverage**:
The fraction of a window's frames spanned by one action category, measured from the `act_cat`
segments overlapping that crop. A window's target is the multi-hot vector of every label whose
coverage reaches `label_min_coverage` (0.15) — the same definition serving the loss and the
metric, so "present" never means two things. Windows are not cut to segment boundaries: at a
median segment length of 1.1 s a 4 s window spans several, which is what multi-label is for.
Deliberately not BABEL's official protocol, which scores one label per chunk and duplicates a
*k*-label segment into *k* samples, capping such a chunk at 50% Top-1. Comparability to
published BABEL numbers is given up on purpose, in exchange for a loss with no ceiling and a
downstream path that reuses the pretraining one. Both frame-level and sequence-level `act_cat` spans
count -- a sequence annotation is a span over `0..dur`, and 40% of sequences carry no frame
annotation at all -- while a sample with no `act_cat` of any kind is *unknown* rather than
negative, and is excluded. Labels outside the chosen vocabulary
contribute nothing, so a window covering only out-of-scope segments trains as an all-negative
example rather than being dropped: dropping would bias evaluation toward the segments that
happen to carry scoreable labels, and at 19% `transition` that bias is not small.
_Avoid_: chunk, multi-hot, annotation

**Protocol**:
How a pretrained backbone is evaluated. *Linear probe*: backbone frozen and in `eval()`,
excluded from the optimizer, pooled over the whole grid to `d_model`, then
`BatchNorm1d(affine=False)` and one `Linear`. *Finetune*: backbone trainable, pooled over time
only so the DOF axis survives into `43 * d_model`, MLP head. Held as independent knobs on
`MotionWindowClassifier` — `pool`, `head`, `freeze_backbone` — never an enum, so a frozen
backbone under a deep head stays expressible as the control it is.
_Avoid_: eval mode, task head, downstream mode
