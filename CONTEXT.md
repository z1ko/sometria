# Sometria

Self-supervised representation learning on human motion: OpenSim biomechanical
takes imported from AMASS, encoded into per-joint feature tensors, then modelled.

## Language

**Representation**:
Meaning of a feature channel — which channels exist, which DOFs survive, how
raw physical units and model units convert. It a value, carries own name, holds no stats.
_Avoid_: feature spec, encoding config, feature builder

**Raw motion**:
Kinematic and dynamic channels as OpenSim produced them, physical units:
position, velocity, acceleration, torque. Quality judged here, never after encoding.
_Avoid_: raw features, signal

**Features**:
What Representation produces from raw motion, what stored on disk:
`sin, cos, vel, acc, tau` per kept DOF, derivative channels signed-log compressed.
_Avoid_: encoded motion, tensors

**Stats**:
Per-DOF, per-channel mean and std over one training split. Passed to
Representation as argument, never held by it: layout and statistics change
on different clocks. Computed by `normalization.py` over whole sample view — corpus
work, unlike per-file ingest.
_Avoid_: normalization, norm params

**DOF**:
One degree of freedom of human model, a joint angle. DOF axis of feature
tensor a per-joint token space.
_Avoid_: joint, channel, column

**Sample**:
One motion take: stored feature tensor plus catalog row (provenance, timing,
quality). Identified by `sample_id`, derived from source path so re-import lands
on same id.
_Avoid_: clip, sequence, motion file

**Split set**:
Named partitioning of samples (`babel_official`, `pretrain_v1`); a **split** one
part of it (`train`, `val`, `test`). Split membership and labels separate tables:
sample can belong to split with no labels published.
_Avoid_: dataset, fold

**Processed layout**:
On-disk contract for `data/processed/`: tensors under `motions/`, parquet tables
under `tables/`, normalization artifacts under `stats/`. Owned by `catalog.py` both
directions — reads and writes. Nothing else builds path into processed root or
writes table; callers name *what* they want, never *where* it lives.
_Avoid_: storage layer, data dir, repository

**Motion ingest**:
Turn one raw file into one stored sample: load, resample to `TARGET_HZ`, measure
quality on raw motion, encode, write tensor and catalog row. Owned by `preprocess.py`.
Exactly one ingest path because every corpus reaches us as OpenSim CSV — conversion
happens upstream. `import_opensim_csv_dataset` specific about *format*,
not *corpus*.
_Avoid_: loader, ETL, AMASS importer

**Label source**:
External annotation corpus: what a motion segment *is*, plus how that corpus
identifies its sequences. BABEL only one today, owned by `babel.py`. Responsible for
three things: map its sequence identifiers onto our `sample_id`s,
import split membership, import labels.

Where next seam goes: when second label source arrives, part that genuinely
varies the **key mapping**. Rest table writing `catalog.py`
already owns. No importer protocol before then — one adapter a
hypothetical seam, two a real one.
_Avoid_: annotation provider, label adapter

**Ontology**:
One of several parallel vocabularies a label source attaches to same segment.
BABEL carries three: `raw` (annotator free text), `proc` (its processed form),
`act_cat` (entries from action taxonomy). Stored as rows under an `ontology`
column, not columns, so downstream task can ask for exactly one.
_Avoid_: label type, taxonomy, namespace

**Vocabulary**:
Fixed set of labels a benchmark scores, and integer each maps to. Annotations
say what sample *is*; vocabulary says which answers *admissible* and in what
order. Labels present in data but absent from vocabulary out of scope for
that benchmark, not errors. BABEL ships one frequency-ordered list of 150;
`babel_action_60` and `babel_action_120` its index prefixes, materialized as own
`label_set` rows so benchmark named rather than carried as a cutoff. `transition`
belongs to none — `action_label_2_idx.json` gives it no index at all, despite being
19% of `act_cat` rows. Three sets cover 70.3%, 75.3% and 76.0% of annotation rows, so
90 labels between 60 and 150 worth 5.7 points of coverage between them.
_Avoid_: class list, label map, classes

**Corpus policy**:
Which samples constitute a named view, a decision not a mechanism —
`pretrain_v1` is "BABEL train, plus every sample BABEL never saw". Lives in `catalog.py`
beside `build_motion_view`: persisting a view and materializing one same
question asked two directions. Deliberately not in `babel.py` — policy is "hold
out whatever an evaluation benchmark reserved", which stops being BABEL-shaped as soon
as second label source exists.
_Avoid_: split strategy, sampling policy

**Window**:
Fixed-length crop one batch item carries: `window_frames` frames taken from one sample
by `WindowCollate`. Window what model sees; a
patch how that window tokenized. Samples shorter than a window excluded by
`MotionViewSpec.min_frames`, not padded — 22.8% of `pretrain_v1/train` samples but only
8.0% of its frames, and zero-padded window would fill context set with
tokens scoring as motionless. Offsets random while training,
deterministic while evaluating: random validation crop moves metric for reasons
unrelated to model, and `ModelCheckpoint` then selects on crop luck.
_Avoid_: clip, crop, patch

**Patch**:
One token: single DOF over `patch_size` consecutive frames, so `patch_size * 5` values.
Grid `TP x 43`, flattened as `t * 43 + d` — ordering `PositionalEncoding.get_flat()`
already produces. One DOF wide on purpose: 43 prime, so no uniform spatial grouping
exists, and DOF axis already a per-joint token space. `patch_size = 8` at 60 Hz is
133 ms, same physical duration as MAMP's 4 frames at 30 Hz.
_Avoid_: window, chunk, segment, token grid cell

**Context** and **Target**:
Two halves of masked window, held as index sets into flattened token grid:
context what encoder sees, target what decoder must predict. Disjoint,
exhaustive, fixed-size once mask ratio chosen. Indices not binary
mask, so loss gathers targets instead of multiplying across every token, and so
mask can be plotted without instantiating a model.
_Avoid_: visible/masked, keep/remove, unmasked

**Motion-aware masking**:
Choose targets in proportion to how much a patch moves, so informative parts of
window are the ones held out. Score `|vel|` meaned over patch's frames — read
through channel `Representation` names, never a literal index — turned into
distribution by `softmax(score / (max * tau))` and drawn without replacement by Gumbel
top-k. `tau` interpolates: small sharpens toward deterministic top-k, large flattens
toward uniform, `<= 0` plain random masking, which makes ablation baseline
same function rather than second code path.
_Avoid_: importance sampling, saliency masking, hard mining

**Backbone**:
`nn.Module` that turns window into tokens and nothing else: patch projection,
positional encoding, transformer blocks, final norm. Built from `EncoderSpec` — what a
YAML `encoder:` block maps to, and what travels in checkpoint's hparams so classifier
reloads without being told architecture twice. Every objective owns one (JEPA owns
two, why "the encoder" not a usable name); no objective's decoder, mask token or
prediction head belongs to it. Exposes `embed_tokens` (full grid), `embed` (pooled),
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
Self-supervised training task over a backbone: what hidden, what predicted, what
loss is. One `LightningModule` per objective under `models/` — masked reconstruction today,
JEPA later. MAE and MAMP *not* separate objectives: one
`MaskedMotionAutoencoder` at two configurations, differing in `tau` (uniform vs motion-aware
masking) and `target` (reconstruct input vs predict its temporal difference). Both follow
reference in taking motion target by differencing *the input the encoder sees*
(`extract_motion`, over window not per patch), and in standardizing every target
token before loss (`norm_targets`, reference's `norm_skes_loss`).

Tempting shortcut — select stored `vel` channel and call that the difference — was
tried, does not work. That channel signed-log compressed and normalized per DOF, so its
per-frame values near-unpredictable from context: objective explained 7.7% of its
target's variance in 10 epochs where pose reconstruction explained 98.8%, and backbone it
produced probed *below* random init. Standardizing per token matters same
reason — unnormalized squared error on motion carried by few fastest tokens, which
motion-aware masking deliberately selected for.
_Avoid_: model, task, head

**Label coverage**:
Fraction of a window's frames spanned by one action category, measured from `act_cat`
segments overlapping that crop. Window's target multi-hot vector of every label whose
coverage reaches `label_min_coverage` (0.15) — same definition serving loss and
metric, so "present" never means two things. Windows not cut to segment boundaries: at
median segment length 1.1 s a 4 s window spans several, what multi-label for.
Deliberately not BABEL's official protocol, which scores one label per chunk and duplicates a
*k*-label segment into *k* samples, capping such chunk at 50% Top-1. Comparability to
published BABEL numbers given up on purpose, in exchange for loss with no ceiling and
downstream path reusing pretraining one. Both frame-level and sequence-level `act_cat` spans
count -- sequence annotation a span over `0..dur`, and 40% of sequences carry no frame
annotation at all -- while a sample with no `act_cat` of any kind *unknown* rather than
negative, excluded. Labels outside chosen vocabulary
contribute nothing, so window covering only out-of-scope segments trains as all-negative
example rather than dropped: dropping would bias evaluation toward segments that
happen to carry scoreable labels, and at 19% `transition` that bias not small.
_Avoid_: chunk, multi-hot, annotation

**Protocol**:
How pretrained backbone evaluated. *Linear probe*: backbone frozen and in `eval()`,
excluded from optimizer, pooled over whole grid to `d_model`, then
`BatchNorm1d(affine=False)` and one `Linear`. *Finetune*: backbone trainable, pooled over time
only so DOF axis survives into `43 * d_model`, MLP head. Held as independent knobs on
`MotionWindowClassifier` — `pool`, `head`, `freeze_backbone` — never an enum, so frozen
backbone under deep head stays expressible as the control it is.
_Avoid_: eval mode, task head, downstream mode
