# BABEL-60 segmentation probe

The convex probe moved one axis over: same frozen backbones, same BABEL-60 vocabulary,
same L-BFGS fit, with the target resolved **per time patch** instead of collapsed over
the window. `mean` pooling reduces the DOF axis only, so the backbone's `TP x 43` grid
becomes `30 x d_model` — one row per 133 ms patch, thirty supervised rows per window
(`sometria.downstream.segmentation_convex.MotionConvexSegmenter`).

It exists because window classification is weakly discriminative. A linear model on the
mean, std, min and max of each channel reaches 94% of a pretrained probe there, and the
cells this matrix ranks differ by 0.003–0.010 macro mAP. Summary statistics are
order-invariant by construction, so they cannot produce a target that varies *inside* the
window; this task can separate a representation that encodes time from one that does not,
and the window task cannot.

## How to read the two columns

**Macro mAP (per patch) is the ranking.** Same unit as the published window mAP, computed
over `(window, patch)` rows.

**Boundary F1 is a gate, not a rank.** It scores only where the annotation *changes*
between consecutive patches, so a prediction that is constant across the window earns zero
however well it names the action. A cell that beats the moments baseline on mAP but not on
boundary F1 has not demonstrated temporal structure, and this page says so in words rather
than leaving it to the table. It is reported swept over thresholds (`boundary_best_f1s`,
with the winning cutoff in `metrics.csv`): the median patch carries ~2 of 60 labels, so a
calibrated head keeps every score far below 0.5 and a fixed cutoff reports calibration
instead of localization.

Neither column selects `weight_decay` except the first. Selection is on macro mAP alone —
boundary F1 moves with where the threshold falls as much as with the model, and tuning the
regularizer on the gate would leave nothing independent holding the ranking honest.

## Settings

Data settings come from `config/experiment_segmentation.yaml` and are not revisited:
`label_min_coverage` 0.5 (a patch is 8 frames, so a segment either covers most of it or
misses it), `label_types: ["frame"]` (BABEL's sequence-level spans run `0..dur` and are
constant across every window, so they cannot produce a boundary), 240-frame windows, 30
time patches, boundary rate 0.080.

`train_passes` is **2**, not the window probe's 20. Rows here are
`passes x windows x 30 patches`, so the feature matrix is thirty times taller per pass; the
20 passes exist to recover crop augmentation, and one window already yields 30 supervised
rows that differ from each other. `check_feature_memory` refuses before extraction rather
than letting a full-batch L-BFGS discover it mid-solve.

Backbones are the 3x3 `input channels x loss channels` pretraining matrix: `p` is
`sin, cos`, `pk` adds `vel, acc`, `pkd` adds `tau`. Best `epoch=*.ckpt` per cell.

The pooling is `mean` and is not a knob. `attentive` carries its own weights, which makes
the fit non-convex — the AdamW/attentive `MotionSegmenter` stays available as an optional
ceiling run, and its number does not belong in this table.

## Baseline

`scripts/moments_baseline.py` run under **this same config** is the floor, and it reports
two arms once `label_patches` is set: `patch` (moments of each patch's own 8 frames — the
real competitor) and `window` (one window-level prediction broadcast to every patch, whose
boundary F1 is exactly 0 by construction). Both land in `baselines.json`, in the same
units and on the same split.

## Running

```
./scripts/segmentation_matrix.sh                       # or SEEDS="42 1 2" / OBJECTIVE=jepa
uv run python scripts/probe_segmentation.py --self-check
uv run python results/probe/babel_60_segmentation/gen.py
```

Read from `runs/probe/babel_60_segmentation/`. **No runs exist yet** — the generated block
below is empty until `segmentation_matrix.sh` has written some.

## Parked, and what to check first

One cell (`in_pk__loss_pk`, seed 1) has run. **Boundary F1 did not clear its trivial
floor by enough to trust, and that floor was never in the plan.**

`boundary_f1` scores F1 over label *changes* between consecutive patches. On this
validation split -- 3,669 windows, 106,401 patch transitions -- the boundary rate is
0.0800, so a predictor claiming a change at *every* transition gets recall 1.0,
precision 0.0800, and **F1 = 0.1481** with nothing fit at all.

| | value | vs trivial |
| :--- | ---: | ---: |
| trivial "always a boundary" | 0.1481 | -- |
| probe `boundary_best_f1s` (thr 0.20) | 0.1640 | +0.0159 |
| probe `boundary_f1s` @ 0.5 | 0.1281 | -0.0200 |
| best `boundary_best_f1s` anywhere in the wd sweep (3e-05) | 0.1690 | +0.0209 |

The +0.0159 is optimistic on top of that: `best_boundary_f1` maximizes over 30 thresholds
on validation, while the trivial predictor has no threshold to tune, so the selection bias
runs one way only.

Per-patch `macro_map` was 0.3001 (window-level convex probe on the same cell: 0.3658), at
`weight_decay` 3e-06 -- interior to the grid, so not censored. It is unreadable until the
moments floor runs.

What to check before spending the 3x3 x 3 sweep:

1. **Run the moments floor.** `uv run python scripts/moments_baseline.py --config
   config/experiment_segmentation.yaml`. Nothing here means anything without it.
2. **Add the trivial all-change ceiling as a permanent baseline.** The standing rule is
   that every task publishes its floor; for boundary F1 the binding floor is this constant
   predictor, not moments. One pass over labels, no backbone.
3. **Score the baseline through the same threshold sweep**, or score the probe at a fixed
   threshold, so the two are read the same way.
4. **The weight-decay sweep selects on `macro_map` alone**, and `boundary_f1s` is still
   rising as wd falls (0.1358 at 1e-08). The selected point is not boundary F1's best, so
   one metric is being reported off the other's optimum.
5. Whether `label_min_coverage: 0.5` and `label_types: ["frame"]` leave enough boundaries
   to score: 0.0800 is sparse, and it is what makes the trivial predictor competitive.


<!-- GENERATED by gen.py -- do not edit below this line -->
<!-- /GENERATED -->
