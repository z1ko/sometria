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

*138 cells across 2 corpora and 3 architectures. Regenerate with `gen.py`.*

*Held out of this report: `amass_motionx_clean`. The runs exist; `EXCLUDE_CORPORA` in `gen.py` is what drops them.*

*An architecture whose name carries an objective suffix (`_jepa`, `_simmim`) was pretrained with that objective; an unsuffixed one is the MAE baseline. JEPA has no reconstruction target, so its sections tabulate the input axis alone and it cannot speak to the loss-channel axis at all.*

*In the matrices, the percentage beside each cell is its relative difference from the best cell of that table: `(cell - best) / best`. It is within-table only, so it compares cells to each other and never across corpora or architectures.*

## Macro mAP (per patch)

![Each matrix on its own color scale](figures/macro_map_matrix.png)

*Each matrix on its own color scale.*

![What each axis buys, per architecture](figures/macro_map_ladder.png)

*What each axis buys, per architecture.*

![Corpus against corpus, cell by cell](figures/macro_map_corpora.png)

*Corpus against corpus, cell by cell.*

### `amass_clean` / medium_100ep (4 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | 0.2738 ±0.0033 (-11.1%) | 0.3013 ±0.0028 (-2.2%) | 0.2992 ±0.0051 (-2.8%) | 0.3008 ±0.0037 (-2.3%) |
| **pv** | 0.2826 ±0.0014 (-8.2%) | 0.3048 ±0.0070 (-1.0%) | 0.2989 ±0.0004 (-2.9%) | 0.3067 ±0.0055 (-0.4%) |
| **pk** | 0.2817 ±0.0053 (-8.5%) | **0.3079** ±0.0040 (+0.0%) | 0.3046 ±0.0032 (-1.1%) | 0.3040 ±0.0031 (-1.3%) |
| **pkd** | 0.2837 ±0.0015 (-7.9%) | 0.3068 ±0.0021 (-0.4%) | 0.2995 ±0.0027 (-2.7%) | 0.2986 ±0.0022 (-3.0%) |

### `amass_clean` / medium_100ep_jepa (4 seeds, input axis only)

| Input | Macro mAP (per patch) |
| :--- | :---: |
| **p** | 0.1938 ±0.0223 (-28.3%) |
| **pv** | 0.2043 ±0.0112 (-24.4%) |
| **pk** | 0.2339 ±0.0332 (-13.4%) |
| **pkd** | **0.2701** ±0.0101 (+0.0%) |

### `amass_clean` / medium_100ep_simmim (3 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | 0.1341 ±0.0096 (-43.4%) | 0.1872 ±0.0192 (-21.0%) | 0.1995 ±0.0018 (-15.8%) | 0.2051 ±0.0071 (-13.5%) |
| **pv** | 0.1234 ±0.0151 (-47.9%) | 0.1904 ±0.0054 (-19.7%) | 0.1890 ±0.0062 (-20.3%) | **0.2371** ±0.0056 (+0.0%) |
| **pk** | 0.1196 ±0.0135 (-49.5%) | 0.1961 ±0.0027 (-17.3%) | 0.1789 ±0.0153 (-24.5%) | 0.2345 ±0.0031 (-1.1%) |
| **pkd** | 0.1059 ±0.0154 (-55.3%) | 0.1863 ±0.0086 (-21.4%) | 0.1779 ±0.0127 (-25.0%) | 0.1993 ±0.0057 (-15.9%) |

### `amass_smpl` / medium_100ep (3 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | 0.2891 ±0.0033 (-8.0%) | **0.3141** ±0.0050 (+0.0%) | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

### `amass_smpl` / medium_100ep_jepa (2 seeds, input axis only)

| Input | Macro mAP (per patch) |
| :--- | :---: |
| **p** | **0.2252** ±0.0104 (+0.0%) |
| **pv** | -- |
| **pk** | -- |
| **pkd** | -- |

### `amass_smpl` / medium_100ep_simmim (2 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | **0.2135** ±0.0002 (+0.0%) | -- | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

### Delta, `amass_smpl` − `amass_clean` (medium_100ep)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | +0.0153 (+6.0σ) | +0.0129 (+5.0σ) | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0033 is the seed spread pooled over cells; with n = 4 vs 3 runs the difference of two means has SE = 0.0025, so 1σ here = 0.0025 Macro mAP (per patch). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 16-cell matrix and picking the largest needs |z| > 2.96 to keep the same 5% false-positive rate over the table.*

### Delta, `amass_smpl` − `amass_clean` (medium_100ep_jepa)

| Input \ Loss | -- |
| :--- | :---: |
| **p** | +0.0314 (+1.9σ) |
| **pv** | -- |
| **pk** | -- |
| **pkd** | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0192 is the seed spread pooled over cells; with n = 4 vs 2 runs the difference of two means has SE = 0.0166, so 1σ here = 0.0166 Macro mAP (per patch). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 4-cell matrix and picking the largest needs |z| > 2.50 to keep the same 5% false-positive rate over the table.*

### Delta, `amass_smpl` − `amass_clean` (medium_100ep_simmim)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | +0.0794 (+9.5σ) | -- | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0092 is the seed spread pooled over cells; with n = 3 vs 2 runs the difference of two means has SE = 0.0084, so 1σ here = 0.0084 Macro mAP (per patch). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 16-cell matrix and picking the largest needs |z| > 2.96 to keep the same 5% false-positive rate over the table.*

## Boundary F1 (swept)

![Each matrix on its own color scale](figures/boundary_best_f1s_matrix.png)

*Each matrix on its own color scale.*

![What each axis buys, per architecture](figures/boundary_best_f1s_ladder.png)

*What each axis buys, per architecture.*

![Corpus against corpus, cell by cell](figures/boundary_best_f1s_corpora.png)

*Corpus against corpus, cell by cell.*

### `amass_clean` / medium_100ep (4 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | 0.1611 ±0.0015 (-4.9%) | 0.1640 ±0.0009 (-3.2%) | 0.1637 ±0.0002 (-3.4%) | 0.1668 ±0.0031 (-1.5%) |
| **pv** | 0.1640 ±0.0017 (-3.2%) | 0.1681 ±0.0053 (-0.8%) | 0.1597 ±0.0023 (-5.7%) | 0.1683 ±0.0036 (-0.7%) |
| **pk** | 0.1621 ±0.0014 (-4.3%) | 0.1684 ±0.0025 (-0.6%) | 0.1615 ±0.0018 (-4.7%) | **0.1694** ±0.0022 (+0.0%) |
| **pkd** | 0.1631 ±0.0009 (-3.7%) | 0.1656 ±0.0036 (-2.2%) | 0.1621 ±0.0033 (-4.3%) | 0.1614 ±0.0017 (-4.7%) |

### `amass_clean` / medium_100ep_jepa (4 seeds, input axis only)

| Input | Boundary F1 (swept) |
| :--- | :---: |
| **p** | 0.1512 ±0.0025 (-7.1%) |
| **pv** | 0.1577 ±0.0013 (-3.1%) |
| **pk** | 0.1575 ±0.0025 (-3.2%) |
| **pkd** | **0.1627** ±0.0069 (+0.0%) |

### `amass_clean` / medium_100ep_simmim (3 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | 0.1514 ±0.0025 (-1.3%) | 0.1523 ±0.0008 (-0.7%) | 0.1509 ±0.0011 (-1.7%) | 0.1509 ±0.0009 (-1.7%) |
| **pv** | 0.1526 ±0.0012 (-0.5%) | 0.1513 ±0.0012 (-1.4%) | 0.1505 ±0.0004 (-1.9%) | 0.1525 ±0.0009 (-0.6%) |
| **pk** | 0.1519 ±0.0010 (-1.0%) | 0.1505 ±0.0008 (-1.9%) | 0.1503 ±0.0010 (-2.1%) | **0.1534** ±0.0006 (+0.0%) |
| **pkd** | 0.1516 ±0.0012 (-1.2%) | 0.1511 ±0.0010 (-1.5%) | 0.1498 ±0.0008 (-2.3%) | 0.1504 ±0.0005 (-2.0%) |

### `amass_smpl` / medium_100ep (3 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | **0.1634** ±0.0020 (+0.0%) | 0.1576 ±0.0016 (-3.6%) | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

### `amass_smpl` / medium_100ep_jepa (2 seeds, input axis only)

| Input | Boundary F1 (swept) |
| :--- | :---: |
| **p** | **0.1483** ±0.0027 (+0.0%) |
| **pv** | -- |
| **pk** | -- |
| **pkd** | -- |

### `amass_smpl` / medium_100ep_simmim (2 seeds)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | **0.1513** ±0.0011 (+0.0%) | -- | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

### Delta, `amass_smpl` − `amass_clean` (medium_100ep)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | +0.0024 (+1.4σ) | -0.0064 (-3.7σ) | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0023 is the seed spread pooled over cells; with n = 4 vs 3 runs the difference of two means has SE = 0.0017, so 1σ here = 0.0017 Boundary F1 (swept). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 16-cell matrix and picking the largest needs |z| > 2.96 to keep the same 5% false-positive rate over the table.*

### Delta, `amass_smpl` − `amass_clean` (medium_100ep_jepa)

| Input \ Loss | -- |
| :--- | :---: |
| **p** | -0.0029 (-1.0σ) |
| **pv** | -- |
| **pk** | -- |
| **pkd** | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0033 is the seed spread pooled over cells; with n = 4 vs 2 runs the difference of two means has SE = 0.0028, so 1σ here = 0.0028 Boundary F1 (swept). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 4-cell matrix and picking the largest needs |z| > 2.50 to keep the same 5% false-positive rate over the table.*

### Delta, `amass_smpl` − `amass_clean` (medium_100ep_simmim)

| Input \ Loss | p | pv | pk | pkd |
| :--- | :---: | :---: | :---: | :---: |
| **p** | -0.0001 (-0.1σ) | -- | -- | -- |
| **pv** | -- | -- | -- | -- |
| **pk** | -- | -- | -- | -- |
| **pkd** | -- | -- | -- | -- |

*Each cell reads `delta (z)`. The σ figure **is** z = delta / SE: how many standard errors of the difference the delta is, not how many seed sd's. σ = 0.0010 is the seed spread pooled over cells; with n = 3 vs 2 runs the difference of two means has SE = 0.0009, so 1σ here = 0.0009 Boundary F1 (swept). |z| > 2 is marginal (p < 0.05 for one pre-chosen cell); reading the whole 16-cell matrix and picking the largest needs |z| > 2.96 to keep the same 5% false-positive rate over the table.*

## Can the cells be ranked?

| corpus | arch | cells | range | seed σ | range / σ | noise bar | verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| amass_clean | medium_100ep | 16 | 0.0341 | 0.0033 | 10.23 | 3.53 | **above noise** |
| amass_clean | medium_100ep_jepa | 4 | 0.0764 | 0.0192 | 3.98 | 2.06 | **above noise** |
| amass_clean | medium_100ep_simmim | 16 | 0.1312 | 0.0092 | 14.29 | 3.53 | **above noise** |
| amass_smpl | medium_100ep | 2 | 0.0250 | 0.0041 | 6.02 | 1.13 | **above noise** |

*Range across the cell means over the seed sd pooled across cells. The bar is E[range] of that many draws from pure noise -- nine identical cells still spread, so the comparison is against ~3σ, not 0. Above the bar the matrix carries structure worth ranking; below it, the best cell is a selection effect.*

## Marginal level of each axis

| corpus | arch | axis | `p` | `pv` | `pk` | `pkd` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| amass_clean | medium_100ep | input | 0.2938 | 0.2982 | 0.2995 | 0.2972 |
| amass_clean | medium_100ep | loss | 0.2804 | 0.3052 | 0.3006 | 0.3025 |
| amass_clean | medium_100ep_jepa | input | 0.1938 | 0.2043 | 0.2339 | 0.2701 |
| amass_clean | medium_100ep_simmim | input | 0.1815 | 0.1850 | 0.1823 | 0.1673 |
| amass_clean | medium_100ep_simmim | loss | 0.1207 | 0.1900 | 0.1863 | 0.2190 |
| amass_smpl | medium_100ep | input | 0.3016 | -- | -- | -- |
| amass_smpl | medium_100ep | loss | 0.2891 | 0.3141 | -- | -- |
| amass_smpl | medium_100ep_jepa | input | 0.2252 | -- | -- | -- |
| amass_smpl | medium_100ep_simmim | input | 0.2135 | -- | -- | -- |
| amass_smpl | medium_100ep_simmim | loss | 0.2135 | -- | -- | -- |

*Each value averages over the other axis. Levels only -- whether a difference between them is real is the next table.*

## Where along the ladder the gain appears

| corpus | arch | axis | `p`→`pv` | `pv`→`pk` | `pk`→`pkd` | total `p`→`pkd` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| amass_clean | medium_100ep | input | +0.0045 (0.099) | +0.0013 (0.325) | -0.0024 (0.070) | +0.0034 (0.040) |
| amass_clean | medium_100ep | loss | **+0.0247** (<0.001) | -0.0046 (0.079) | +0.0020 (0.260) | **+0.0221** (<0.001) |
| amass_clean | medium_100ep_jepa | input | +0.0105 (0.550) | +0.0296 (0.130) | +0.0362 (0.092) | +0.0764 (0.014) |
| amass_clean | medium_100ep_simmim | input | +0.0035 (0.057) | -0.0027 (0.310) | -0.0150 (0.027) | -0.0142 (0.002) |
| amass_clean | medium_100ep_simmim | loss | +0.0693 (0.003) | -0.0037 (0.258) | +0.0327 (0.006) | +0.0982 (0.002) |
| amass_smpl | medium_100ep | input | -- | -- | -- | -- |
| amass_smpl | medium_100ep | loss | +0.0250 (0.017) | -- | -- | -- |
| amass_smpl | medium_100ep_jepa | input | -- | -- | -- | -- |
| amass_smpl | medium_100ep_simmim | input | -- | -- | -- | -- |
| amass_smpl | medium_100ep_simmim | loss | -- | -- | -- | -- |

*Paired step to the next rung, signed so positive means widening helped. Cells share a seed and seed spread is several times these effects, so each step is paired on replicate (n = 4) rather than compared as independent means. Parenthesis is the two-sided paired p; bold is significant at 0.05 Bonferroni-corrected over the 40 steps and totals shown. At three replicates a p rests on two degrees of freedom -- read direction and magnitude before decisions.*

## Best cell per corpus

| corpus | arch | cell | runs | macro_map | boundary_best_f1s |
| :--- | :--- | :--- | ---: | ---: | ---: |
| amass_clean | medium_100ep | `in_pk__loss_pv` | 4 | 0.3079 | 0.1684 |
| amass_clean | medium_100ep_jepa | `in_pkd` | 4 | 0.2701 | 0.1627 |
| amass_clean | medium_100ep_simmim | `in_pv__loss_pkd` | 3 | 0.2371 | 0.1525 |
| amass_smpl | medium_100ep | `in_p__loss_pv` | 3 | 0.3141 | 0.1576 |
| amass_smpl | medium_100ep_jepa | `in_p` | 2 | 0.2252 | 0.1483 |
| amass_smpl | medium_100ep_simmim | `in_p__loss_p` | 2 | 0.2135 | 0.1513 |

## Replicate spread

| corpus | arch | runs/cell | cells | sd macro_map | sd boundary_best_f1s |
| :--- | :--- | ---: | ---: | ---: | ---: |
| amass_clean | medium_100ep | 4 | 16 | 0.0033 | 0.0023 |
| amass_clean | medium_100ep_jepa | 4 | 4 | 0.0192 | 0.0033 |
| amass_clean | medium_100ep_simmim | 3 | 16 | 0.0092 | 0.0010 |
| amass_smpl | medium_100ep | 3 | 2 | 0.0041 | 0.0018 |
| amass_smpl | medium_100ep_jepa | 2 | 1 | 0.0104 | 0.0027 |
| amass_smpl | medium_100ep_simmim | 2 | 1 | 0.0002 | 0.0011 |

## Reference points

| baseline | Macro mAP (per patch) | Boundary F1 (swept) | vs best cell | protocol |
| :--- | ---: | ---: | ---: | :--- |
| chance | 0.0421 | 0.0000 | 7.54x | exact, no model |
| moments (patch) | 0.1906 | 0.1565 | 1.67x | matched data/labels/metrics, AdamW head |
| moments (window) | 0.1837 | 0.0000 | 1.73x | matched data/labels/metrics, AdamW head |

*Best cell: 0.3174 macro mAP, taken over every architecture present, so it is the best result on the board rather than the best MAE one.*

- **chance** -- Average precision of a random ranking equals the class positive rate, carried over from the window benchmark because the vocabulary and the split are the same. Boundary F1 is entered as 0 rather than a positive rate: the boundary metric scores only patches where the annotation changes, and a random prediction has no reason to change there in particular.
- **moments (patch)** -- mean, std, min and max over each time patch's own 8 frames -- statistics that may vary in time, but see no context on either side. This is the real competitor for a segmentation probe, and the number that makes this page worth having: the best cell beats it by 61.5% on macro mAP, against 19.5% on the BABEL-chunk benchmark and 12% on tiled windows. Run via `scripts/moments_baseline.py --config config/experiment_segmentation.yaml`.
- **moments (window)** -- The same four statistics over the whole window, one prediction broadcast to every patch -- it can name the action and can never say when. Its boundary F1 is 0.0000 measured, not assumed, which is the check that the boundary metric does what it claims. Note how little the patch arm gains over it on mAP (0.1906 against 0.1837): almost all of the moments' score here is window-level naming, not localization.

## Provenance

| corpus | arch | replicates | normalization |
| :--- | :--- | :--- | :--- |
| amass_clean | medium_100ep | seed1, seed2, seed3, seed42 | `pretrain_v1_stats_train_clean.pt` |
| amass_clean | medium_100ep_jepa | seed1, seed2, seed3, seed42 | `pretrain_v1_stats_train_clean.pt` |
| amass_clean | medium_100ep_simmim | seed1, seed2, seed42 | `pretrain_v1_stats_train_clean.pt` |
| amass_smpl | medium_100ep | seed1, seed2, seed42 | `pretrain_paired_v1_train_clean.pt` |
| amass_smpl | medium_100ep_jepa | seed1, seed42 | `pretrain_paired_v1_train_clean.pt` |
| amass_smpl | medium_100ep_simmim | seed1, seed42 | `pretrain_paired_v1_train_clean.pt` |

<!-- /GENERATED -->
