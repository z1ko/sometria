# Representation

What the pretrained backbones *are*, measured before anything is fit on top of them. Three
things share this directory because they share one feature extraction pass and answer one
question between them: geometry (phase 0 of `EVALUATION_PLAN.md`), label efficiency, and
kNN retrieval (phase 2).

The reason any of it exists: the matrix is ranked today by window-level multi-label mAP
under a convex probe and a fine-tune, the cells differ there by 0.003-0.010, and a linear
model on four summary statistics per channel reaches 94% of a pretrained probe's score.
Each measurement below attacks that from a different side -- geometry can explain a score
before it is measured, the curve asks for fewer labels rather than a higher score, and kNN
drops the head entirely.

## Geometry (phase 0)

Cheapest thing here and the first to run, because it can invalidate a later table before
that table is measured: if a cell's pooled features are near rank-1, every task score for
that cell is explained before anyone scores it.

**Participation ratio**, `(sum lambda)^2 / sum lambda^2` over the eigenvalues of the
feature covariance. It reads as "how many directions the features effectively use": `d`
when the covariance is isotropic, 1 when it is rank-1, and variance-weighted in between,
which is the point -- a covariance whose 200th eigenvalue is 1e-9 has full rank and one
useful direction.

On the pooled features the convex probe fits, through the probe's own
`MotionConvexClassifier` -- same pooler, same parameter-free `LayerNorm`. The statistic is
only about `results/probe/babel_60_convex/`'s table if it is measured on that table's exact
tensor, and it explains that table and nothing else, which is why it is not measured on
raw tokens or on a pooling of convenience.

**Linear CKA** between cells, all pairs, with one cell's cross-seed CKA beside it as the
noise floor. The cross-cell number alone says nothing: the question is always whether two
cells differ by more than two seeds of one cell do, which is how
`results/decoder_capacity/README.md` reads its 0.934 against 0.959. Reported as a pair
here for the same reason.

**The validation split, tiled, one pass** -- not the training split the probe literally
fits. `LabelledWindows(tiles=False)` draws a random crop per sample per pass, so two cells'
training features are two different sets of windows and the row alignment CKA requires does
not exist. Evaluation tiles are deterministic, so cell A's row *i* and cell B's row *i* are
the same frames by construction, and the participation ratio is then read off that same
matrix rather than a second one.

### What it showed

**Nothing is collapsed**, which is what this phase was run first to find out. All nine MAE
cells use 15.5-29.0 of 256 available directions, so no later table is pre-explained by a
degenerate representation and phase 1 could start. That is the gate, and it is the boring
answer; the interesting one is below it.

**Torque in the loss expands the representation -- but only when torque is not already in
the input.** `in_pk/loss_pkd` reaches 28.98 where `in_pkd/loss_pkd` sits at 15.53: a gap of
13.45 against a pooled seed sd of 1.31, about ten sigma, and the largest effect anywhere in
the table. The same pair differs by 0.005 macro mAP. Read down the `loss_pkd` column: 23.99,
28.98, then 15.53. The expansion is worth nothing once `tau` is handed to the encoder as an
input channel, because then there is nothing left for it to reconstruct that it was not
already given.

This is a direct reading on `results/decoder_capacity/README.md`, which asks why the torque
loss channel buys ten times less BABEL mAP than velocity and acceleration do. It is *not*
inert. It nearly doubles the number of directions the encoder uses. It only looks inert
through a window-level action classifier, which does not use them -- which is the same
thing the CKA number there was pointing at (0.934 cross-cell against 0.959 cross-seed:
"the torque target does change the representation, just not in a direction a mean-pooled
linear probe reads"). Here that sentence has a magnitude attached.

**Geometry does not stand in for the benchmark.** Participation ratio against BABEL-60
macro mAP over the nine cells is r = +0.571, Spearman rho = +0.667. With n = 9 that is not
significant -- rho needs about 0.683 at p = 0.05 -- so it is a trend and is quoted as one.
Which is the useful outcome: if it tracked mAP it would be a cheaper copy of a table that
already exists, and instead it measures something that table is blind to.

**CKA**: 0.8636 +- 0.0498 across cells (n = 135) against 0.9357 +- 0.0223 across seeds of
one cell (n = 30). Cells are further apart than replicates, so the matrix axes are doing
something, and the floor is measured rather than assumed this time.

## Label efficiency (phase 2)

The convex probe refit at 5 / 10 / 25 / 50 / 100% of the labelled training set, scored
every time against the full validation split. A representation is supposed to be worth
something because it needs *fewer* labels, and the 100% number cannot express that at all.
The floor is the moments baseline run through the identical head and the identical
fractions: a cell whose curve is merely the moments curve shifted upward separates better
without being more label-efficient, and that is a different claim from the one pretraining
is usually sold on.

**Every fraction is scored on the same label set**, and this is the correctness property
the whole table rests on. `WindowMeanAveragePrecision` averages over the labels that have
positive support and silently drops the rest, so a 5% subsample -- which loses exactly the
rare labels, which are also the hard ones -- would otherwise report a macro average over a
*different vocabulary* than the 100% point. That is five metrics plotted as one curve.
`scripts/label_efficiency.py` therefore fixes the vocabulary once, to the labels carrying a
positive at the smallest fraction (intersected with validation support so the printed count
is the count actually averaged), and scores all five points through that one mask. The
surviving label count is in the table caption, and the label *indices* are written into
every `metrics.json` so two cells can be checked for having scored the same 18 labels
rather than trusted to have.

**Subsampled by `sample_id`, never by window.** Windows are crops of samples and the probe
extracts 20 passes, so 5% of *windows* would still touch nearly every sample and its
neighbours' crops: that is not 5% of the labelled data, it is the whole label set with less
augmentation. The draws are nested prefixes of one shuffle, so the curve is about adding
labels to a fixed starting set rather than five unrelated draws. The validation split is
never subsampled.

**One extraction, five fits, and a fresh weight-decay sweep inside each.** The backbone
forward is the entire bill and does not depend on which rows are kept, exactly as
`MotionConvexClassifier.sweep` extracts once and refits per candidate. The sweep is redone
per fraction because the L-BFGS objective is `mean BCE + wd * ||W||^2` and the mean is over
rows: at 5% the same `wd` shrinks against a twentieth of the data, so reusing the 100%
optimum would report a regularization artefact as a data-efficiency result. The head is fit
over the full 60-label vocabulary and only *scored* through the mask, since the fit is
separable per label and slicing first would change the loss-to-penalty ratio instead.

## kNN retrieval (phase 2)

The same features with no head at all: a validation window takes the softmax-temperature
weighted labels of its nearest training windows under cosine similarity, swept over
k = 1, 5, 10, 20, 50. It asks whether same-action windows land near each other under *any*
geometry, which a linear probe cannot see -- a backbone that ranks well here and badly
under a linear head is saying something specific, and it costs nearly nothing to ask.

`scripts/knn_probe.py` decided all of this already and is used unchanged;
`scripts/knn_matrix.sh` is only the matrix loop it never had. Two pieces of glue live in
that loop rather than in the probe: the checkpoint is resolved from a run directory with
the probe's own `latest_checkpoint`, and the corpus normalization is patched into a copy of
the config per cell, because `knn_probe.py` takes no dotlist overrides and a frozen backbone
fed windows normalized by the wrong statistics fails silently rather than loudly. The
moments floor is measured once per matrix, since it reads no checkpoint.

The reported row is the best k by macro mAP with the whole sweep kept beside it in
`metrics.json`; the distances are computed once, so reporting only the argmax would throw
away the only evidence that k mattered.

## Running it

```
SEEDS="1 2 3" CKA=1 ./scripts/geometry_matrix.sh
SEEDS="1 2 3" ./scripts/knn_matrix.sh
SEEDS="1 2 3" ./scripts/label_efficiency_matrix.sh

uv run python results/representation/gen.py
```

Each runner carries its own moments floor and skips any cell that already has a
`metrics.json`, so a re-run costs nothing. Three CSVs rather than one: `metrics.csv`
(geometry, one row per cell), `label_efficiency.csv` (one row per cell and fraction) and
`knn.csv` (one row per cell and k). They are not one table because they do not have one
row.

<!-- GENERATED by gen.py -- do not edit below this line -->

*27 cells, 4936 validation windows, d = 256 (`mean` pooling). Regenerate with `gen.py`.*

## Participation ratio

**amass_clean / medium_100ep** -- rows are the encoder's input channels, columns its loss channels.

| input \ loss | `p` | `pk` | `pkd` |
| :--- | ---: | ---: | ---: |
| `p` | 16.36 ±1.60 | 19.00 ±0.86 | 23.99 ±3.92 |
| `pk` | 16.36 ±0.33 | 17.04 ±1.22 | 28.98 ±2.18 |
| `pkd` | 15.84 ±0.87 | 17.37 ±0.21 | 15.53 ±0.63 |

*± is the sd over pretraining replicates (3 per cell). Pooled over cells it is 1.31, so a difference below ~2.6 is inside seed noise. Dimensionality and variance are not interchangeable: read this beside `total_variance` in `metrics.csv`, which moves the other way.*

## Linear CKA

**amass_clean/medium_100ep**

| pairs | mean | sd | n |
| :--- | ---: | ---: | ---: |
| different cell, same seed | 0.8636 | 0.0498 | 135 |
| same cell, different seed | 0.9357 | 0.0223 | 30 |

## Label efficiency

**amass_clean / medium_100ep** -- macro mAP at each fraction of the labelled training set, every point scored on the same 60 of 60 labels (those with support at 5%). Mean over 1 pretraining seed(s).

| cell | 5% | 10% | 25% | 50% | 100% | 5% / 100% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_pkd__loss_p` | 0.2244 | 0.2690 | 0.2978 | 0.3285 | 0.3483 | 0.64 |

## kNN retrieval

**amass_clean / medium_100ep** -- macro mAP, no head fit, neighbour count across the columns. Mean over 3 pretraining seed(s).

| cell | k=1 | k=5 | k=10 | k=20 | k=50 | k=100 | k=200 | k=500 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_p__loss_p` | 0.1544 | 0.2847 | 0.3135 | 0.3295 | 0.3398 | 0.3431 | 0.3444 | 0.3452 |
| `in_p__loss_pk` | 0.1581 | 0.2989 | 0.3269 | 0.3435 | 0.3522 | 0.3555 | 0.3564 | 0.3569 |
| `in_p__loss_pkd` | 0.1589 | 0.2956 | 0.3229 | 0.3378 | 0.3475 | 0.3507 | 0.3520 | 0.3526 |
| `in_pk__loss_p` | 0.1604 | 0.2961 | 0.3261 | 0.3414 | 0.3510 | 0.3536 | 0.3549 | 0.3556 |
| `in_pk__loss_pk` | 0.1585 | 0.2971 | 0.3246 | 0.3405 | 0.3507 | 0.3537 | 0.3544 | 0.3548 |
| `in_pk__loss_pkd` | 0.1592 | 0.3000 | 0.3264 | 0.3420 | 0.3508 | 0.3535 | 0.3544 | 0.3550 |
| `in_pkd__loss_p` | 0.1558 | 0.2921 | 0.3222 | 0.3385 | 0.3493 | 0.3527 | 0.3537 | 0.3545 |
| `in_pkd__loss_pk` | 0.1567 | 0.2969 | 0.3275 | 0.3431 | 0.3522 | 0.3550 | 0.3558 | 0.3561 |
| `in_pkd__loss_pkd` | 0.1577 | 0.2967 | 0.3241 | 0.3385 | 0.3481 | 0.3514 | 0.3523 | 0.3531 |
| `moments` (floor) | 0.1539 | 0.2843 | 0.3101 | 0.3251 | 0.3310 | 0.3315 | 0.3298 | 0.3278 |

<!-- /GENERATED -->
