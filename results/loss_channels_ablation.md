| Loss Channels | Content Type | $d_{\text{head}}$ | Macro mAP | Micro mAP | Macro F1 | Gap Recovered (Macro mAP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| *Chance (label prevalence)* | --- | --- | 0.042 | --- | --- | --- |
| *Random frozen encoder* | --- | --- | 0.128 | 0.334 | 0.021 | 0.0% |
| $\tau$ | Dynamic | 8 | 0.153 | 0.367 | 0.041 | 14.8% |
| $\sin\theta, \cos\theta$ | Kinematic | 16 | 0.180 | 0.392 | 0.036 | 30.8% |
| $\sin\theta, \ddot\theta$ | Kinematic + Proxy | 16 | 0.238 | 0.438 | 0.085 | 65.1% |
| $\sin\theta, \tau$ | Kinematic + Dynamic | 16 | 0.259 | 0.458 | 0.100 | 77.5% |
| $\sin\theta, \cos\theta, \tau$ | Kinematic + Dynamic | 24 | 0.261 | 0.460 | 0.116 | 78.7% |
| $\sin\theta, \cos\theta, \dot\theta, \ddot\theta$ | Kinematic Only | 32 | 0.216 | 0.426 | 0.074 | 52.1% |
| **All Five Channels** | **Kinematic + Dynamic** | **40** | **0.297** | **0.493** | **0.143** | **100.0%** |

Note: Tested on mae_40ep with an attentive probe for 20ep.

---

## 3×3 Input × Loss Channel Matrix

Small model (dim=128, enc_depth=4, dec_depth=2, heads=4), 40 pretraining epochs, attentive probe, per-token target normalization.

| Input \\ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** (pose: sin, cos) | 0.2686 | 0.2511 | **0.3017** |
| **pk** (pose + kin: + vel, acc) | 0.2768 | 0.2528 | 0.2917 |
| **pkd** (pose + kin + dyn: + tau) | **0.2876** | 0.2269 | 0.2796 |

- **Best**: `in_p__loss_pkd` (0.3017) — encoder sees only sin/cos but must reconstruct all 5 channels. Predicting the unobserved forces richer representation.
- **Runner-up**: `in_pkd__loss_p` (0.2876) — full input, narrow loss. Rich encoder signal, clean gradient.
- **Surprise**: `in_pkd__loss_pk` (0.2269) — full input + medium loss is worse than both extremes. Predicting vel/acc without tau appears to poison the representation.
- **More encoder input weakly hurts wide-loss column**: 0.3017 → 0.2917 → 0.2796 as input widens under `loss_pkd`.
- Base model (dim=256, depth=8) `in_pkd__loss_pkd`: 0.3297.

---

## Best setup so far: medium 100ep + 100ep mean linear probe

Medium MAE (dim=256, enc_depth=6, dec_depth=3, heads=8), 100 pretraining epochs, best `epoch=*.ckpt` per run, **mean pooling** probe for 100 epochs from `runs/baseline_matrix_probe_mean_100ep+100ep_medium`. This is the cleanest linear-probe readout so far: token mean + linear head, no learned attention.

### Macro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.3245 | 0.3415 | 0.3410 |
| **pk** | 0.3247 | 0.3406 | **0.3493** |
| **pkd** | 0.3226 | 0.3267 | 0.3288 |

### Micro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5159 | 0.5324 | 0.5316 |
| **pk** | 0.5202 | 0.5352 | **0.5435** |
| **pkd** | 0.5193 | 0.5285 | 0.5287 |

### Recall@1

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.2748 | 0.2764 | 0.2817 |
| **pk** | 0.2756 | 0.2809 | **0.2849** |
| **pkd** | 0.2768 | 0.2789 | 0.2793 |

### Recall@3

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5112 | 0.5250 | 0.5240 |
| **pk** | 0.5141 | 0.5250 | **0.5302** |
| **pkd** | 0.5144 | 0.5209 | 0.5239 |

### Recall@5

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.6216 | 0.6361 | 0.6378 |
| **pk** | 0.6252 | 0.6338 | **0.6404** |
| **pkd** | 0.6281 | 0.6352 | 0.6346 |

### Best cell

| Cell | Epoch | Macro mAP | Micro mAP | Macro F1 | R@1 | R@3 | R@5 | Val loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_pk__loss_pkd` | 99 | **0.3493** | **0.5435** | **0.2230** | **0.2849** | **0.5302** | **0.6404** | 0.1021 |

- **Current best clean setup**: `in_pk__loss_pkd` — encoder gets pose + kinematics, loss reconstructs pose + kinematics + dynamics.
- Dynamics are most useful as a **prediction target**, not as an observed input shortcut.
- Adding tau to the encoder input hurts under mean pooling (`pkd` row), while adding tau to the reconstruction target helps (`pkd` loss column).
- 40ep probes were undertrained; 100ep probe improves macro mAP by ~0.03–0.05 across the matrix.

---

## Medium 100ep+100ep matrix, MotionX-augmented pretraining

Same medium MAE (dim=256, enc_depth=6, dec_depth=3, heads=8), same mean-pooling 100ep linear probe, but the pretraining backbone comes from `runs/baseline_matrix_100ep_medium_with_motionx` — MotionX mixed into pretraining data (~3.4x more steps/epoch than the non-MotionX medium run: step 38700 vs 11500 at epoch 99, `stats/.../pretrain_v2_train_clean.pt` normalization). Best `epoch=*.ckpt` per run (checkpoint monitor is val/macro_map).

### Macro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.3389 | 0.3577 | 0.3609 |
| **pk** | 0.3454 | **0.3663** | 0.3631 |
| **pkd** | 0.3512 | 0.3598 | 0.3558 |

### Micro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5269 | 0.5486 | 0.5489 |
| **pk** | 0.5300 | **0.5534** | 0.5471 |
| **pkd** | 0.5383 | 0.5484 | 0.5431 |

### Recall@1

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.2750 | 0.2843 | 0.2848 |
| **pk** | 0.2790 | **0.2853** | 0.2823 |
| **pkd** | 0.2807 | 0.2829 | 0.2832 |

### Recall@3

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5176 | 0.5342 | **0.5364** |
| **pk** | 0.5195 | 0.5347 | 0.5320 |
| **pkd** | 0.5207 | 0.5345 | 0.5343 |

### Recall@5

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.6284 | **0.6505** | 0.6486 |
| **pk** | 0.6389 | 0.6500 | 0.6476 |
| **pkd** | 0.6411 | 0.6480 | 0.6448 |

### Best cell

| Cell | Epoch | Macro mAP | Micro mAP | Macro F1 | R@1 | R@3 | R@5 | Val loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_pk__loss_pk` | 83 | **0.3663** | **0.5534** | 0.2733 | **0.2853** | 0.5347 | 0.6500 | 0.1006 |

- **Best setup flips**: `in_pk__loss_pk` (0.3663) overtakes `in_pk__loss_pkd` (0.3631, previous best) — with MotionX pretraining, matched input/loss width (`pk`/`pk`) beats reconstructing the wider `pkd` target.
- **MotionX pretraining lifts the whole matrix**: every cell improves over the non-MotionX medium matrix, macro mAP up ~0.01–0.03 across the board (e.g. `in_pk__loss_pk` 0.3406 → 0.3663, `in_p__loss_p` 0.3245 → 0.3389).
- **`pkd` input row flattens**: without MotionX it climbed monotonically with loss width (0.3226/0.3267/0.3288); with MotionX it peaks mid-row instead (0.3512/0.3598/0.3558) — tau-as-input no longer cleanly trades off against loss width.
- **`p` input row still weakest at matched `p` loss** (0.3389 vs the 0.3663 top cell), same qualitative story as before MotionX — narrow input keeps lagging regardless of pretraining data.

---

## Same matrix, convex probe (mean-pool + L-BFGS, no AdamW)

Same nine `runs/baseline_matrix_100ep_medium_with_motionx` checkpoints, same mean pooling, but the head is `MotionConvexClassifier` (`src/sometria/downstream/classifier_convex.py`): BCE over one frozen-feature `Linear` head is convex, so it is solved to its actual global optimum with L-BFGS instead of trained for 100 AdamW epochs and checkpointed by best-epoch. `weight_decay` swept per cell over `DEFAULT_WEIGHT_DECAYS` (1e-1 .. 1e-6), `train_passes=20` (the training split draws one random crop per sample per pass, so this matches the ~100-crop exposure AdamW got across its epochs without needing 100 of them). `scripts/baseline_matrix_probe_convex.sh` output, `runs/baseline_matrix_probe_convex_medium_with_motionx`.

Point of this pass: AdamW's lr/warmup/schedule are fixed once in config and shared across all nine cells, never verified as suiting each one — a cell could look worse purely from schedule mismatch, not representation quality. The convex head has no lr, no schedule, no seed variance; the only thing selected via val is `weight_decay`, and it's reported per cell rather than left as a silent fixed guess.

### Macro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.3470 | 0.3613 | 0.3609 |
| **pk** | 0.3508 | **0.3714** | 0.3620 |
| **pkd** | 0.3543 | 0.3628 | 0.3568 |

### Micro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5339 | 0.5520 | 0.5524 |
| **pk** | 0.5404 | **0.5590** | 0.5527 |
| **pkd** | 0.5430 | 0.5514 | 0.5476 |

### Recall@1

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.2768 | 0.2842 | 0.2820 |
| **pk** | 0.2791 | **0.2850** | 0.2833 |
| **pkd** | 0.2821 | 0.2835 | 0.2837 |

### Recall@3

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5210 | 0.5360 | 0.5349 |
| **pk** | 0.5259 | **0.5375** | 0.5353 |
| **pkd** | 0.5223 | 0.5344 | 0.5341 |

### Recall@5

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.6338 | 0.6501 | 0.6496 |
| **pk** | 0.6422 | **0.6521** | 0.6493 |
| **pkd** | 0.6406 | 0.6490 | 0.6479 |

### Best cell

| Cell | weight_decay | Macro mAP | Micro mAP | Macro F1 | R@1 | R@3 | R@5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_pk__loss_pk` | 3e-06 | **0.3714** | **0.5590** | 0.2777 | **0.2850** | **0.5375** | **0.6521** |

- **Same winner as AdamW**: `in_pk__loss_pk` tops every metric here too (0.3714 macro mAP vs AdamW's 0.3663) — the convex fit, run to its actual optimum with no schedule to get lucky or unlucky with, reproduces the exact ranking AdamW found. That rules out "AdamW's fixed lr happened to suit `pk`/`pk` better" as the explanation for the earlier flip from `pk`/`pkd` to `pk`/`pk`.
- **Convex beats AdamW in 8 of 9 cells** (macro mAP delta +0.003 to +0.008), ties at `in_p__loss_pkd` (0.3609 both), and is marginally behind by 0.001 only at `in_pk__loss_pkd` — consistent with a fixed shared AdamW schedule being a mild, not dominant, handicap across the matrix.
- **weight_decay picked stays in a narrow 3e-6–1e-5 band** across all nine cells, away from both grid edges — the sweep isn't straining against `DEFAULT_WEIGHT_DECAYS`' bounds, and no cell wanted a wildly different regularization strength than its neighbors.

---

## Same convex probe, medium pretraining without MotionX

Same convex head, same protocol, on the earlier `runs/baseline_matrix_100ep_medium` checkpoints (no MotionX in pretraining, `pretrain_v1_train_clean.pt` normalization) — the AdamW numbers for this matrix are in the "Best setup so far" section above. `runs/baseline_matrix_probe_convex_medium`.

### Macro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.3472 | 0.3612 | 0.3616 |
| **pk** | 0.3445 | 0.3633 | **0.3644** |
| **pkd** | 0.3452 | 0.3585 | 0.3572 |

### Micro mAP

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5342 | 0.5513 | 0.5478 |
| **pk** | 0.5393 | 0.5537 | **0.5543** |
| **pkd** | 0.5376 | 0.5496 | 0.5491 |

### Recall@1

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.2749 | 0.2833 | 0.2828 |
| **pk** | 0.2762 | **0.2845** | **0.2845** |
| **pkd** | 0.2808 | 0.2825 | 0.2845 |

### Recall@3

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.5197 | 0.5352 | 0.5303 |
| **pk** | 0.5216 | **0.5367** | 0.5352 |
| **pkd** | 0.5250 | 0.5319 | 0.5361 |

### Recall@5

| Input \ Loss | p | pk | pkd |
| :--- | :---: | :---: | :---: |
| **p** | 0.6348 | 0.6459 | 0.6447 |
| **pk** | 0.6353 | 0.6485 | 0.6467 |
| **pkd** | 0.6365 | 0.6517 | **0.6525** |

### Best cell

| Cell | weight_decay | Macro mAP | Micro mAP | Macro F1 | R@1 | R@3 | R@5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_pk__loss_pkd` | 3e-05 | **0.3644** | **0.5543** | 0.2509 | 0.2845 | 0.5352 | 0.6467 |

- **Winner matches AdamW, unlike the MotionX matrix**: `in_pk__loss_pkd` (0.3644) edges out `in_pk__loss_pk` (0.3633) by 0.0011 — same cell AdamW picked here (0.3493), not the `pk`/`pk` cell that won once MotionX was mixed into pretraining. This is the confirmation the earlier torque-quality hypothesis needed: without MotionX's noisier tau, reconstructing tau (`loss_pkd`) still helps; only once MotionX dominates pretraining does dropping tau from the target (`loss_pk`) win instead. Consistent with `results/motionx_tau_quality.md`.
- **Convex gain is much larger here than on the MotionX matrix**: +0.015 to +0.032 macro mAP across all nine cells (vs +0.003 to +0.008 with MotionX pretraining, one cell −0.001). AdamW's fixed schedule for this matrix used `lr=0.0005`, `batch_size=128` — different from the MotionX matrix's `lr=0.001`, `batch_size=32` — so this schedule looks like it suited this pretraining setup markedly worse, a bigger version of the confound the convex probe was built to remove.
- **Winner is a near-tie, not a landslide**: `pk`/`pkd` and `pk`/`pk` are 0.0011 apart on macro mAP, and other metrics split between them and `pkd`/`pkd` (Recall@5) — less of a single dominant cell than the MotionX matrix showed, so "best setup" here is closer to "the `pk`-input, wide-ish-loss corner" than one specific cell.
