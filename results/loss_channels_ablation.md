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
