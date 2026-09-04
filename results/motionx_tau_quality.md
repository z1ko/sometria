# MotionX torque quality check

## Finding

MotionX is not uniformly broken. The current `broken` flag is an AMASS-calibrated torque-rate heuristic:

```python
broken = nonfinite or max(abs(diff(tau))) * hz > 3e5
```

On `pretrain_v2`, this marks many MotionX samples because MotionX torque scale/noise differs strongly by subset.

## Current data

| subset | samples | provenance | broken % |
|---|---:|---|---:|
| aist | 1,467 | multi-view video, re-annotated by Motion-X | 99.7 |
| game_motion | 10,208 | online/game motion videos | 87.2 |
| HAA500 | 5,228 | action-recognition videos, re-annotated | 86.2 |
| animation | 329 | online/animation videos | 72.9 |
| dance | 162 | online videos | 50.0 |
| fitness | 16,686 | online videos | 46.3 |
| humman | 744 | multimodal/action dataset, re-annotated | 45.6 |
| kungfu | 1,028 | online videos | 42.3 |
| perform | 474 | online videos | 32.7 |
| idea400 | 12,424 | Motion-X self-recorded subset | 9.9 |
| music | 3,541 | online videos | 9.0 |

MotionX kept-DOF torque magnitudes are also larger than AMASS:

| metric (sampled kept DOFs) | AMASS | MotionX |
|---|---:|---:|
| median kept `tau_absmax` | ~947 | ~1,816 |
| median kept `tau_absmax / (mass*g)` | ~1.46 | ~2.85 |
| median kept `tau_absmax / (mass*g*height)` | ~0.84 | ~1.66 |

## Interpretation

Likely source/pipeline artifact, not proof that all MotionX motion is unusable. Motion-X mixes mocap-like sources, multi-view video, single-view action videos, self-recorded video, and online videos. Many subsets are markerless/pseudo-SMPL-X annotations; inverse dynamics on those can amplify jitter into large `tau` spikes.

## Recommendation

Keep tau, but do not treat current `broken` as universal corruption for MotionX.

Minimal next step:

1. Scale torque before encoding: `tau /= mass * g * height` for rotational DOFs.
2. Measure quality on kept DOFs only, not excluded pelvis/root DOFs.
3. Report/train with subset stratification; consider excluding worst subsets first (`aist`, `game_motion`, `HAA500`) if tau-target runs degrade.

Sources: Motion-X paper/data provenance: https://arxiv.org/html/2307.00818v2 and https://motion-x-dataset.github.io/.
