# sometria

Self-supervised pretraining on biomechanical DOF sequences, evaluated by frozen probes.

Three pretext objectives share one encoder, one dataloader and one optimizer, so a
difference between them is attributable to the objective:

| `model.name` | Predicts | Cells per sweep |
| :--- | :--- | ---: |
| `mae` | the raw values of the masked patches, through a transformer decoder | 9 |
| `simmim` | the same values, but mask tokens enter the *encoder* and one linear layer reads them out | 9 |
| `jepa` | an EMA teacher's embedding at the masked positions | 3 |

JEPA reconstructs nothing, so it has no `channels_output` to vary: its sweep is the input
axis alone, three cells rather than nine. The scripts detect that from the base config.

## Pretraining

Every run is a `(corpus, architecture, epoch budget, objective)` point, swept over input
channels and, where the objective has them, loss channels.

```bash
# MAE -- the baseline. 9 cells per seed.
CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS=1 ./scripts/pretrain_matrix.sh

# JEPA -- 3 cells, no loss axis.
BASE=config/pretrain_jepa.yaml CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS=1 \
  ./scripts/pretrain_matrix.sh

# SimMIM -- 9 cells.
BASE=config/pretrain_simmim.yaml CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS=1 \
  ./scripts/pretrain_matrix.sh
```

Pass `SEEDS` even for a first run. Dropping it is legal and writes the cells straight into
the architecture directory with no seed segment, but every run currently on disk is
seeded, the report's replicate-spread and rankability tables need more than one run per
cell to say anything, and a lone unseeded run sits in a directory the seeded sweeps never
look at.

`BASE` is the only thing that changes. It sets the objective, and everything else follows:
the size overlay resolves to `config/<objective>/<arch>.yaml` (falling back to
`config/mae/<arch>.yaml` where an objective has no file of its own), and the output path
gains an objective suffix for everything except MAE, which keeps its unsuffixed
directories so runs already on disk stay where they are.

```
runs/pretrain/amass_clean/medium_100ep/in_<x>__loss_<y>          MAE
runs/pretrain/amass_clean/medium_100ep_jepa/in_<x>               JEPA
runs/pretrain/amass_clean/medium_100ep_simmim/in_<x>__loss_<y>   SimMIM
```

Knobs, all environment variables:

| | |
| :--- | :--- |
| `CORPUS` | names both `config/dataloader/<corpus>.yaml` and the path segment |
| `ARCH` | names both `config/<objective>/<arch>.yaml` and the path segment |
| `SEEDS="1 2 3"` | replicates, each nested under `seed<n>/`. Empty means the canonical run |
| `CELLS="in_pk__loss_pkd"` | run only these cells |
| `DRY=1` | print the resolved commands and change nothing |

A single cell, without the sweep:

```bash
uv run python train.py \
  --config config/pretrain_jepa.yaml config/dataloader/amass_clean.yaml config/jepa/medium.yaml \
  --output runs/pretrain/amass_clean/medium_100ep_jepa/in_pkd \
  training.epochs=100 model.channels_input=[0,1,2,3,4]
```

Dotlist overrides merge last and beat every config file, so `model.mask_ratio=0.7` on the
end of that command does what it says. Put anything you intend to run more than once in a
config file instead, so the run records it.

## Probing

Reads a pretraining tree, writes one `metrics.json` per cell. `OBJECTIVE` picks the tree;
it is named here rather than sniffed, because this script never sees a pretraining config.

`SEEDS` must name the pretraining replicates to score, and must match what was pretrained.
It is not optional in practice: without it the script looks for cells directly under the
architecture directory, finds none, and exits.

```bash
CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS="1 2 3 42" ./scripts/probe_matrix.sh
OBJECTIVE=jepa   CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS=42 ./scripts/probe_matrix.sh
OBJECTIVE=simmim CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS=42 ./scripts/probe_matrix.sh
```

Check what exists before scoring it, since the seeds differ per objective:

```bash
ls runs/pretrain/amass_clean/medium_100ep_simmim    # -> seed42
```

The backbone is frozen, and normalization statistics are read from the *corpus* config
rather than the probe config. Feeding a frozen encoder windows normalized by statistics
other than the ones it pretrained under is not an error anywhere; it silently offsets
every feature. That is why `CORPUS` is a knob here and not just a path segment.

`SEEDS` mirrors into the output path, so a probe is always filed under the backbone it
scored. `REPEATS` re-probes one backbone, which bounds the probe's own non-determinism;
the training split draws a random crop per sample per pass.

The other benchmark needs its own probe config:

```bash
BENCHMARK=carepd_updrs_convex CONFIG=config/experiment_probe_carepd.yaml \
  SPLIT_SET=carepd_lodo_BMCLab ./scripts/probe_matrix.sh
```

That scores one split set. CARE-PD's published protocols are 126 of them, and the frozen
backbone means all 126 read the same features, so `goodnight_carepd.sh` extracts once per
cell and reduces each fold to an L-BFGS solve. It takes the same `OBJECTIVE` knob:

```bash
SEEDS="42 1 2" bash goodnight_carepd.sh
OBJECTIVE=simmim SEEDS="42 1 2" bash goodnight_carepd.sh
OBJECTIVE=jepa   SEEDS="42 1 2" bash goodnight_carepd.sh
DRY=1 bash goodnight_carepd.sh    # print the plan, change nothing
```

The SMPL arm cannot be scored here: CARE-PD is imported in the OpenSim representation
only, and an `amass_smpl` backbone expects 21 DOF x 18 features against `config/smpl.yaml`.

A single checkpoint, without the sweep. Pass a run directory and it takes the monitored
best checkpoint; pass a `.ckpt` path and it takes that one. The objective is resolved from
the weights, so nothing needs to be told which model wrote the file:

```bash
uv run python scripts/probe_convex_mae.py \
  --checkpoint runs/pretrain/amass_clean/medium_100ep_simmim/in_p__loss_p \
  --config config/experiment_linear_probe.yaml \
  --output runs/probe/babel_60_convex/amass_clean/medium_100ep_simmim/in_p__loss_p \
  dataloader.normalization=stats/opensim_sincos_log_vel_acc_tau_v2/pretrain_v1_stats_train_clean.pt
```

## Reading the results

```bash
uv run python results/probe/babel_60_convex/gen.py
```

Walks `runs/probe/<benchmark>/`, writes `metrics.csv`, and replaces the generated block of
`README.md` in that directory. Hand-written prose outside the markers survives. Numbers are
read from each run's own `config.yaml`, never transcribed.

Two cautions that are properties of the objectives, not of the code.

**JEPA's `val/loss` is not a quality signal.** A student and teacher that agree on a
constant drive it to zero having learned nothing, so `config/pretrain_jepa.yaml` selects
checkpoints on `val/loss_over_null` instead. 1.0 means the predictor is doing no better
than emitting the batch mean; below 1 is real prediction. Watch it, not the loss.

**SimMIM's frozen features are near rank-1 here.** Its own paper reports linear probing at
56.7% against 83.8% fine-tuned and declines to be judged on the former. Measured on
BABEL-60, its mean-pooled features use 1.1 to 2.5 of 256 effective dimensions against MAE's
11.6 to 17.6, and its probe score correlates with that rank at r = 0.93 against MAE's 0.32.
Its *input*-channel ordering is usable and does replicate MAE's; its *loss*-channel ordering
is measuring escape from collapse and should not be read as a channel effect.
