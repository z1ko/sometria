#!/usr/bin/env bash
# Overnight: every downstream task, on the SMPL arm's seven pretrained backbones.
#
#   bash goodnight_smpl.sh              # all stages
#   DRY=1 bash goodnight_smpl.sh        # print the plan and exit
#   STAGES="1 2" bash goodnight_smpl.sh # a subset, by the numbers below
#
# The point is a like-for-like comparison of representations: the OpenSim arm encodes 43
# anatomical DOFs x 5 channels, the SMPL arm 21 body joints x 18 channels of rot6d, and
# every task below is otherwise identical -- same label set, same head, same metrics.
#
# Why this is not a matrix script. SMPL is one configuration, not a sweep: its features are
# rot6d with no torque, so the p/pv/pk/pkd ladder -- indices into the 5-channel OpenSim
# stack -- does not describe it. Consequently its pretraining tree has no cell level at
# all; the checkpoint sits at `<arch>/seed<N>/` where the OpenSim tree has
# `<arch>/seed<N>/in_pk__loss_pv/`. Every scripts/*_matrix.sh globs that missing level, so
# none of them can drive this arm and each task is invoked directly below, exactly as the
# amass_smpl stages of the old goodnight.sh did. The cell name is applied to the *output*
# path only, so gen.py files these rows beside the OpenSim ones: `in_p` for an objective
# with no loss axis, `in_p__loss_p` for one with it.
#
# Seven pose-only backbones -- MAE seeds 1/2/42, JEPA seeds 1/42, SimMIM seeds 1/42 -- plus
# three from stage 0 below, for ten.
#
#   0. pretrain MAE `in_p__loss_pv`        3 runs, ~1.8h -- see below
#   1. BABEL-60 probe, official protocol   7 runs
#   2. segmentation probe                  7 runs
#   3. representation: geometry            7 runs
#   4. representation: kNN                 7 runs
#   5. representation: label efficiency    7 runs
#   6. reconstruction, both masks         14 runs -- NOT in the default STAGES, see below
#   7. fine-tune                           7 runs (3 already on disk, they skip)
#
# Stage 0 is the experiment that decides what the rest of this comparison means. Every
# winning OpenSim cell reconstructs velocity (`loss_pv`: sin, cos in; sin, cos, vel out),
# and the pose-only SMPL arm above was never offered that target. SMPL's stack already
# carries `vel0-5` in slots 6-11 as `np.gradient(rot6d)` -- the same construction as the
# OpenSim channel -- so the matched cell is rot6d in, rot6d + velocity out. It lands as a
# cell *inside* the existing seed directory, `seed<N>/in_p__loss_pv/`, giving this arm the
# cell level the OpenSim tree has without touching the pose-only checkpoint beside it:
# latest_checkpoint() reads only `<run>/lightning_logs/`, so a sibling subdirectory is
# invisible to it. Downstream, those rows land in the `pv` column of each page's SMPL
# matrix, where gen.py's delta table already computes SMPL - OpenSim at `in_p__loss_pv`.
#
# One asymmetry: OpenSim velocity is 1 channel per DOF (43), SMPL's is 6 per joint (126),
# so this decoder reconstructs a much larger velocity target than its OpenSim counterpart.
#
# Stage 6 is out of the default because scripts/probe_reconstruction.py cannot score this
# corpus. Its metric is `atan2(sin, cos)`: normalization leaves the two pose channels
# untouched, so that expression on the model's own input *is* the joint angle in radians,
# which is what makes "degrees" mean anything (sometria/downstream/reconstruction.py). SMPL
# is rot6d -- six channels holding the first two columns of a rotation matrix -- and has no
# such pair, so the call raises `KeyError: not feature channels: ['sin', 'cos']`.
#
# Supporting it is not a channel-name swap. It needs rot6d -> rotation matrix -> geodesic
# angle, and the result is a different quantity from the OpenSim arm's: 3-DOF joint
# rotation error over 21 joints against per-DOF angle error over 43 anatomical DOFs. The
# two "degrees" columns would not be comparable even once both exist. Run `STAGES=6` if
# that metric ever lands.
#
# The tiled convex probe is not here: all seven are already scored in
# results/probe/babel_60_convex/.
#
# KNOWN AND ACCEPTED: this arm scores `babel_paired` -- BABEL's split restricted to takes
# both imports produced -- while the OpenSim pages score `babel_official`. Numbers from
# here are therefore NOT comparable cell-for-cell with results/probe/babel_60_official/ or
# results/probe/babel_60_segmentation/. They are comparable across SMPL objectives, and to
# an OpenSim run on `babel_paired` that does not exist yet. Running that second sweep is
# what would make the representation comparison airtight; this script deliberately does
# not, because the ask was the SMPL side alone.
#
# No CKA stage. CKA is an all-pairs comparison across the cells of a matrix, and this arm
# has exactly one cell per architecture, so there is no pair to take.
#
# Every step skips work already on disk (`metrics.json`) and refuses a backbone with no
# `done` marker, so a crash is resumed by re-running this file. Nothing here deletes.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_smpl_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
# Everything below lands in the log as well as the terminal: an unattended job that fails
# at 3am is only debuggable if its output outlived the session that started it.
exec > >(tee -a "$LOG") 2>&1

PYTHON=(${PYTHON:-uv run python})

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

run() {
    if [ -n "${DRY:-}" ]; then printf '  %q' "$@"; printf '\n'
    else "$@"; fi
}

# 6 omitted on purpose -- see the header. Pass it explicitly to attempt it anyway.
STAGES=${STAGES:-"0 1 2 3 4 5 7"}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

CORPUS=amass_smpl
PROBE_CONFIG=${PROBE_CONFIG:-config/experiment_probe_babel_official_smpl.yaml}
SEG_CONFIG=${SEG_CONFIG:-config/experiment_segmentation_smpl.yaml}
REP_CONFIG=${REP_CONFIG:-config/experiment_linear_probe_smpl.yaml}
FT_CONFIG=${FT_CONFIG:-config/experiment_finetune_smpl.yaml}
FT_EXTRA=${FT_EXTRA:-"model.layer_decay=0.5 training.patience=20"}
DECODER_EPOCHS=${DECODER_EPOCHS:-40}
MASKS=${MASKS:-"random suffix"}

# Stage 0. Seeds match the pose-only MAE arm so the two pair up replicate for replicate.
PV_SEEDS=${PV_SEEDS:-"1 2 42"}
PV_INPUT="[0,1,2,3,4,5]"                   # rot6d
PV_OUTPUT="[0,1,2,3,4,5,6,7,8,9,10,11]"    # rot6d + vel

# One row per pretrained backbone:
#   <arch> <seed> <output cell name> <checkpoint submodule> <checkpoint subdir, or ->
# The submodule is the objective's, not a preference: JEPA has no `backbone.` at all. The
# subdir is explicit rather than probed for, on purpose: a fallback of "use the cell
# directory if it exists, else the seed directory" would, for a stage-0 cell not yet
# trained, silently resolve to the *pose-only* checkpoint beside it -- which has a `done`
# marker -- and file its scores under `in_p__loss_pv`. Named, a missing cell simply fails
# the `done` check and is skipped loudly.
BACKBONES=${BACKBONES:-"
medium_100ep        seed1  in_p__loss_p  backbone  -
medium_100ep        seed2  in_p__loss_p  backbone  -
medium_100ep        seed42 in_p__loss_p  backbone  -
medium_100ep_jepa   seed1  in_p          teacher   -
medium_100ep_jepa   seed42 in_p          teacher   -
medium_100ep_simmim seed1  in_p__loss_p  backbone  -
medium_100ep_simmim seed42 in_p__loss_p  backbone  -
medium_100ep        seed1  in_p__loss_pv backbone  in_p__loss_pv
medium_100ep        seed2  in_p__loss_pv backbone  in_p__loss_pv
medium_100ep        seed42 in_p__loss_pv backbone  in_p__loss_pv
"}

# Iterate BACKBONES, calling `$1 <arch> <seed> <cell> <encoder> <checkpoint-dir>` for each
# one that finished pretraining. The `done` marker rather than the directory: a backbone
# still training would be scored at whatever epoch it has reached and then sit in the
# tables as a replicate.
for_each_backbone() {
    local task=$1 arch seed cell encoder subdir checkpoint
    while read -r arch seed cell encoder subdir; do
        [ -n "$arch" ] || continue
        checkpoint="runs/pretrain/$CORPUS/$arch/$seed"
        [ "$subdir" != "-" ] && checkpoint="$checkpoint/$subdir"
        if [ ! -e "$checkpoint/done" ]; then
            echo "UNFINISHED  $checkpoint -- no done marker, skipping"
            continue
        fi
        "$task" "$arch" "$seed" "$cell" "$encoder" "$checkpoint"
    done <<< "$(echo "$BACKBONES" | sed '/^[[:space:]]*$/d')"
}

skip_done() {
    [ -e "$1/metrics.json" ] && { echo "skip   $1"; return 0; }
    return 1
}


# --- 1. BABEL-60 probe, official protocol ----------------------------------------------

official_probe() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/probe/babel_60_official/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "probe  $arch/$seed/$cell"
    run "${PYTHON[@]}" scripts/probe_convex_mae.py \
        --checkpoint "$checkpoint" --config "$PROBE_CONFIG" --output "$out" \
        "model.encoder=$encoder"
}


# --- 2. segmentation -------------------------------------------------------------------

segmentation_probe() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/probe/babel_60_segmentation/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "segment  $arch/$seed/$cell"
    run "${PYTHON[@]}" scripts/probe_segmentation.py \
        --checkpoint "$checkpoint" --config "$SEG_CONFIG" --output "$out" \
        "model.encoder=$encoder"
}


# --- 3. representation geometry --------------------------------------------------------

geometry() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/representation/geometry/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "geometry  $arch/$seed/$cell"
    run "${PYTHON[@]}" scripts/representation_geometry.py \
        --checkpoint "$checkpoint" --config "$REP_CONFIG" --output "$out" \
        "model.encoder=$encoder"
}


# --- 4. kNN ----------------------------------------------------------------------------

# ponytail: the config-write and the knn.txt -> metrics.json reduction are copied from
# scripts/knn_matrix.sh, because knn_probe.py takes a checkpoint *file* and writes nothing
# -- it is used by hand and its runner adapts it. ~30 duplicated lines. Factor both helpers
# into a shared scripts/knn_helpers.sh and source it from here and from knn_matrix.sh if a
# third caller ever appears; two is not yet worth the indirection.
knn() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/representation/knn/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "knn  $arch/$seed/$cell"

    if [ -n "${DRY:-}" ]; then
        printf '  %q' "${PYTHON[@]}" scripts/knn_probe.py --config "$out/config.yaml" \
            --encoder "$encoder" --sources "<latest ckpt of $checkpoint>"; printf '\n'
        return 0
    fi

    mkdir -p "$out"
    local ckpt
    ckpt=$("${PYTHON[@]}" - "$checkpoint" "$out" "$REP_CONFIG" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, "scripts")
from probe_convex_mae import latest_checkpoint

run, out, config_path = sys.argv[1:]
OmegaConf.save(OmegaConf.load(config_path), Path(out) / "config.yaml")
print(latest_checkpoint(Path(run)))
PY
)
    "${PYTHON[@]}" scripts/knn_probe.py \
        --config "$out/config.yaml" --encoder "$encoder" --sources "$ckpt" | tee "$out/knn.txt"

    "${PYTHON[@]}" - "$out/knn.txt" "$out/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

log, out = Path(sys.argv[1]), Path(sys.argv[2])
names, rows, source = None, {}, None
for line in log.read_text().splitlines():
    fields = line.split()
    if not fields:
        continue
    if fields[0] == "source":
        names = fields[1:]
    elif names and len(fields) > 2 and fields[1].startswith("k="):
        source = fields[0]
        rows[int(fields[1][2:])] = dict(zip(names, (float(value) for value in fields[2:])))

if not rows:
    raise SystemExit(f"no kNN table in {log}")

best = max(rows, key=lambda k: rows[k]["macro_map"])
out.write_text(json.dumps({"metrics": rows[best], "k": best, "sweep": rows, "source": source}, indent=2))
PY
}


# --- 5. label efficiency ---------------------------------------------------------------

label_efficiency() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/representation/label_efficiency/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "label_efficiency  $arch/$seed/$cell"
    run "${PYTHON[@]}" scripts/label_efficiency.py \
        --checkpoint "$checkpoint" --config "$REP_CONFIG" --output "$out" \
        "model.encoder=$encoder"
}


# --- 6. reconstruction -----------------------------------------------------------------

reconstruction() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5 mask out
    for mask in $MASKS; do
        out="runs/reconstruction/$mask/$CORPUS/$arch/$seed/$cell"
        skip_done "$out" && continue
        echo "decode  $mask  $arch/$seed/$cell"
        run "${PYTHON[@]}" scripts/probe_reconstruction.py \
            --checkpoint "$checkpoint" --config "$REP_CONFIG" --output "$out" \
            --mask "$mask" --epochs "$DECODER_EPOCHS" \
            "model.encoder=$encoder"
    done
}


# --- 7. fine-tune ----------------------------------------------------------------------

finetune() {
    local arch=$1 seed=$2 cell=$3 encoder=$4 checkpoint=$5
    local out="runs/finetune/babel_60_finetune/$CORPUS/$arch/$seed/$cell"
    skip_done "$out" && return 0
    echo "finetune  $arch/$seed/$cell"
    run "${PYTHON[@]}" scripts/finetune_baseline_mae.py \
        --checkpoint "$checkpoint" --config "$FT_CONFIG" --output "$out" \
        "model.encoder=$encoder" $FT_EXTRA
}


stage "start"
echo "log: $LOG"
echo "stages: $STAGES   masks: $MASKS"
echo "corpus: $CORPUS   backbones: $(echo "$BACKBONES" | sed '/^[[:space:]]*$/d' | wc -l)"

if want 0; then
    stage "0/7  pretrain MAE in_p__loss_pv -- seeds $PV_SEEDS"
    for seed in $PV_SEEDS; do
        out="runs/pretrain/$CORPUS/medium_100ep/seed$seed/in_p__loss_pv"
        if [ -e "$out/done" ]; then echo "skip   $out"; continue; fi
        # A directory without `done` is a crashed or live run. Refuse rather than write a
        # second training into it -- same rule as the OpenSim pretraining script.
        if [ -d "$out" ]; then
            echo "Refusing: $out exists with no done marker. Confirm nothing is training," >&2
            echo "then delete it and re-run." >&2
            exit 1
        fi
        echo "pretrain  medium_100ep/seed$seed/in_p__loss_pv"
        run "${PYTHON[@]}" train.py \
            --config config/pretrain_mae.yaml config/dataloader/amass_smpl.yaml config/mae/medium.yaml \
            --output "$out" \
            "training.epochs=100" "training.seed=$seed" \
            "model.channels_input=$PV_INPUT" \
            "model.channels_output=$PV_OUTPUT"
    done
fi

want 1 && { stage "1/7  BABEL-60 probe, official protocol"; for_each_backbone official_probe; }
want 2 && { stage "2/7  segmentation probe";                for_each_backbone segmentation_probe; }
want 3 && { stage "3/7  representation geometry";           for_each_backbone geometry; }
want 4 && { stage "4/7  kNN";                               for_each_backbone knn; }
want 5 && { stage "5/7  label efficiency";                  for_each_backbone label_efficiency; }
want 6 && { stage "6/7  reconstruction (masks: $MASKS)";    for_each_backbone reconstruction; }
want 7 && { stage "7/7  fine-tune";                         for_each_backbone finetune; }

stage "summarize"
# Each page collects every corpus under its own runs/ tree, so the SMPL rows appear as a
# new corpus section beside the OpenSim ones with nothing to configure.
for page in results/probe/babel_60_official results/probe/babel_60_segmentation \
            results/representation results/reconstruction results/finetune/babel_60_finetune; do
    run "${PYTHON[@]}" "$page/gen.py"
done

stage "done"
echo "read the new \`amass_smpl\` sections of each page. The headline comparison is"
echo "OpenSim 43 DOF x 5 channels against SMPL 21 joints x 18 -- and mind the babel_paired"
echo "caveat in the header of this file before quoting the two side by side."
