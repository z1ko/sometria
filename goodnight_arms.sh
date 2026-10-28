#!/usr/bin/env bash
# Overnight: the three-arm representation comparison -- OpenSim, SMPL and COCO keypoints --
# pretrained and scored on exactly the same AMASS takes.
#
#   bash goodnight_arms.sh                          # all stages
#   DRY=1 bash goodnight_arms.sh                    # print the plan and exit
#   STAGES="0" bash goodnight_arms.sh               # a subset, by the numbers below
#   ARMS_ONLY="coco" SEEDS="1" bash goodnight_arms.sh
#   ARCHS="medium_100ep medium_100ep_jepa" bash goodnight_arms.sh
#
# The question is which input representation of the same motion pretrains best:
#
#   opensim  43 anatomical DOFs x  5 channels  (sin, cos, vel, acc, tau)
#   smpl     21 body joints     x 18 channels  (rot6d, vel, acc)
#   coco     18 keypoints       x  9 channels  (body-frame xyz, vel, acc)
#
# Everything else is held equal, and this file is where that is enforced:
#
#   - Same takes. Every arm trains on `pretrain_paired_coco_v1` and is scored on
#     `babel_paired_coco`, both built by notebooks/preprocess_amass_coco.ipynb as the
#     intersection of the three imports. AMASS_COCO3D covers 5 subsets, so this is about a
#     third of `pretrain_paired_v1`: numbers here are comparable across arms, NOT with the
#     amass_clean / amass_smpl pages under results/.
#   - Same configs. Every task reads the `_coco` experiment config for every arm, with only
#     the corpus keys overridden per arm (joint definition, statistics, source dataset,
#     encoder shape). The OpenSim base configs differ from the `_smpl`/`_coco` ones in more
#     than the corpus -- pooling, checkpoint submodule -- so they are deliberately not used.
#   - Same cells, by meaning rather than index: the full input x loss grid over each arm's
#     channel ladder (p, pv, pk, and pkd where there is torque), translated per arm in
#     ARMS below -- 16 cells for OpenSim, 9 each for SMPL and COCO. One asymmetry stays:
#     a derivative is 1 channel per OpenSim DOF, 6 per SMPL joint, 3 per keypoint, so the
#     same cell is a different-sized target per arm.
#   - Same seeds, so arms pair up replicate for replicate and gen.py can test differences
#     with a paired t-test.
#
#   0. pretrain                         arms x archs x cells x seeds  (102 by default)
#   1. BABEL-60 probe, official protocol
#   2. segmentation probe
#   3. representation: geometry
#   4. representation: kNN
#   5. representation: label efficiency
#   6. fine-tune
#
# Not here: reconstruction, which scores joint angles through atan2(sin, cos) and so only
# describes the OpenSim arm (see the header of goodnight_smpl.sh); CKA, which compares cells
# of one layout. Results land under runs/arms/<task>/<arm>/<arch>/seed<N>/<cell>/ and are
# summarized by results/arms/gen.py.
#
# Every step skips work already on disk (`done` / `metrics.json`) and refuses a backbone
# with no `done` marker, so a crash is resumed by re-running this file. Nothing here deletes.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_arms_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

PYTHON=(${PYTHON:-uv run python})

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

run() {
    if [ -n "${DRY:-}" ]; then printf '  %q' "$@"; printf '\n'
    else "$@"; fi
}

STAGES=${STAGES:-"0 1 2 3 4 5 6"}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

RUNS=${RUNS:-runs/arms}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-"1 2 42"}
ARMS_ONLY=${ARMS_ONLY:-"opensim smpl coco"}
ARCHS=${ARCHS:-"medium_100ep"}

PRETRAIN_SPLIT=pretrain_paired_coco_v1
STATS_NAME=${PRETRAIN_SPLIT}_train_clean
# The template every arm is overridden from. Its split sets are already the three-way ones.
DATALOADER=config/dataloader/amass_coco.yaml
PROBE_CONFIG=config/experiment_probe_babel_official_coco.yaml
SEG_CONFIG=config/experiment_segmentation_coco.yaml
REP_CONFIG=config/experiment_linear_probe_coco.yaml
FT_CONFIG=config/experiment_finetune_coco.yaml
FT_EXTRA=${FT_EXTRA:-"model.layer_decay=0.5 training.patience=20"}

# One row per arm:
#   <arm> <source_dataset> <human> <representation> <dofs> <features> <channel sets...>
# The channel sets are the ladder of scripts/pretrain_matrix.sh by meaning -- p pose, pv
# + velocity, pk + acceleration, pkd + torque -- translated into each arm's indices. An arm
# sweeps every set it has: SMPL and COCO carry no torque, so they stop at pk.
ARMS="
opensim AMASS      config/human.yaml opensim_sincos_log_vel_acc_tau_v2 43  5 p=[0,1] pv=[0,1,2] pk=[0,1,2,3] pkd=[0,1,2,3,4]
smpl    AMASS-SMPL config/smpl.yaml  smpl_rot6d_log_vel_acc_v1         21 18 p=[0,1,2,3,4,5] pv=[0,1,2,3,4,5,6,7,8,9,10,11] pk=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
coco    AMASS-COCO config/coco.yaml  coco18_canonpos_vel_acc_v1        18  9 p=[0,1,2] pv=[0,1,2,3,4,5] pk=[0,1,2,3,4,5,6,7,8]
"

# One row per pretraining objective:
#   <arch> <base config> <size config> <checkpoint submodule> <loss axis: yes|no>
# Cells are the arm's full input x loss grid (`in_pv__loss_pk`). JEPA reconstructs
# nothing, so it has no loss axis and sweeps inputs only (`in_pv`). The submodule is the
# objective's, not a preference: JEPA has no `backbone.` at all.
ARCH_TABLE="
medium_100ep        config/pretrain_mae.yaml    config/mae/medium.yaml    backbone yes
medium_100ep_jepa   config/pretrain_jepa.yaml   config/jepa/medium.yaml   teacher  no
medium_100ep_simmim config/pretrain_simmim.yaml config/simmim/medium.yaml backbone yes
"
CELLS_ONLY=${CELLS_ONLY:-}   # e.g. "in_p__loss_p in_p__loss_pv" for a subset of the grid

rows() { echo "$1" | sed '/^[[:space:]]*$/d'; }

# Sets the arm's columns as globals: source human representation dofs features sets
load_arm() {
    local name
    while read -r name source human representation dofs features sets; do
        [ "$name" = "$1" ] && return 0
    done <<< "$(rows "$ARMS")"
    echo "unknown arm: $1" >&2; exit 1
}

# The corpus keys, and only those. Shared by pretraining and every downstream config, so an
# arm cannot be scored under a different joint definition or statistics than it trained on.
corpus_overrides() {
    printf '%s\n' \
        "dataloader.human=$human" \
        "dataloader.normalization=stats/$representation/$STATS_NAME.pt" \
        "dataloader.train.source_datasets=[$source]" \
        "dataloader.val.source_datasets=[$source]"
}

# read -ra, not `for set in $sets`: an unquoted `[0,1]` is a glob pattern.
set_names() { local set all; read -ra all <<< "$sets"; for set in "${all[@]}"; do echo "${set%%=*}"; done; }

channels() {   # p | pv | pk | pkd -> the current arm's channel list
    local set all
    read -ra all <<< "$sets"
    for set in "${all[@]}"; do [ "${set%%=*}" = "$1" ] && { echo "${set#*=}"; return 0; }; done
    echo "arm has no channel set '$1'" >&2; exit 1
}

# The arm's cells for one objective, in ladder order.
cells_of() {
    local input loss
    for input in $(set_names); do
        if [ "$1" = yes ]; then
            for loss in $(set_names); do echo "in_${input}__loss_${loss}"; done
        else
            echo "in_${input}"
        fi
    done
}

# Iterate arms x archs x seeds x cells, calling `$1 <arm> <arch> <seed> <cell> <encoder>
# <base> <size>` with the arm loaded.
for_each_cell() {
    local task=$1 arm arch base size encoder axis cell seed
    for arm in $ARMS_ONLY; do
        load_arm "$arm"
        # fd 3, not stdin: a task that reads stdin would otherwise eat the remaining rows.
        while read -r arch base size encoder axis <&3; do
            [[ " $ARCHS " == *" $arch "* ]] || continue
            for seed in $SEEDS; do
                for cell in $(cells_of "$axis"); do
                    [ -n "$CELLS_ONLY" ] && [[ " $CELLS_ONLY " != *" $cell "* ]] && continue
                    "$task" "$arm" "$arch" "$seed" "$cell" "$encoder" "$base" "$size"
                done
            done
        done 3<<< "$(rows "$ARCH_TABLE")"
    done
}

pretrained() { echo "$RUNS/pretrain/$1/$2/seed$3/$4"; }

skip_done() {
    [ -e "$1/metrics.json" ] && { echo "skip   $1"; return 0; }
    return 1
}

# Downstream wrapper: refuses unfinished backbones, skips scored ones, then runs the task.
# The `done` marker rather than the directory: a backbone still training would be scored at
# whatever epoch it has reached and then sit in the tables as a replicate.
downstream() {
    local task=$1 script=$2 config=$3; shift 3
    local arm=$1 arch=$2 seed=$3 cell=$4 encoder=$5
    local checkpoint out
    checkpoint=$(pretrained "$arm" "$arch" "$seed" "$cell")
    out="$RUNS/$task/$arm/$arch/seed$seed/$cell"
    if [ -z "${DRY:-}" ] && [ ! -e "$checkpoint/done" ]; then
        echo "UNFINISHED  $checkpoint -- no done marker, skipping"; return 0
    fi
    skip_done "$out" && return 0
    echo "$task  $arm/$arch/seed$seed/$cell"
    mapfile -t corpus < <(corpus_overrides)
    run "${PYTHON[@]}" "$script" --checkpoint "$checkpoint" --config "$config" --output "$out" \
        "${corpus[@]}" "encoder.num_dofs=$dofs" "encoder.num_features=$features" \
        "model.encoder=$encoder" "${EXTRA[@]}"
}


# --- 0. pretrain -----------------------------------------------------------------------

pretrain() {
    local arm=$1 arch=$2 seed=$3 cell=$4 base=$6 size=$7 out input loss
    out=$(pretrained "$arm" "$arch" "$seed" "$cell")
    if [ -e "$out/done" ]; then echo "skip   $out"; return 0; fi
    # A directory without `done` is a crashed or live run. Refuse rather than write a second
    # training into it -- same rule as every other pretraining script here.
    if [ -d "$out" ]; then
        echo "Refusing: $out exists with no done marker. Confirm nothing is training," >&2
        echo "then delete it and re-run." >&2
        exit 1
    fi

    input=${cell#in_}; input=${input%%__*}
    local model=("model.channels_input=$(channels "$input")")
    if [[ $cell == *__loss_* ]]; then
        loss=${cell##*__loss_}
        model+=("model.channels_output=$(channels "$loss")")
    fi

    echo "pretrain  $arm/$arch/seed$seed/$cell"
    mapfile -t corpus < <(corpus_overrides)
    run "${PYTHON[@]}" train.py --config "$base" "$DATALOADER" "$size" --output "$out" \
        "${corpus[@]}" "model.num_dofs=$dofs" \
        "training.epochs=$EPOCHS" "training.seed=$seed" "${model[@]}"
}


# --- 1-6. downstream -------------------------------------------------------------------

EXTRA=()
official_probe()     { downstream probe/babel_60_official   scripts/probe_convex_mae.py       "$PROBE_CONFIG" "$@"; }
segmentation_probe() { downstream probe/babel_60_segmentation scripts/probe_segmentation.py  "$SEG_CONFIG"   "$@"; }
geometry()           { downstream representation/geometry  scripts/representation_geometry.py "$REP_CONFIG" "$@"; }
label_efficiency()   { downstream representation/label_efficiency scripts/label_efficiency.py "$REP_CONFIG" "$@"; }
finetune() {
    local EXTRA; read -ra EXTRA <<< "$FT_EXTRA"
    downstream finetune/babel_60_finetune scripts/finetune_baseline_mae.py "$FT_CONFIG" "$@"
}

# knn_probe.py takes a checkpoint *file* and a config, and prints rather than writes, so it
# gets its own adapter: the overridden config is saved beside the result, and the printed
# table reduced by scripts/knn_reduce.py.
knn() {
    local arm=$1 arch=$2 seed=$3 cell=$4 encoder=$5 checkpoint out ckpt
    checkpoint=$(pretrained "$arm" "$arch" "$seed" "$cell")
    out="$RUNS/representation/knn/$arm/$arch/seed$seed/$cell"
    if [ -z "${DRY:-}" ] && [ ! -e "$checkpoint/done" ]; then
        echo "UNFINISHED  $checkpoint -- no done marker, skipping"; return 0
    fi
    skip_done "$out" && return 0
    echo "knn  $arm/$arch/seed$seed/$cell"
    if [ -n "${DRY:-}" ]; then
        printf '  %q' "${PYTHON[@]}" scripts/knn_probe.py --config "$out/config.yaml" \
            --encoder "$encoder" --sources "<latest ckpt of $checkpoint>"; printf '\n'
        return 0
    fi

    mkdir -p "$out"
    ckpt=$("${PYTHON[@]}" - "$checkpoint" "$out" "$REP_CONFIG" \
        $(corpus_overrides) "encoder.num_dofs=$dofs" "encoder.num_features=$features" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, "scripts")
from probe_convex_mae import latest_checkpoint

run, out, config_path, *overrides = sys.argv[1:]
config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(overrides))
OmegaConf.save(config, Path(out) / "config.yaml")
print(latest_checkpoint(Path(run)))
PY
)
    "${PYTHON[@]}" scripts/knn_probe.py \
        --config "$out/config.yaml" --encoder "$encoder" --sources "$ckpt" | tee "$out/knn.txt"
    "${PYTHON[@]}" scripts/knn_reduce.py "$out/knn.txt" "$out/metrics.json"
}


stage "start"
echo "log: $LOG"
echo "stages: $STAGES   arms: $ARMS_ONLY   archs: $ARCHS   seeds: $SEEDS   epochs: $EPOCHS"

# Fail before the first GPU hour, not after it: every arm needs its statistics on the
# three-way split, which only the preprocessing notebook writes.
for arm in $ARMS_ONLY; do
    load_arm "$arm"
    stats="data/processed/stats/$representation/$STATS_NAME.pt"
    if [ ! -e "$stats" ]; then
        echo "missing $stats -- run notebooks/preprocess_amass_coco.ipynb first" >&2
        exit 1
    fi
done

want 0 && { stage "0/6  pretrain";                          for_each_cell pretrain; }
want 1 && { stage "1/6  BABEL-60 probe, official protocol"; for_each_cell official_probe; }
want 2 && { stage "2/6  segmentation probe";                for_each_cell segmentation_probe; }
want 3 && { stage "3/6  representation geometry";           for_each_cell geometry; }
want 4 && { stage "4/6  kNN";                               for_each_cell knn; }
want 5 && { stage "5/6  label efficiency";                  for_each_cell label_efficiency; }
want 6 && { stage "6/6  fine-tune";                         for_each_cell finetune; }

stage "summarize"
run "${PYTHON[@]}" results/arms/gen.py

stage "done"
echo "read results/arms/README.md"
