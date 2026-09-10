#!/usr/bin/env bash
# 3x3 MAE pretraining sweep -- encoder input channels x reconstruction/loss channels --
# for one (corpus, architecture, epoch budget) point.
#
#   ./scripts/pretrain_matrix.sh
#   CORPUS=amass_motionx_complete ARCH=medium EPOCHS=100 ./scripts/pretrain_matrix.sh
#   ARCH=tiny EPOCHS=2 DRY=1 ./scripts/pretrain_matrix.sh
#
# The output path is derived, never typed:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep/in_<x>__loss_<y>
#
# Every dimension gets a slot and none is optional. That is the whole point: the old flat
# names marked MotionX with a `_with_motionx` suffix and marked AMASS-only with nothing, so
# a directory could silently mean "whatever the default corpus was that week".
#
# CORPUS is simultaneously the dataloader config name and the path segment, so there is one
# string to change and no mapping table to keep in sync:
#
#   CORPUS=amass_motionx_complete
#     -> config/dataloader/amass_motionx_complete.yaml
#     -> runs/pretrain/amass_motionx_complete/...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_motionx_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

# Merged left to right by train.py, so later files win: the base carries model and training
# defaults, the dataloader config decides which corpus and which normalization statistics,
# and the size config overrides the model dimensions only.
BASE=${BASE:-config/pretrain_mae.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}
SIZE=${SIZE:-config/mae/${ARCH}.yaml}

RUNS=${RUNS:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep}
DRY=${DRY:-}

for config in "$BASE" "$DATALOADER" "$SIZE"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

names=(p pk pkd)
channels=("[0,1]" "[0,1,2,3]" "[0,1,2,3,4]")

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

done_already() {
    [ -e "$1/config.yaml" ] && { echo "skip   $1"; return 0; }
    return 1
}

# Print the resolved plan before doing anything. Reading it back off the paths afterwards
# is what the old layout made hard, and a dry run should show the same lines as a real one.
echo "corpus     $CORPUS  ($DATALOADER)"
echo "arch       $ARCH  ($SIZE)"
echo "epochs     $EPOCHS"
echo "output     $RUNS"
echo

for i in "${!names[@]}"; do
    for j in "${!names[@]}"; do
        name="in_${names[$i]}__loss_${names[$j]}"
        out="$RUNS/$name"
        done_already "$out" && continue

        echo "train  $name"
        run "${PYTHON[@]}" train.py \
            --config "$BASE" "$DATALOADER" "$SIZE" \
            --output "$out" \
            "training.epochs=$EPOCHS" \
            "model.channels_input=${channels[$i]}" \
            "model.channels_output=${channels[$j]}"
    done
done
