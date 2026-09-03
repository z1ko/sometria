#!/usr/bin/env bash
# 3x3 baseline MAE sweep: encoder input channels x reconstruction/loss channels.
#
#   ./scripts/baseline_matrix.sh
#   EPOCHS=10 ./scripts/baseline_matrix.sh
#   DRY=1 ./scripts/baseline_matrix.sh
#   RUNS=runs/other PYTHON=python ./scripts/baseline_matrix.sh
#   SIZE=config/mae/tiny.yaml ./scripts/baseline_matrix.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})
CONFIG=${CONFIG:-config/pretrain_mae.yaml}
SIZE=${SIZE:-config/mae/small.yaml}
RUNS=${RUNS:-runs/baseline_matrix}
EPOCHS=${EPOCHS:-10}
DRY=${DRY:-}

names=(p pk pkd)
channels=("[0,1]" "[0,1,2,3]" "[0,1,2,3,4]")

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

epochs_override() {
    [ -n "$EPOCHS" ] && echo "training.epochs=$EPOCHS"
}

done_already() {
    [ -e "$1/config.yaml" ] && { echo "skip  $1"; return 0; }
    return 1
}

for i in "${!names[@]}"; do
    for j in "${!names[@]}"; do
        name="in_${names[$i]}__loss_${names[$j]}"
        out="$RUNS/$name"
        done_already "$out" && continue

        echo "train  $name"
        run "${PYTHON[@]}" train.py --config "$CONFIG" "$SIZE" --output "$out" \
            $(epochs_override) \
            "model.channels_input=${channels[$i]}" \
            "model.channels_output=${channels[$j]}"
    done
done
