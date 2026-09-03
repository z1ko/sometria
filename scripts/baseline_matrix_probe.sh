#!/usr/bin/env bash
# Train BABEL60 probes for every baseline_matrix.sh checkpoint.
#
#   ./scripts/baseline_matrix_probe.sh
#   EPOCHS=20 POOL=attentive_factorized ./scripts/baseline_matrix_probe.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})
RUNS=${RUNS:-runs/baseline_matrix}
OUT=${OUT:-runs/baseline_matrix_probe}
CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
EPOCHS=${EPOCHS:-20}
POOL=${POOL:-attentive_factorized}
DRY=${DRY:-}

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

done_already() {
    [ -e "$1/config.yaml" ] && { echo "skip  $1"; return 0; }
    return 1
}

for dir in "$RUNS"/in_*__loss_*; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    out="$OUT/$name"
    done_already "$out" && continue

    echo "probe  $name"
    run "${PYTHON[@]}" scripts/probe_baseline_mae.py \
        --checkpoint "$dir" \
        --config "$CONFIG" \
        --output "$out" \
        "training.epochs=$EPOCHS" \
        "model.pool=$POOL"
done
