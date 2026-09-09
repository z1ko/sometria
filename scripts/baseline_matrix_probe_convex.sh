#!/usr/bin/env bash
# Convex-probe (mean-pool + L-BFGS) every baseline_matrix.sh checkpoint.
# Each run internally sweeps weight_decay by default -- see probe_convex_mae.py.
#
#   ./scripts/baseline_matrix_probe_convex.sh
#   RUNS=runs/baseline_matrix_100ep_medium ./scripts/baseline_matrix_probe_convex.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})
RUNS=${RUNS:-runs/baseline_matrix}
OUT=${OUT:-runs/baseline_matrix_probe_convex}
CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
DRY=${DRY:-}

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

done_already() {
    [ -e "$1/metrics.json" ] && { echo "skip  $1"; return 0; }
    return 1
}

for dir in "$RUNS"/in_*__loss_*; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    out="$OUT/$name"
    done_already "$out" && continue

    echo "probe  $name"
    run "${PYTHON[@]}" scripts/probe_convex_mae.py \
        --checkpoint "$dir" \
        --config "$CONFIG" \
        --output "$out"
done
