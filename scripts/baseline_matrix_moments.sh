#!/usr/bin/env bash
# Moments baselines for baseline_matrix.sh.
# Only three arms are needed: moments depend on input channels, not MAE loss channels.
#
#   ./scripts/baseline_matrix_moments.sh
#   PASSES=4 EPOCHS=40 ./scripts/baseline_matrix_moments.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})
CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
RUNS=${RUNS:-runs/baseline_matrix}
PASSES=${PASSES:-4}
EPOCHS=${EPOCHS:-40}
DRY=${DRY:-}

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

mkdir -p "$RUNS"

run "${PYTHON[@]}" scripts/moments_baseline.py \
    --config "$CONFIG" \
    --passes "$PASSES" \
    --epochs "$EPOCHS" \
    --channels 0,1 0,1,2,3 all | tee "$RUNS/moments.txt"
