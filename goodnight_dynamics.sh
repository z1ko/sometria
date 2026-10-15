#!/usr/bin/env bash
# Overnight: the forecast_dynamics benchmark -- can frozen backbones forecast joint torque
# (and vel, acc, angles), and infer it from motion? No pretraining; every backbone exists.
#
#   bash goodnight_dynamics.sh                 # all stages
#   DRY=1 bash goodnight_dynamics.sh           # print the plan and exit
#   STAGES="1" bash goodnight_dynamics.sh      # a subset, by the numbers below
#
#   1. MAE   16 cells x 4 seeds: forecast + infer (64 + 64), infer_hold on in_pkd* (16).
#   2. JEPA   4 cells x 4 seeds: forecast + infer (16 + 16), infer_hold on in_pkd (4).
#   3. summarize.
#
# Measured: 324 s per run on one RTX 5090 (forecast, in_pkd__loss_pkd, 40 decoder epochs,
# including the linear-floor fit pass). 180 runs is ~16 h -- more than one night. Every run
# is resumable, so either let it spill into the next day or start with SEEDS="1 2 3".
#
# No SimMIM: it is below the moments floor everywhere under a frozen probe, and a
# collapsed encoder forecasting torque would be scored on the decoder alone.
#
# Every runner checks the backbone's `done` marker and skips cells whose metrics.json
# exists, so a crash is resumed by re-running this file.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_dynamics_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

maybe() {
    if [ -n "${DRY:-}" ]; then printf '  %q' "$@"; printf '\n'; else "$@"; fi
}

STAGES=${STAGES:-"1 2 3"}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

SEEDS=${SEEDS:-"1 2 3 42"}
MODES=${MODES:-"forecast infer infer_hold"}

stage "start"
echo "log: $LOG"
echo "stages: $STAGES   seeds: $SEEDS   modes: $MODES"

if want 1; then
    stage "1/3  forecast_dynamics -- MAE seeds $SEEDS"
    MODES="$MODES" SEEDS="$SEEDS" ./scripts/forecast_dynamics_matrix.sh
fi

if want 2; then
    stage "2/3  forecast_dynamics -- JEPA seeds $SEEDS"
    OBJECTIVE=jepa MODES="$MODES" SEEDS="$SEEDS" ./scripts/forecast_dynamics_matrix.sh
fi

if want 3; then
    stage "3/3  summarize"
    maybe uv run python results/forecast_dynamics/gen.py
fi

stage "done"
echo "read: results/forecast_dynamics/README.md  (H1-H6 against the floors)"
