#!/usr/bin/env bash
# Overnight: the two axes MAE has finished and the other objectives have not, plus the
# first half of evaluation-plan phase 2. No pretraining -- every backbone exists.
#
#   bash goodnight.sh                     # all three stages
#   DRY=1 bash goodnight.sh               # print the plan and exit
#   STAGES="3" bash goodnight.sh          # a subset, by the numbers below
#   MASKS="random suffix" bash goodnight.sh   # see stage 3
#
# Where the core matrix stands (128 backbones: amass_clean x MAE 16x4, JEPA 4x4,
# SimMIM 16x3). Both window probes are complete at 128/128. Segmentation, geometry, kNN
# and label-efficiency are complete for MAE at 64/64 and empty for the other two.
# Reconstruction has 10 runs, all suffix, all seed2, all on the pre-`pv` cell set.
#
#   1. segmentation, JEPA      16 cells.
#   2. segmentation, SimMIM    48 cells.
#   3. reconstruction, suffix  54 cells.
#
# On stage 2, so it is on the record rather than a surprise in the table: SimMIM is below
# the moments floor on all 16 window cells at 3 seeds, and below it again on the BABEL
# chunk protocol at 0/16. The expected result here is that it is below the segmentation
# floor too. That is worth 48 cells only as a published negative -- if you want the cheap
# version, STAGES="1 3" skips it and costs nothing else.
#
# On stage 3, MASKS is `suffix` rather than the runner's own `"random suffix"` default.
# Suffix is forecasting, random is reconstruction, and the plan specifies both as one rig
# with one decoder. Widening it to both makes the stage 118 cells instead of 54 and is a
# one-word edit here. Nothing downstream assumes one or the other.
#
# The decoder budget is 40 epochs and that is deliberate -- see the DECODER_EPOCHS comment
# in scripts/reconstruction_matrix.sh, where 10 was measured to censor the metric. It also
# means this stage has no measured per-cell wall time in this repo, so stage 3 is the one
# that could run long. It is last for that reason: stages 1 and 2 land their pages first.
#
# Every runner checks the backbone's `done` marker itself and prints `UNFINISHED ...
# skipping`, so there is no orphan preflight and nothing here deletes anything. Every step
# skips work already on disk, so a crash is resumed by re-running this file.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
# Everything below lands in the log as well as the terminal: an unattended job that fails
# at 3am is only debuggable if its output outlived the session that started it.
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

maybe() {
    if [ -n "${DRY:-}" ]; then printf '  %q' "$@"; printf '\n'; else "$@"; fi
}

STAGES=${STAGES:-"1 2 3"}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Seeds per objective: MAE and JEPA hold 1/2/3/42 on amass_clean, SimMIM only 1/2/42.
JEPA_SEEDS=${JEPA_SEEDS:-"1 2 3 42"}
SIMMIM_SEEDS=${SIMMIM_SEEDS:-"1 2 42"}
RECON_SEEDS=${RECON_SEEDS:-"1 2 3 42"}
MASKS=${MASKS:-suffix}

stage "start"
echo "log: $LOG"
echo "stages: $STAGES   masks: $MASKS"


if want 1; then
    stage "1/3  segmentation -- JEPA seeds $JEPA_SEEDS"
    OBJECTIVE=jepa SEEDS="$JEPA_SEEDS" ./scripts/segmentation_matrix.sh
fi

if want 2; then
    stage "2/3  segmentation -- SimMIM seeds $SIMMIM_SEEDS"
    OBJECTIVE=simmim SEEDS="$SIMMIM_SEEDS" ./scripts/segmentation_matrix.sh
fi

if want 1 || want 2; then
    # The patch and window moments floors are already in baselines.json and the task has
    # not changed, so they are not re-run.
    stage "1-2/3  summarize segmentation"
    maybe uv run python results/probe/babel_60_segmentation/gen.py
fi


if want 3; then
    stage "3/3  reconstruction (masks: $MASKS) -- MAE seeds $RECON_SEEDS"
    MASKS="$MASKS" SEEDS="$RECON_SEEDS" ./scripts/reconstruction_matrix.sh

    stage "3/3  summarize reconstruction"
    maybe uv run python results/reconstruction/gen.py
fi


stage "done"
echo "read: results/probe/babel_60_segmentation/README.md  (does either objective move boundary F1?)"
echo "      results/reconstruction/README.md               (against the zero- and constant-velocity floors)"
