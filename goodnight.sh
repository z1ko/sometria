#!/usr/bin/env bash
# Overnight: extra pretraining seeds for amass_motionx_clean/medium_100ep, probe them all,
# regenerate the results summary.
#
#   bash goodnight.sh                # SEEDS defaults to "1 2"
#   SEEDS=1 bash goodnight.sh        # one seed, ~12h, fits a single night
#   DRY=1 bash goodnight.sh          # print the plan and exit
#
# This is the run that decides the MotionX question. amass_clean already has three seeds
# and sigma = 0.0025; MotionX has one, which is why every corpus delta in
# results/probe/babel_60_convex/README.md is marginal at best.
#
#   SEEDS=1    -> n=3 vs n=2, SE 0.0023, the observed +0.0069 becomes 3.0 sigma
#   SEEDS="1 2"-> n=3 vs n=3, SE 0.0020, it becomes 3.4 sigma
#
# Bonferroni over the nine cells needs 2.77 sigma, so one extra seed already decides the
# headline comparison. The second buys a sigma estimate for MotionX itself, which is
# currently assumed equal to AMASS's and untested.
#
# MotionX is ~3.4x the steps per epoch: 76 min per cell against 22 for amass_clean. Two
# seeds is ~24h, not one night.
#
# seed42 is the run that was already there, now filed under seed42/. The probe pass names
# it so a deleted result gets redone, and skips it otherwise.
#
# Fails fast on purpose. Both matrix scripts skip work that is already on disk, so a
# crashed run is resumed by re-running this file rather than needing a cleanup first.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
# Everything below lands in the log as well as the terminal: an unattended job that fails
# at 3am is only debuggable if its output outlived the session that started it.
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

stage "start"
echo "log: $LOG"

CORPUS=${CORPUS:-amass_motionx_clean}
SEEDS=${SEEDS:-1 2}

stage "pretrain $CORPUS medium_100ep seeds $SEEDS"
CORPUS=$CORPUS ARCH=medium EPOCHS=100 SEEDS="$SEEDS" ./scripts/pretrain_matrix.sh

# 42 first so an existing result is confirmed rather than assumed; it skips in seconds.
stage "probe $CORPUS medium_100ep seeds 42 $SEEDS"
CORPUS=$CORPUS ARCH=medium EPOCHS=100 SEEDS="42 $SEEDS" ./scripts/probe_matrix.sh

stage "summarize"
uv run python results/probe/babel_60_convex/gen.py

stage "done"
echo "read: results/probe/babel_60_convex/README.md  (Replicate spread section)"
