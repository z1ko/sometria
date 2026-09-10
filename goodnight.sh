#!/usr/bin/env bash
# Overnight: two extra pretraining seeds for amass_clean/medium_100ep, probe all three,
# regenerate the results summary.
#
#   bash goodnight.sh          # ~7.6h: 6.6h pretraining (18 cells) + 1h probing
#   DRY=1 bash goodnight.sh    # print the plan and exit
#
# Existing seed42 is the run that was already there; the probe pass names it so that a
# deleted result gets redone, and skips it otherwise.
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

stage "pretrain amass_clean medium_100ep seeds 1 2"
CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS="1 2" ./scripts/pretrain_matrix.sh

stage "probe amass_clean medium_100ep seeds 42 1 2"
CORPUS=amass_clean ARCH=medium EPOCHS=100 SEEDS="42 1 2" ./scripts/probe_matrix.sh

stage "summarize"
uv run python results/probe/babel_60_convex/gen.py

stage "done"
echo "read: results/probe/babel_60_convex/README.md  (Replicate spread section)"
