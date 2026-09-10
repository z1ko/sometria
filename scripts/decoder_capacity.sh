#!/usr/bin/env bash
# Is the decoder too strong to let the loss channels shape the encoder?
#
#   ./scripts/decoder_capacity.sh            # ~1.5h pretraining + ~20min probing
#   DRY=1 ./scripts/decoder_capacity.sh      # print the plan and exit
#   ARCHES="medium_thindec" ./scripts/decoder_capacity.sh   # one rung only
#
# THE QUESTION
#
# On `amass_clean/medium_100ep`, three seeds each, BABEL-60 macro mAP:
#
#   loss_p    0.3508
#   loss_pk   0.3645     +0.0137 over loss_p
#   loss_pkd  0.3658     +0.0013 over loss_pk
#
# Adding velocity and acceleration to the reconstruction target buys ten times what adding
# torque buys, and the torque step is inside seed noise. The same null shows up on CARE-PD
# (paired t = +0.03 over 60 protocol groups). It is not that the two encoders are the same
# -- their pooled features sit at CKA 0.934, further apart than two seeds of one cell at
# 0.959 -- so the torque target does change the representation, just not in a direction a
# mean-pooled linear probe reads.
#
# THE HYPOTHESIS
#
# The decoder absorbs it. `medium` runs dec_depth 3 at full width and, at mask_ratio 0.90,
# over all 1290 tokens where the encoder sees 128 -- 8.6x the encoder's FLOPs, against
# 0.35x for MAE's own default. A decoder that large can compute torque itself: torque is
# a function of the kinematics the encoder is already required to reconstruct, and a plain
# linear map from (sin, cos, vel, acc) recovers R^2 = 0.685 of it. So the extra target
# demands nothing new of the encoder, and the decoder is thrown away afterwards.
#
# THE TEST
#
# Weaken the decoder in two steps and re-run only the two cells that differ.
#
#   medium           dec_depth 3, dec_dim 256   8.6x encoder FLOPs   (already on disk)
#   medium_thindec   dec_depth 1, dec_dim 128   ~1.0x
#   medium_tinydec   dec_depth 1, dec_dim  64   ~0.4x   (about where MAE sits)
#
# Predicted if the hypothesis holds: the pk-to-pkd gap widens monotonically across the
# three, because the encoder now has to carry what the decoder can no longer compute.
# Absolute scores are expected to *fall* for both cells at the same time -- MAE Table 1a
# finds linear-probe accuracy improves with decoder depth -- so read the gap, not the
# level. A gap that stays at ~0.001 across an 8x and a 20x cut in decoder capacity
# refutes the hypothesis and points at the readout instead.
#
# One seed per rung. This is a direction-of-effect question against an effect that has to
# beat 0.0025 seed sd to matter at all; if a rung looks promising it earns more seeds.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=${LOG:-runs/decoder_capacity_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

CORPUS=${CORPUS:-amass_clean}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-42}
# Only the two cells whose difference is in question. The other seven would be seven
# unread runs per rung.
CELLS=${CELLS:-"in_pk__loss_pk in_pk__loss_pkd"}
ARCHES=${ARCHES:-"medium_thindec medium_tinydec"}
BENCHMARK=${BENCHMARK:-babel_60_convex}
DRY=${DRY:-}

stage "start"
echo "log      $LOG"
echo "corpus   $CORPUS   epochs $EPOCHS   seeds $SEEDS"
echo "cells    $CELLS"
echo "arches   $ARCHES   (baseline 'medium' is already on disk)"
echo "probe    $BENCHMARK"

for arch in $ARCHES; do
    [ -f "config/mae/${arch}.yaml" ] && continue
    echo "missing config/mae/${arch}.yaml" >&2
    exit 1
done

for arch in $ARCHES; do
    stage "pretrain $arch"
    CORPUS=$CORPUS ARCH=$arch EPOCHS=$EPOCHS SEEDS="$SEEDS" CELLS="$CELLS" DRY=$DRY \
        ./scripts/pretrain_matrix.sh
done

# Probed on BABEL-60 rather than CARE-PD: that is where the three-seed baseline for the
# same two cells already exists, so the new numbers drop straight into the comparison.
for arch in $ARCHES; do
    stage "probe $arch"
    BENCHMARK=$BENCHMARK CORPUS=$CORPUS ARCH=$arch EPOCHS=$EPOCHS SEEDS="$SEEDS" DRY=$DRY \
        ./scripts/probe_matrix.sh
done

stage "summarize"
[ -n "$DRY" ] || uv run python results/probe/${BENCHMARK}/gen.py

stage "done"
cat <<'EOF'

Compare the pk -> pkd gap across decoder capacity. Baseline, 3 seeds:

    medium (8.6x)   loss_pk 0.3645   loss_pkd 0.3658   gap +0.0013

Widening monotonically supports "the decoder absorbs the torque target".
A flat gap does not, and points at the mean-pool linear readout instead.
EOF
