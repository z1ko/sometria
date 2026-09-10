#!/usr/bin/env bash
# Overnight: score every cell of one pretraining matrix against CARE-PD's published
# evaluation protocols -- leave-one-subject-out, cross-dataset and leave-one-dataset-out.
#
#   bash goodnight_carepd.sh                     # amass_clean/medium_100ep, seed42, 9 cells
#   SEEDS="42 1 2" bash goodnight_carepd.sh      # all three pretraining replicates
#   CELLS="in_pk__loss_pk in_pk__loss_pkd" SEEDS="42 1 2" bash goodnight_carepd.sh
#   CORPUS=amass_motionx_clean bash goodnight_carepd.sh
#   DRY=1 bash goodnight_carepd.sh               # print the plan and exit
#
# 126 split sets per cell: 110 LOSO folds, 12 cross-dataset pairs, 4 LODO. MIDA is
# deliberately left out -- it is another 110 folds answering a question ("does a little
# in-domain data recover the LODO loss") that only becomes interesting once the LODO number
# itself is established.
#
# One call per cell, not per split set. The backbone is frozen, so all 126 protocols read
# the same features and probe_convex_protocol.py extracts them once; the folds are then
# L-BFGS solves on row subsets. That is the whole reason this fits in an evening: scored the
# other way round it would be 126 extractions per cell, about a day each.
#
# Both matrix passes skip work already on disk, so a crashed run is resumed by re-running
# this file rather than needing a cleanup first.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_carepd_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
# Everything below lands in the log as well as the terminal: an unattended job that fails
# at 3am is only debuggable if its output outlived the session that started it.
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-42}
# Which input/loss cells to score. Names or globs, space separated. The default is every
# cell of the matrix; narrow it to try a protocol out on two cells before committing an
# evening to twenty-seven.
CELLS=${CELLS:-in_*__loss_*}
BENCHMARK=${BENCHMARK:-carepd_updrs_convex}
CONFIG=${CONFIG:-config/experiment_probe_carepd.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}
DRY=${DRY:-}

# Cohorts are spelled out rather than globbed, and this is not fussiness: `carepd_*_loso_*`
# also matches `carepd_mida_<cohort>_loso_<k>`, because the leading `*` happily absorbs
# "mida_<cohort>". That pattern selects 220 split sets, not 110, and would quietly run the
# protocol this script exists to exclude.
SPLIT_SETS=${SPLIT_SETS:-"carepd_3DGait_loso_* carepd_BMCLab_loso_* carepd_PD-GaM_loso_* carepd_T-SDU-PD_loso_* carepd_cross_*_to_* carepd_lodo_*"}

# Tuned once and reused everywhere, which is the paper's procedure (App. C: hyperparameters
# tuned by 6-fold CV on BMCLab, then applied to every dataset). Sweeping per fold instead
# would pick weight decay on the evaluation set and report an optimistic number.
#
# It does mean the BMCLab-target results saw BMCLab data during tuning. That is true of the
# paper's numbers too, so the comparison is like for like; it is not true of the other three
# targets, which is where the cross-site claim actually rests.
TUNE_ON=${TUNE_ON:-carepd_BMCLab_6fold_1}
# Not macro_map. A LOSO fold holds out one patient, who often walks at a single severity, so
# the only class present is positive everywhere and average precision is 1.0 by construction
# -- measured at 1.0 on 5 of 14 T-SDU-PD folds. Selecting on it selects on nothing.
SELECT_ON=${SELECT_ON:-macro_f1_012}

for config in "$CONFIG" "$DATALOADER"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

# One source of truth: the corpus config says what its checkpoints normalized under, and the
# frozen backbone must be fed the same statistics or every feature is silently offset.
NORMALIZATION=$(awk '/^[[:space:]]*normalization:[[:space:]]/ {print $2; exit}' "$DATALOADER")
if [ -z "$NORMALIZATION" ]; then
    echo "no dataloader.normalization in $DATALOADER" >&2
    exit 1
fi

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep}
OUT=${OUT:-runs/probe/${BENCHMARK}/protocols/${CORPUS}/${ARCH}_${EPOCHS}ep}

stage "start"
echo "log         $LOG"
echo "benchmark   $BENCHMARK  ($CONFIG)"
echo "corpus      $CORPUS  ($DATALOADER)"
echo "normalize   $NORMALIZATION"
echo "arch        $ARCH  epochs $EPOCHS  seeds $SEEDS"
echo "cells       $CELLS"
echo "protocols   $SPLIT_SETS"
echo "tune on     $TUNE_ON  (by $SELECT_ON)"
echo "read        $PRETRAIN"
echo "write       $OUT"

found=0
for seed in $SEEDS; do
    source_root="$PRETRAIN/seed$seed"
    if [ ! -d "$source_root" ]; then
        echo "no pretraining runs at $source_root -- run scripts/pretrain_matrix.sh first" >&2
        exit 1
    fi

    for pattern in $CELLS; do
        # A mistyped cell name would otherwise expand to nothing and silently halve an
        # overnight run, so say so rather than quietly doing less work than asked.
        matched=0
        for dir in "$source_root"/$pattern; do
            [ -d "$dir" ] || continue
            matched=$((matched + 1))
            found=$((found + 1))
            name=$(basename "$dir")
            out="$OUT/seed$seed/$name"

            if [ -e "$out/metrics.json" ]; then
                echo "skip   seed$seed/$name"
                continue
            fi

            stage "probe seed$seed/$name"
            # shellcheck disable=SC2086  # SPLIT_SETS is a list of patterns on purpose
            if [ -n "$DRY" ]; then
                echo "  ${PYTHON[*]} scripts/probe_convex_protocol.py --checkpoint $dir" \
                     "--config $CONFIG --split-sets $SPLIT_SETS --tune-on $TUNE_ON" \
                     "--select-on $SELECT_ON --output $out" \
                     "dataloader.normalization=$NORMALIZATION"
            else
                "${PYTHON[@]}" scripts/probe_convex_protocol.py \
                    --checkpoint "$dir" \
                    --config "$CONFIG" \
                    --split-sets $SPLIT_SETS \
                    --tune-on "$TUNE_ON" \
                    --select-on "$SELECT_ON" \
                    --output "$out" \
                    "dataloader.normalization=$NORMALIZATION"
            fi
        done
        [ "$matched" -gt 0 ] || echo "warning: no cell matches '$pattern' under $source_root" >&2
    done
done

[ "$found" -gt 0 ] || { echo "no cell matches '$CELLS' under $PRETRAIN" >&2; exit 1; }

stage "done"
echo "read the 'pooled' column, not the per-fold mean -- see probe_convex_protocol.py"
echo "results: $OUT/*/*/metrics.json"
