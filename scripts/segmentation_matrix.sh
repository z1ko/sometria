#!/usr/bin/env bash
# Segmentation-probe every cell of one pretraining matrix. Mirrors scripts/probe_matrix.sh
# line for line -- what differs is the script it calls, the config it defaults to, and where
# the results land.
#
#   ./scripts/segmentation_matrix.sh
#   OBJECTIVE=jepa SEEDS="42 1 2" ./scripts/segmentation_matrix.sh
#   CELLS="in_pk__loss_pk" SEEDS="42 1 2" DRY=1 ./scripts/segmentation_matrix.sh
#
# Same four knobs as the pretraining sweep, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>          read
#   runs/probe/babel_60_segmentation/<corpus>/<arch>_<epochs>ep<objective>/<cell> written
#
# OBJECTIVE mirrors the slot pretrain_matrix.sh writes, and carries the same asymmetry:
# empty for MAE so the paths match the window probe tree cell for cell, `_jepa` and so on
# otherwise. A cell is `in_<x>__loss_<y>` where the objective has a loss axis and plain
# `in_<x>` where it does not, so CELLS below matches on the part they share.
#
# The normalization statistics are taken from the *corpus* config, not from the probe
# config, and this is the whole reason the corpus is a knob here rather than a path segment.
# The backbone is frozen: feed it windows normalized by statistics other than the ones it
# pretrained under and every feature is offset, silently and without error.
# config/experiment_segmentation.yaml pins pretrain_v1 statistics, which are wrong for any
# checkpoint pretrained on anything else.
#
# Cost note. Unlike the window probe this extracts 30 rows per window, so the feature matrix
# is 30x taller. training.train_passes is 2 rather than 20 for that reason; raise it through
# EXTRA only after reading segmentation_convex.DEFAULT_TRAIN_PASSES, and expect
# check_feature_memory to refuse rather than the run to OOM.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

BENCHMARK=${BENCHMARK:-babel_60_segmentation}
CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

CONFIG=${CONFIG:-config/experiment_segmentation.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# Which pretraining objective's tree to read. Named rather than sniffed, for probe_matrix.sh's
# reason: this script only ever sees the probe config and the corpus.
OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

# Two independent replicate axes, because they measure different things.
#
# SEEDS selects which *pretraining* replicate to read, and mirrors into the output path so a
# probe is always filed under the backbone it scored. Empty means the canonical run.
#
# REPEATS re-probes one backbone. The probe is not deterministic: the training split draws a
# random crop per sample per pass, so refitting the same frozen backbone gives a slightly
# different head. Repeats bound that, and it is the smaller of the two.
SEEDS=${SEEDS:-}
REPEATS=${REPEATS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi
if [ -z "$REPEATS" ]; then repeats=(""); else read -ra repeats <<< "$REPEATS"; fi

# Which cells to run, as a glob against the pretraining tree -- the same knob and the same
# default as finetune_matrix.sh, so "the nine cells the fine-tune table ranks" is one
# spelling in both. `in_*` is everything and matches both naming schemes.
CELLS=${CELLS:-in_*}
# Extra OmegaConf dotlist overrides, appended verbatim to every call. Not validated here:
# OmegaConf already rejects a key the config has no slot for.
#
#   EXTRA="training.train_passes=1" ./scripts/segmentation_matrix.sh
EXTRA=${EXTRA:-}

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT=${OUT:-runs/probe/${BENCHMARK}/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
DRY=${DRY:-}

for config in "$CONFIG" "$DATALOADER"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

# One source of truth: whatever the corpus config says it normalized with is what the probe
# applies. Reading it back beats restating the path here.
NORMALIZATION=$(awk '/^[[:space:]]*normalization:[[:space:]]/ {print $2; exit}' "$DATALOADER")
if [ -z "$NORMALIZATION" ]; then
    echo "no dataloader.normalization in $DATALOADER" >&2
    exit 1
fi

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

done_already() {
    [ -e "$1/metrics.json" ] && { echo "skip   $1"; return 0; }
    return 1
}

echo "benchmark  $BENCHMARK  ($CONFIG)"
echo "corpus     $CORPUS  ($DATALOADER)"
echo "normalize  $NORMALIZATION"
echo "arch       $ARCH"
echo "objective  $OBJECTIVE"
echo "epochs     $EPOCHS"
echo "cells      $CELLS"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
echo "repeats    ${REPEATS:-<none, probe seed from config>}"
echo "read       $PRETRAIN"
echo "write      $OUT"
echo

found=0
for seed in "${seeds[@]}"; do
    source_root="$PRETRAIN${seed:+/seed$seed}"
    if [ ! -d "$source_root" ]; then
        echo "no pretraining runs at $source_root -- run scripts/pretrain_matrix.sh first" >&2
        exit 1
    fi

    for repeat in "${repeats[@]}"; do
        out_root="$OUT${seed:+/seed$seed}${repeat:+/rep$repeat}"
        [ -n "$seed$repeat" ] && echo "== ${seed:+seed $seed }${repeat:+repeat $repeat }-> $out_root"

        for dir in "$source_root"/$CELLS; do
            [ -d "$dir" ] || continue
            found=$((found + 1))
            name=$(basename "$dir")
            out="$out_root/$name"
            done_already "$out" && continue

            # The pretraining `done` marker, not just the directory. Without this a cell
            # still training is read at whatever checkpoint it has reached -- which happened
            # once and sat in the tables as a replicate. Loud, because a silent skip is how
            # that went unnoticed.
            if [ ! -e "$dir/done" ]; then
                echo "UNFINISHED  $dir -- no done marker, skipping"
                continue
            fi
            echo "segment  $name"
            # A repeat index doubles as the probe seed. probe_segmentation.py seeds from
            # training.seed, so without this every repeat would refit identical crops.
            run "${PYTHON[@]}" scripts/probe_segmentation.py \
                --checkpoint "$dir" \
                --config "$CONFIG" \
                --output "$out" \
                "dataloader.normalization=$NORMALIZATION" \
                ${repeat:+"training.seed=$repeat"} \
                ${EXTRA:+$EXTRA}
        done
    done
done

[ "$found" -gt 0 ] || { echo "no $CELLS cells under $PRETRAIN" >&2; exit 1; }
