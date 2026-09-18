#!/usr/bin/env bash
# Reconstruct and forecast every cell of one pretraining matrix, mirroring scripts/probe_matrix.sh.
#
#   ./scripts/reconstruction_matrix.sh
#   MASKS=suffix ./scripts/reconstruction_matrix.sh
#   SEEDS="1 2 3" DRY=1 ./scripts/reconstruction_matrix.sh
#
# Same knobs as the probe sweep, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>               read
#   runs/reconstruction/<mask>/<corpus>/<arch>_<epochs>ep<objective>/<cell>  written
#
# MASKS is the one addition, and it is the whole difference between phase 3 and phase 4 of
# EVALUATION_PLAN.md: `random` holds out 75% of the tokens uniformly (reconstruction, a
# diagnostic), `suffix` holds out the last 8 time patches across all 43 DOFs (forecasting,
# ~1 s). One rig, one decoder, two masks -- so the mask leads the output path rather than
# sitting at the end of it, and the two tables never collide.
#
# DECODER_EPOCHS is the *decoder's* budget. EPOCHS stays what it is everywhere else: the
# pretraining epoch count, which is a path segment here and not a training knob.
#
# The normalization statistics come from the *corpus* config, exactly as in
# probe_matrix.sh and for the same reason: the backbone is frozen, so windows normalized
# by other statistics are silently offset features with no error anywhere. It matters
# doubly here -- the target is `sin, cos`, which Representation leaves unnormalized, so a
# mismatch would move the encoder's input while leaving the metric's units intact.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}
MASKS=${MASKS:-"random suffix"}
# Decoder budget, not the backbone's. 40 because 10 was measured to be short: on
# `in_pk__loss_pkd` seed1 under the suffix mask, val degrees ran 10.75 (ep 0) -> 5.16
# (ep 10) -> 4.43 (ep 30) and then sat flat at 4.30-4.44 through epoch 50. At 10 epochs
# the probe scores 5.16 against a 5.82 zero-velocity floor and the task looks barely
# alive; converged it scores 4.32, a 26% margin. A short budget measures the decoder's
# optimization, not the representation -- the same censoring that bit the weight-decay
# grid and the kNN k sweep. Reported from the final epoch, never the best: selecting the
# epoch on val degrees would be selecting on the reported metric.
DECODER_EPOCHS=${DECODER_EPOCHS:-40}

CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# Which pretraining objective's tree to read. Named rather than sniffed, same as
# probe_matrix.sh, and carrying the same asymmetry: empty suffix for MAE so paths already
# on disk keep their names.
OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

# SEED selects which pretraining replicate to read and mirrors into the output path, so a
# number is always filed under the backbone it scored. REPEAT re-runs one backbone: the
# decoder is randomly initialized and the random mask is redrawn per step, so a repeat
# bounds this rig's own noise the way probe_matrix.sh's repeats bound the probe's.
SEEDS=${SEEDS:-}
REPEATS=${REPEATS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi
if [ -z "$REPEATS" ]; then repeats=(""); else read -ra repeats <<< "$REPEATS"; fi

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT_ROOT=${OUT_ROOT:-runs/reconstruction}
DRY=${DRY:-}

for config in "$CONFIG" "$DATALOADER"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

# One source of truth: whatever the corpus config says it normalized with is what this
# applies. Reading it back beats restating the path, which would be a second place to
# forget when a corpus is added.
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

echo "corpus     $CORPUS  ($DATALOADER)"
echo "normalize  $NORMALIZATION"
echo "arch       $ARCH"
echo "objective  $OBJECTIVE"
echo "epochs     $EPOCHS pretraining / $DECODER_EPOCHS decoder"
echo "masks      $MASKS"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
echo "repeats    ${REPEATS:-<none, decoder seed from config>}"
echo "read       $PRETRAIN"
echo "write      $OUT_ROOT/<mask>/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}"
echo

found=0
for mask in $MASKS; do
    for seed in "${seeds[@]}"; do
        source_root="$PRETRAIN${seed:+/seed$seed}"
        if [ ! -d "$source_root" ]; then
            echo "no pretraining runs at $source_root -- run scripts/pretrain_matrix.sh first" >&2
            exit 1
        fi

        for repeat in "${repeats[@]}"; do
            out_root="$OUT_ROOT/$mask/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}${seed:+/seed$seed}${repeat:+/rep$repeat}"
            echo "== $mask ${seed:+seed $seed }${repeat:+repeat $repeat }-> $out_root"

            # `in_*`, not `in_*__loss_*`: it matches both naming schemes, since a cell
            # without a loss axis is named for its input channels alone.
            for dir in "$source_root"/in_*; do
                [ -d "$dir" ] || continue
                found=$((found + 1))
                out="$out_root/$(basename "$dir")"
                [ -e "$out/metrics.json" ] && { echo "skip   $out"; continue; }

                # The pretraining `done` marker, not just the directory. Without it a cell
                # still training is read at whatever checkpoint it has reached -- probe_matrix.sh
                # has the war story; loud, because a silent skip is how it went unnoticed.
                if [ ! -e "$dir/done" ]; then
                    echo "UNFINISHED  $dir -- no done marker, skipping"
                    continue
                fi
                echo "decode $(basename "$dir")"
                # A repeat index doubles as the seed: without it every repeat would draw
                # the same masks and initialize the same decoder, and measure nothing.
                run "${PYTHON[@]}" scripts/probe_reconstruction.py \
                    --checkpoint "$dir" \
                    --config "$CONFIG" \
                    --output "$out" \
                    --mask "$mask" \
                    --epochs "$DECODER_EPOCHS" \
                    "dataloader.normalization=$NORMALIZATION" \
                    ${repeat:+"training.seed=$repeat"}
            done
        done
    done
done

[ "$found" -gt 0 ] || { echo "no in_* cells under $PRETRAIN" >&2; exit 1; }

[ -n "$DRY" ] || "${PYTHON[@]}" results/reconstruction/gen.py
