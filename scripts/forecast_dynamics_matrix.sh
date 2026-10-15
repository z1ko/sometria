#!/usr/bin/env bash
# Forecast and infer torque on every cell of one pretraining matrix, mirroring
# scripts/reconstruction_matrix.sh.
#
#   ./scripts/forecast_dynamics_matrix.sh
#   OBJECTIVE=jepa SEEDS="1 2 3 42" ./scripts/forecast_dynamics_matrix.sh
#   MODES=forecast DRY=1 ./scripts/forecast_dynamics_matrix.sh
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>                 read
#   runs/forecast_dynamics/<mode>/<corpus>/<arch>_<epochs>ep<objective>/<cell> written
#
# <mode> is `forecast`, `infer` (tau hidden with zeros), or `infer_hold` (tau hidden by
# repeating the last observed value). infer_hold runs on `in_pkd*` cells only: no other
# encoder reads tau, so for them the two fills are the same run twice. The gap between
# infer and infer_hold on pkd bounds how much of its number is the overwrite artifact --
# see sometria.downstream.forecast_dynamics.
#
# Same knobs as reconstruction_matrix.sh, and DECODER_EPOCHS=40 for the reason given there.
# Normalization comes from the corpus config, also for the reason given there -- doubly so
# here, because the N m/kg metric inverts it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}
MODES=${MODES:-"forecast infer infer_hold"}
DECODER_EPOCHS=${DECODER_EPOCHS:-40}

CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

SEEDS=${SEEDS:-}
REPEATS=${REPEATS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi
if [ -z "$REPEATS" ]; then repeats=(""); else read -ra repeats <<< "$REPEATS"; fi

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT_ROOT=${OUT_ROOT:-runs/forecast_dynamics}
DRY=${DRY:-}

for config in "$CONFIG" "$DATALOADER"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

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
echo "modes      $MODES"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
echo "repeats    ${REPEATS:-<none, decoder seed from config>}"
echo "read       $PRETRAIN"
echo "write      $OUT_ROOT/<mode>/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}"
echo

found=0
for mode in $MODES; do
    case "$mode" in
        forecast)   flags=(--mode forecast) ;;
        infer)      flags=(--mode infer --fill zero) ;;
        infer_hold) flags=(--mode infer --fill hold) ;;
        *) echo "unknown mode: $mode" >&2; exit 1 ;;
    esac

    for seed in "${seeds[@]}"; do
        source_root="$PRETRAIN${seed:+/seed$seed}"
        if [ ! -d "$source_root" ]; then
            echo "no pretraining runs at $source_root -- run scripts/pretrain_matrix.sh first" >&2
            exit 1
        fi

        for repeat in "${repeats[@]}"; do
            out_root="$OUT_ROOT/$mode/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}${seed:+/seed$seed}${repeat:+/rep$repeat}"
            echo "== $mode ${seed:+seed $seed }${repeat:+repeat $repeat }-> $out_root"

            for dir in "$source_root"/in_*; do
                [ -d "$dir" ] || continue
                cell=$(basename "$dir")
                [ "$mode" = infer_hold ] && [[ "$cell" != in_pkd* ]] && continue
                found=$((found + 1))
                out="$out_root/$cell"
                [ -e "$out/metrics.json" ] && { echo "skip   $out"; continue; }

                if [ ! -e "$dir/done" ]; then
                    echo "UNFINISHED  $dir -- no done marker, skipping"
                    continue
                fi
                echo "probe  $cell"
                run "${PYTHON[@]}" scripts/probe_forecast_dynamics.py \
                    --checkpoint "$dir" \
                    --config "$CONFIG" \
                    --output "$out" \
                    "${flags[@]}" \
                    --epochs "$DECODER_EPOCHS" \
                    "dataloader.normalization=$NORMALIZATION" \
                    ${repeat:+"training.seed=$repeat"}
            done
        done
    done
done

[ "$found" -gt 0 ] || { echo "no in_* cells under $PRETRAIN" >&2; exit 1; }

[ -n "$DRY" ] || "${PYTHON[@]}" results/forecast_dynamics/gen.py
