#!/usr/bin/env bash
# Label-efficiency curve over every cell of one pretraining matrix, mirroring
# scripts/probe_matrix.sh.
#
#   ./scripts/label_efficiency_matrix.sh
#   SEEDS="1 2 3" ./scripts/label_efficiency_matrix.sh
#   CORPUS=amass_clean ARCH=small EPOCHS=40 DRY=1 ./scripts/label_efficiency_matrix.sh
#
# Same knobs as the probe, geometry and kNN sweeps, so the same command line names the
# same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>                     read
#   runs/representation/label_efficiency/<corpus>/<arch>_<epochs>ep<objective>/... written
#
# Simpler than knn_matrix.sh because scripts/label_efficiency.py already takes a run
# directory for --checkpoint, an --output, and dotlist overrides -- so the normalization
# is passed on the command line the way probe_matrix.sh passes it, and none of that
# script's config-patching glue is needed here.
#
# Normalization is taken from the *corpus* config for the same reason probe_matrix.sh does
# it: the backbone is frozen, so windows normalized by other statistics offset every
# feature silently and without error.
#
# The moments floor is measured once per matrix, not per cell: it reads no checkpoint. On
# this task the floor is the whole point -- moments reach 94% of a pretrained probe under a
# linear head at full supervision, and a pretrained curve that is merely *parallel* to the
# moments curve has shown separability rather than label efficiency.
#
# No REPEATS knob, unlike probe_matrix.sh. The head is convex and refit to convergence by
# L-BFGS, and the sample subsampling is seeded, so re-running a cell reproduces its number.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# Which pretraining objective's tree to read. Named rather than sniffed, same as
# probe_matrix.sh, and carrying the same asymmetry: empty suffix for MAE.
OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

# Which pretraining replicates to read, space separated. Empty means the canonical run.
SEEDS=${SEEDS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi

# Extra OmegaConf dotlist overrides, appended verbatim to every call, same as
# finetune_matrix.sh. Deliberately not validated here: OmegaConf already rejects a key the
# config has no slot for, and re-listing the valid ones would be a second place to forget.
EXTRA=${EXTRA:-}

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT=${OUT:-runs/representation/label_efficiency/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
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

done_already() {
    [ -e "$1/metrics.json" ] && { echo "skip   $1"; return 0; }
    return 1
}

echo "corpus     $CORPUS  ($DATALOADER)"
echo "normalize  $NORMALIZATION"
echo "arch       $ARCH"
echo "objective  $OBJECTIVE"
echo "epochs     $EPOCHS"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
echo "overrides  ${EXTRA:-<none>}"
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
    out_root="$OUT${seed:+/seed$seed}"
    [ -n "$seed" ] && echo "== seed $seed -> $out_root"

    # `in_*`, not `in_*__loss_*`: matches both naming schemes, since a cell without a loss
    # axis is named for its input channels alone.
    for dir in "$source_root"/in_*; do
        [ -d "$dir" ] || continue
        found=$((found + 1))
        name=$(basename "$dir")
        out="$out_root/$name"

        # The pretraining `done` marker, not just the directory -- a cell still training
        # would be scored at whatever epoch it has reached and then sit in the table as a
        # replicate. Loud, because a silent skip is how that went unnoticed before.
        if [ ! -e "$dir/done" ]; then
            echo "UNFINISHED  $dir -- no done marker, skipping"
            continue
        fi
        done_already "$out" && continue

        echo "label_efficiency  ${seed:+seed $seed }$name"
        run "${PYTHON[@]}" scripts/label_efficiency.py \
            --checkpoint "$dir" \
            --config "$CONFIG" \
            --output "$out" \
            "dataloader.normalization=$NORMALIZATION" \
            $EXTRA
    done
done

[ "$found" -gt 0 ] || { echo "no in_* cells under $PRETRAIN" >&2; exit 1; }

# The floor. Last, so a matrix that fails above fails before spending anything on it, and
# outside the seed loop because it reads no checkpoint.
if ! done_already "$OUT/moments"; then
    echo
    echo "== moments floor -> $OUT/moments"
    run "${PYTHON[@]}" scripts/label_efficiency.py \
        --moments \
        --config "$CONFIG" \
        --output "$OUT/moments" \
        "dataloader.normalization=$NORMALIZATION" \
        $EXTRA
fi
