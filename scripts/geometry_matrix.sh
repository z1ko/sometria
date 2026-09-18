#!/usr/bin/env bash
# Measure representation geometry on every cell of one pretraining matrix, mirroring
# scripts/probe_matrix.sh.
#
#   ./scripts/geometry_matrix.sh
#   SEEDS="1 2 3" CKA=1 ./scripts/geometry_matrix.sh
#   CORPUS=amass_clean ARCH=small EPOCHS=40 DRY=1 ./scripts/geometry_matrix.sh
#
# Same knobs as the probe sweep, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>              read
#   runs/representation/geometry/<corpus>/<arch>_<epochs>ep<objective>/...  written
#
# The `geometry` segment is there because results/representation/ is planned to hold four
# measurements over one extraction pass (effective rank, CKA, label efficiency, kNN); this
# script owns the first two.
#
# Normalization is taken from the *corpus* config for the same reason probe_matrix.sh does
# it: the backbone is frozen, so windows normalized by other statistics offset every
# feature silently and without error -- and here that lands as a participation ratio for a
# distribution the encoder never saw.
#
# CKA=1 adds a second pass over the cells already written, all-pairs within each seed plus
# each cell across seeds. It needs SEEDS to say anything: with one seed there is no noise
# floor, and a cross-cell CKA without one is a number nobody can read.
#
# No REPEATS knob, unlike probe_matrix.sh. There is nothing to repeat -- evaluation windows
# are deterministic tiles and no head is fit, so re-running a cell reproduces its number.

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

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT=${OUT:-runs/representation/geometry/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
CKA=${CKA:-}
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
echo "cka        ${CKA:+all pairs after the per-cell pass}"
echo "read       $PRETRAIN"
echo "write      $OUT"
echo

found=0
measured=()
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
        # would be measured at whatever epoch it has reached and then sit in the table as a
        # replicate. Loud, because a silent skip is how that went unnoticed before.
        if [ ! -e "$dir/done" ]; then
            echo "UNFINISHED  $dir -- no done marker, skipping"
            continue
        fi
        measured+=("$out")
        done_already "$out" && continue

        echo "geometry  ${seed:+seed $seed }$name"
        run "${PYTHON[@]}" scripts/representation_geometry.py \
            --checkpoint "$dir" \
            --config "$CONFIG" \
            --output "$out" \
            "dataloader.normalization=$NORMALIZATION"
    done
done

[ "$found" -gt 0 ] || { echo "no in_* cells under $PRETRAIN" >&2; exit 1; }

if [ -n "$CKA" ]; then
    echo
    echo "== cka over ${#measured[@]} cells -> $OUT/cka"
    if [ "${#seeds[@]}" -lt 2 ]; then
        echo "one seed only -- cross-cell CKA will have no cross-seed noise floor to read it against" >&2
    fi
    if ! done_already "$OUT/cka"; then
        run "${PYTHON[@]}" scripts/representation_geometry.py \
            --output "$OUT/cka" --cka "${measured[@]}"
    fi
fi
