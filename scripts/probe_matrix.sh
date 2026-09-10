#!/usr/bin/env bash
# Probe every cell of one pretraining matrix, mirroring scripts/pretrain_matrix.sh.
#
#   ./scripts/probe_matrix.sh
#   CORPUS=amass_motionx_complete ./scripts/probe_matrix.sh
#   CORPUS=amass_clean ARCH=small EPOCHS=40 DRY=1 ./scripts/probe_matrix.sh
#
# Same four knobs as the pretraining sweep, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep/<cell>          read
#   runs/probe/<benchmark>/<corpus>/<arch>_<epochs>ep/<cell> written
#
# The normalization statistics are taken from the *corpus* config, not from the probe
# config, and this is the whole reason the corpus is a knob here rather than just a path
# segment. The backbone is frozen: feed it windows normalized by statistics other than the
# ones it pretrained under and every feature is offset, silently and without error.
# config/experiment_linear_probe.yaml pins pretrain_v1 statistics, which are wrong for any
# checkpoint pretrained on anything else.
#
# BENCHMARK names the output directory only. A second benchmark needs its own probe config
# (different label_set and splits), so pass both:
#
#   BENCHMARK=carepd_updrs_convex CONFIG=config/experiment_probe_carepd.yaml ...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

BENCHMARK=${BENCHMARK:-babel_60_convex}
CORPUS=${CORPUS:-amass_motionx_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

CONFIG=${CONFIG:-config/experiment_linear_probe.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# Two independent replicate axes, because they measure different things.
#
# SEED selects which *pretraining* replicate to read, and mirrors into the output path so
# a probe is always filed under the backbone it scored. Empty means the canonical run.
#
# REPEAT re-probes one backbone. The probe is not deterministic: the training split draws
# a random crop per sample per pass, so refitting the same frozen features gives a
# slightly different head. Repeats bound that, and it is the smaller of the two -- the
# pretraining seed is not controlled at all unless SEED is used.
#
#   SEED=7 ./scripts/probe_matrix.sh                 reads/writes .../seed7/<cell>
#   REPEAT=2 ./scripts/probe_matrix.sh               writes .../rep2/<cell>
# Both are space-separated lists, same shape as SEEDS in pretrain_matrix.sh, and they
# nest: every REPEAT is run against every SEED.
#
#   SEEDS="1 2 3" ./scripts/probe_matrix.sh        probe three pretraining replicates
#   REPEATS="1 2 3" ./scripts/probe_matrix.sh      re-probe the canonical run three times
SEEDS=${SEEDS:-}
REPEATS=${REPEATS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi
if [ -z "$REPEATS" ]; then repeats=(""); else read -ra repeats <<< "$REPEATS"; fi

# Base paths; seed and repeat segments are appended per combination below.
PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep}
OUT=${OUT:-runs/probe/${BENCHMARK}/${CORPUS}/${ARCH}_${EPOCHS}ep}
DRY=${DRY:-}

for config in "$CONFIG" "$DATALOADER"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done


# One source of truth: whatever the corpus config says it normalized with is what the
# probe applies. Reading it back beats restating the path here, which would be a second
# place to forget when a corpus is added.
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
echo "epochs     $EPOCHS"
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

        for dir in "$source_root"/in_*__loss_*; do
            [ -d "$dir" ] || continue
            found=$((found + 1))
            name=$(basename "$dir")
            out="$out_root/$name"
            done_already "$out" && continue

            echo "probe  $name"
            # A repeat index doubles as the probe seed. probe_convex_mae.py seeds from
            # training.seed, so without this every repeat would refit identical crops and
            # measure nothing at all.
            run "${PYTHON[@]}" scripts/probe_convex_mae.py \
                --checkpoint "$dir" \
                --config "$CONFIG" \
                --output "$out" \
                "dataloader.normalization=$NORMALIZATION" \
                ${repeat:+"training.seed=$repeat"}
        done
    done
done

[ "$found" -gt 0 ] || { echo "no in_*__loss_* cells under $PRETRAIN" >&2; exit 1; }
