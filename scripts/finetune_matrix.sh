#!/usr/bin/env bash
# Fine-tune every cell of one pretraining matrix. Same knobs as scripts/probe_matrix.sh,
# which this mirrors line for line -- what differs is the script it calls, where the results
# land, and CELLS, because a fine-tune answers a different question at a very different price.
#
#   ./scripts/finetune_matrix.sh
#   OBJECTIVE=simmim SEEDS="42 1 2" ./scripts/finetune_matrix.sh
#   CELLS="in_pk__loss_pk" SEEDS="42 1 2" DRY=1 ./scripts/finetune_matrix.sh
#
# Read the cost before launching. A convex probe extracts features once and fits a head;
# this trains the whole backbone for `training.epochs`. The full grid is 9 cells x 3 seeds x
# 4 objectives = 108 runs. CELLS is why it exists here and not in probe_matrix.sh: one
# pre-specified cell across the four objectives at three seeds is 12 runs, and that is the
# ranking question the frozen probe could not answer.
#
# Same four knobs as the pretraining sweep, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>          read
#   runs/finetune/<benchmark>/<corpus>/<arch>_<epochs>ep<objective>/<cell> written
#
# OBJECTIVE mirrors the slot pretrain_matrix.sh writes, and carries the same asymmetry:
# empty for MAE so the paths match the probe tree cell for cell, `_jepa` and so on otherwise.
# A cell is `in_<x>__loss_<y>` where the objective has a loss axis and plain `in_<x>` where
# it does not, so the glob below matches on the part they share.
#
# The normalization statistics are taken from the *corpus* config, not from the probe
# config, and this is the whole reason the corpus is a knob here rather than just a path
# segment. The backbone is frozen: feed it windows normalized by statistics other than the
# ones it pretrained under and every feature is offset, silently and without error.
# config/experiment_finetune.yaml pins pretrain_v1 statistics, which are wrong for any
# checkpoint pretrained on anything else.
#
# BENCHMARK names the output directory only. A second benchmark needs its own finetune config
# (different label_set and splits), so pass both:
#
#   BENCHMARK=carepd_updrs_finetune CONFIG=config/experiment_probe_carepd.yaml ...
#
# SPLIT_SET overrides both sides of the config's split and adds itself to the output path.
# For CARE-PD the split set *is* the evaluation protocol -- within-site, cross-site, LODO
# and MIDA differ in nothing else -- so this is how a protocol gets run:
#
#   SPLIT_SET=carepd_lodo_BMCLab BENCHMARK=carepd_updrs_finetune \
#     CONFIG=config/experiment_probe_carepd.yaml ./scripts/finetune_matrix.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

BENCHMARK=${BENCHMARK:-babel_60_finetune}
CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

CONFIG=${CONFIG:-config/experiment_finetune.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# Which pretraining objective's tree to read. Named rather than sniffed, because unlike
# pretrain_matrix.sh there is no base config here to read `model.name` off -- this script
# only ever sees the probe config and the corpus.
#
#   OBJECTIVE=jepa ./scripts/finetune_matrix.sh
OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

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
#   SEED=7 ./scripts/finetune_matrix.sh              reads/writes .../seed7/<cell>
#   REPEAT=2 ./scripts/finetune_matrix.sh            writes .../rep2/<cell>
# Both are space-separated lists, same shape as SEEDS in pretrain_matrix.sh, and they
# nest: every REPEAT is run against every SEED.
#
#   SEEDS="1 2 3" ./scripts/finetune_matrix.sh     fine-tune three pretraining replicates
#   REPEATS="1 2 3" ./scripts/finetune_matrix.sh   re-run the canonical backbone three times
SEEDS=${SEEDS:-}
REPEATS=${REPEATS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi
if [ -z "$REPEATS" ]; then repeats=(""); else read -ra repeats <<< "$REPEATS"; fi

# Base paths; seed and repeat segments are appended per combination below.
SPLIT_SET=${SPLIT_SET:-}
# Which cells to run, as a glob against the pretraining tree. `in_*` is everything and
# matches both naming schemes, since an objective with no loss axis names its cells for
# their input channels alone. Narrow it rather than the seed list when scoping: seeds are
# the replicate axis every claim here is measured against.
CELLS=${CELLS:-in_*}
# Extra OmegaConf dotlist overrides, appended verbatim to every call. A sweep over one
# config knob is otherwise a copy of this whole script with one line changed.
#
#   EXTRA="model.dropout=0.1 model.layer_decay=0.5" ./scripts/finetune_matrix.sh
#
# Deliberately not validated here: OmegaConf already rejects a key the config has no slot
# for, and re-listing the valid ones would be a second place to forget a new knob.
EXTRA=${EXTRA:-}
PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
# The split set joins the path rather than only the config, so two protocols scored from
# the same checkpoint do not overwrite each other's metrics.json.
OUT=${OUT:-runs/finetune/${BENCHMARK}${SPLIT_SET:+/$SPLIT_SET}/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
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
echo "objective  $OBJECTIVE"
echo "epochs     $EPOCHS"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
echo "repeats    ${REPEATS:-<none, run seed from config>}"
echo "split set  ${SPLIT_SET:-<from $CONFIG>}"
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

    for repeat in "${repeats[@]}"; do
        out_root="$OUT${seed:+/seed$seed}${repeat:+/rep$repeat}"
        [ -n "$seed$repeat" ] && echo "== ${seed:+seed $seed }${repeat:+repeat $repeat }-> $out_root"

        for dir in $(compgen -G "$source_root/$CELLS" || true); do
            [ -d "$dir" ] || continue
            found=$((found + 1))
            name=$(basename "$dir")
            out="$out_root/$name"
            done_already "$out" && continue

            echo "finetune  $name"
            # A repeat index doubles as the run seed. finetune_baseline_mae.py seeds from
            # training.seed, so without this every repeat would draw identical crops and an
            # identical head initialization, and measure nothing at all.
            run "${PYTHON[@]}" scripts/finetune_baseline_mae.py \
                --checkpoint "$dir" \
                --config "$CONFIG" \
                --output "$out" \
                "dataloader.normalization=$NORMALIZATION" \
                ${SPLIT_SET:+"dataloader.train.split_set=$SPLIT_SET"} \
                ${SPLIT_SET:+"dataloader.val.split_set=$SPLIT_SET"} \
                ${repeat:+"training.seed=$repeat"} \
                $EXTRA
        done
    done
done

[ "$found" -gt 0 ] || { echo "no cell matching $CELLS under $PRETRAIN" >&2; exit 1; }
