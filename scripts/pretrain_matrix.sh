#!/usr/bin/env bash
# Pretraining sweep over encoder input channels, for one (corpus, architecture, epoch
# budget, objective) point.
#
# 3x3 for MAE -- input channels x reconstruction/loss channels. 3x1 for JEPA, which scores
# an EMA teacher's embedding and so has no reconstruction target to vary. The shape is read
# off the base config rather than switched by hand; see `loss_names` below.
#
#   ./scripts/pretrain_matrix.sh
#   CORPUS=amass_motionx_complete ARCH=medium EPOCHS=100 ./scripts/pretrain_matrix.sh
#   ARCH=tiny EPOCHS=2 DRY=1 ./scripts/pretrain_matrix.sh
#
# The output path is derived, never typed:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep/in_<x>__loss_<y>          MAE
#   runs/pretrain/<corpus>/<arch>_<epochs>ep_<objective>/in_<x>        anything else
#
# Every dimension gets a slot and none is optional. That is the whole point: the old flat
# names marked MotionX with a `_with_motionx` suffix and marked AMASS-only with nothing, so
# a directory could silently mean "whatever the default corpus was that week".
#
# CORPUS is simultaneously the dataloader config name and the path segment, so there is one
# string to change and no mapping table to keep in sync:
#
#   CORPUS=amass_motionx_complete
#     -> config/dataloader/amass_motionx_complete.yaml
#     -> runs/pretrain/amass_motionx_complete/...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=(${PYTHON:-uv run python})

CORPUS=${CORPUS:-amass_motionx_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}

# Merged left to right by train.py, so later files win: the base carries model and training
# defaults, the dataloader config decides which corpus and which normalization statistics,
# and the size config overrides the model dimensions only.
BASE=${BASE:-config/pretrain_mae.yaml}
DATALOADER=${DATALOADER:-config/dataloader/${CORPUS}.yaml}

# The objective is a dimension like any other and gets a path slot -- but only when it is
# not MAE. The asymmetry is deliberate and is the one place this script breaks its own
# rule: every MAE run already on disk lives at the unsuffixed path and the `done` marker is
# keyed on that path, so adding a slot unconditionally would orphan every finished run and
# re-run the lot. New objectives pay for the slot; the incumbent keeps its directories.
# Read from the config rather than from the filename because `model.name` is what train.py
# actually dispatches on.
OBJECTIVE=$(sed -n 's/^[[:space:]]*name:[[:space:]]*\([a-z0-9_]*\).*/\1/p' "$BASE" | head -1)
OBJECTIVE=${OBJECTIVE:-mae}
SUFFIX=""
[ "$OBJECTIVE" != "mae" ] && SUFFIX="_$OBJECTIVE"

# The size overlay prefers the objective's own tree, and falls back to the MAE one when
# that arch has no per-objective file.
#
# The split exists for SimMIM alone: every `config/mae/*.yaml` sets `dec_depth`, and SimMIM
# has no decoder to size and rejects it. JEPA takes `dec_depth` happily, so it has no reason
# to need a full parallel tree -- and without the fallback, `ARCH=tiny` under JEPA would
# stop resolving the moment `config/jepa/` gained a single file, which is a silly way to
# break six working arches.
#
# The fallback only applies to the default. An explicitly passed SIZE is used as given and
# fails the existence check below if it is wrong, rather than being quietly swapped for
# something else. A SimMIM arch with no overlay of its own lands on the MAE file and dies at
# construction with "unexpected keyword argument 'dec_depth'", which names its own fix.
if [ -z "${SIZE:-}" ]; then
    SIZE=config/${OBJECTIVE}/${ARCH}.yaml
    [ -f "$SIZE" ] || SIZE=config/mae/${ARCH}.yaml
fi

# Replicates. SEEDS is a space-separated list, not a count, because a count can only ever
# mean "1..N from scratch" -- a list also expresses "add seed 4 to the three I already
# have", which is what actually happens once a first pass looks marginal.
#
#   ./scripts/pretrain_matrix.sh                  the canonical run, unsuffixed path
#   SEEDS=7 ./scripts/pretrain_matrix.sh          one replicate  -> .../seed7/<cell>
#   SEEDS="1 2 3" ./scripts/pretrain_matrix.sh    three          -> .../seed1|2|3/<cell>
#   SEEDS=$(seq 5) ./scripts/pretrain_matrix.sh   if you did want a count
#
# Empty means the canonical run, at whatever seed the base config sets, in the unsuffixed
# path. That is a named reference point rather than the invisible default this layout
# otherwise avoids, and it keeps every existing run where it is. An explicit seed
# overrides training.seed *and* adds a path segment, so a replicate can never quietly
# overwrite the canonical run.
#
# Pretraining-seed spread is the largest unmeasured source of variance in the matrix:
# every cell is a single run today, so a corpus-to-corpus delta of a few thousandths has
# nothing to be compared against. Nine runs per seed for MAE, three for JEPA, so budget
# accordingly.

# Which cells to run. Empty means all of them; a space-separated list of cell names runs
# only those, which is what an ablation aimed at one row wants rather than eight runs it
# will not read. The names are whatever this objective produces -- `in_<x>__loss_<y>` when
# there is a loss axis, plain `in_<x>` when there is not.
#
#   CELLS="in_pk__loss_pk in_pk__loss_pkd" ./scripts/pretrain_matrix.sh
#   BASE=config/pretrain_jepa.yaml CELLS="in_pk" ./scripts/pretrain_matrix.sh
CELLS=${CELLS:-}

SEEDS=${SEEDS:-}
if [ -z "$SEEDS" ]; then
    seeds=("")
else
    read -ra seeds <<< "$SEEDS"
fi

# The base path. The seed segment is appended per replicate below, so overriding RUNS
# still nests replicates underneath it rather than collapsing them onto each other.
RUNS=${RUNS:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
DRY=${DRY:-}

for config in "$BASE" "$DATALOADER" "$SIZE"; do
    [ -f "$config" ] && continue
    echo "missing config: $config" >&2
    exit 1
done

names=(p pk pkd)
channels=("[0,1]" "[0,1,2,3]" "[0,1,2,3,4]")

# Whether this objective has a loss axis at all. MAE reconstructs channels, so input x loss
# is a real 3x3. JEPA's target is an embedding and its constructor takes no
# `channels_output`, so the same sweep is three cells -- and passing the override anyway is
# not merely wasteful, it is a TypeError on the first cell. Detected from the base config
# because BASE already decides the objective, and a second switch saying the same thing is
# a second thing to get out of sync.
# Anchored to a real YAML key, not the bare word: `config/pretrain_jepa.yaml` explains in a
# comment that it *has* no channels_output, and a loose grep matches that comment and
# cheerfully rebuilds the nine-cell sweep it was meant to collapse.
if grep -qE '^[[:space:]]*channels_output[[:space:]]*:' "$BASE"; then
    loss_names=("${names[@]}")
else
    loss_names=("")
fi

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

done_already() {
    # The marker train.py writes *after* fit returns, not config.yaml, which it writes
    # before training starts. Keying on the config meant a run that crashed at startup
    # left a directory that looked finished and was skipped on every retry -- and a run
    # killed mid-training was skipped at whatever epoch it reached.
    [ -e "$1/done" ] && { echo "skip   $1"; return 0; }
    return 1
}

# Print the resolved plan before doing anything. Reading it back off the paths afterwards
# is what the old layout made hard, and a dry run should show the same lines as a real one.
echo "corpus     $CORPUS  ($DATALOADER)"
echo "arch       $ARCH  ($SIZE)"
echo "epochs     $EPOCHS"
echo "objective  $OBJECTIVE  ($BASE)"
echo "cells      ${CELLS:-<all $(( ${#names[@]} * ${#loss_names[@]} ))>}"
echo "seeds      ${SEEDS:-<base config default, canonical run>}"
echo "output     $RUNS"
echo

for seed in "${seeds[@]}"; do
    root="$RUNS${seed:+/seed$seed}"
    [ -n "$seed" ] && echo "== seed $seed -> $root"

    for i in "${!names[@]}"; do
        for j in "${!loss_names[@]}"; do
            if [ -n "${loss_names[$j]}" ]; then
                name="in_${names[$i]}__loss_${loss_names[$j]}"
            else
                name="in_${names[$i]}"
            fi
            if [ -n "$CELLS" ] && [[ " $CELLS " != *" $name "* ]]; then continue; fi
            out="$root/$name"
            done_already "$out" && continue

            echo "train  $name"
            run "${PYTHON[@]}" train.py \
                --config "$BASE" "$DATALOADER" "$SIZE" \
                --output "$out" \
                "training.epochs=$EPOCHS" \
                ${seed:+"training.seed=$seed"} \
                "model.channels_input=${channels[$i]}" \
                ${loss_names[$j]:+"model.channels_output=${channels[$j]}"}
        done
    done
done
