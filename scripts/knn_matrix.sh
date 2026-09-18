#!/usr/bin/env bash
# kNN retrieval over every cell of one pretraining matrix, mirroring scripts/probe_matrix.sh.
#
#   ./scripts/knn_matrix.sh
#   SEEDS="1 2 3" ./scripts/knn_matrix.sh
#   CORPUS=amass_clean ARCH=small EPOCHS=40 DRY=1 ./scripts/knn_matrix.sh
#
# Same knobs as the probe and geometry sweeps, so the same command line names the same runs:
#
#   runs/pretrain/<corpus>/<arch>_<epochs>ep<objective>/<cell>         read
#   runs/representation/knn/<corpus>/<arch>_<epochs>ep<objective>/...  written
#
# scripts/knn_probe.py is complete and is not touched here -- cosine similarity, softmax
# temperature weighting, the k sweep and the metric set are all decided there. This script
# is the matrix loop it never had, and the two pieces of glue that loop needs:
#
#   1. knn_probe.py takes a checkpoint *file*, while the matrix stores run directories, so
#      the checkpoint is resolved with probe_convex_mae.latest_checkpoint rather than a
#      second globbing rule that could disagree with the probe about which epoch is "the"
#      checkpoint.
#   2. it takes no dotlist overrides, so the corpus normalization cannot be passed on the
#      command line the way probe_matrix.sh passes it. A patched copy of the config is
#      written per cell instead -- which is also what every other script here leaves behind
#      as its record of what it ran.
#
# Normalization is taken from the *corpus* config for the same reason probe_matrix.sh does
# it: the backbone is frozen, so windows normalized by other statistics offset every feature
# silently and without error. Here that lands as a neighbour set from a distribution the
# encoder never saw.
#
# The moments floor is measured once per matrix, not per cell: it reads no checkpoint. It is
# run unconditionally because the standing rule is that every task publishes its floor
# beside it, and on this task the floor is the whole point -- moments reach 94% of a
# pretrained probe under a linear head, and whether they also reach it under a distance is
# the question kNN was added to answer.
#
# No REPEATS knob, unlike probe_matrix.sh: nothing is fit, so a re-run of a cell reproduces
# its number. The training crop is drawn once (knn_probe.py seeds it) and the validation
# windows are deterministic tiles.

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

# Which submodule of a checkpoint holds the weights. JEPA has no `backbone.` at all and
# needs ENCODER=teacher; knn_probe.py refuses to guess, and neither does this.
ENCODER=${ENCODER:-backbone}

# Which pretraining replicates to read, space separated. Empty means the canonical run.
SEEDS=${SEEDS:-}
if [ -z "$SEEDS" ]; then seeds=(""); else read -ra seeds <<< "$SEEDS"; fi

PRETRAIN=${PRETRAIN:-runs/pretrain/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
OUT=${OUT:-runs/representation/knn/${CORPUS}/${ARCH}_${EPOCHS}ep${SUFFIX}}
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

done_already() {
    [ -e "$1/metrics.json" ] && { echo "skip   $1"; return 0; }
    return 1
}

# Writes <out>/config.yaml with the corpus normalization patched in, and prints the
# checkpoint file for a run directory (empty argument: the moments floor, no checkpoint).
prepare() {
    "${PYTHON[@]}" - "$1" "$2" "$CONFIG" "$NORMALIZATION" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, "scripts")
from probe_convex_mae import latest_checkpoint

run, out, config_path, normalization = sys.argv[1:]
config = OmegaConf.load(config_path)
OmegaConf.update(config, "dataloader.normalization", normalization)
OmegaConf.save(config, Path(out) / "config.yaml")
if run:
    print(latest_checkpoint(Path(run)))
PY
}

# knn_probe.py prints a table and writes nothing. Rather than teach it an --output (it is
# finished, and used by hand), its stdout is kept verbatim as knn.txt and reduced here to
# the same metrics.json envelope every other runner leaves, so the skip check above and
# results/representation/gen.py need no special case for this task.
to_metrics() {
    "${PYTHON[@]}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

log, out = Path(sys.argv[1]), Path(sys.argv[2])
names, rows, source = None, {}, None
for line in log.read_text().splitlines():
    fields = line.split()
    if not fields:
        continue
    if fields[0] == "source":
        names = fields[1:]
    elif names and len(fields) > 2 and fields[1].startswith("k="):
        source = fields[0]
        rows[int(fields[1][2:])] = dict(zip(names, (float(value) for value in fields[2:])))

if not rows:
    raise SystemExit(f"no kNN table in {log}")

# Headline is the best k by macro mAP, with the whole sweep kept beside it -- the same
# shape probe_convex_mae.py reports its weight-decay sweep in, and read the same way: the
# sweep is nearly free because the distances are computed once, so reporting only the
# argmax would throw away the only evidence that k mattered at all.
best = max(rows, key=lambda k: rows[k]["macro_map"])
out.write_text(json.dumps({"metrics": rows[best], "k": best, "sweep": rows, "source": source}, indent=2))
PY
}

knn() {   # run dir (empty for the floor), output dir
    local source="$1" out="$2"
    if [ -n "$DRY" ]; then
        echo "  knn ${source:-moments} -> $out"
        return
    fi
    mkdir -p "$out"
    if [ -n "$source" ]; then
        source=$(prepare "$source" "$out")
    else
        prepare "" "$out"
    fi
    "${PYTHON[@]}" scripts/knn_probe.py \
        --config "$out/config.yaml" \
        --encoder "$ENCODER" \
        --sources "${source:-moments}" | tee "$out/knn.txt"
    to_metrics "$out/knn.txt" "$out/metrics.json"
}

echo "corpus     $CORPUS  ($DATALOADER)"
echo "normalize  $NORMALIZATION"
echo "arch       $ARCH"
echo "objective  $OBJECTIVE  (encoder $ENCODER)"
echo "epochs     $EPOCHS"
echo "seeds      ${SEEDS:-<canonical pretraining run>}"
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

        echo "knn  ${seed:+seed $seed }$name"
        knn "$dir" "$out"
    done
done

[ "$found" -gt 0 ] || { echo "no in_* cells under $PRETRAIN" >&2; exit 1; }

# The floor. Last, so a matrix that fails above fails before spending anything on it, and
# outside the seed loop because it reads no checkpoint.
if ! done_already "$OUT/moments"; then
    echo
    echo "== moments floor -> $OUT/moments"
    knn "" "$OUT/moments"
fi
