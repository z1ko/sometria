#!/usr/bin/env bash
#
# The MAE ablation, in the order that answers something.
#
#   ./scripts/ablate.sh noise      two seeds, same config -- the floor every later
#                                  comparison has to clear to mean anything
#   ./scripts/ablate.sh signal     probe a random backbone against a pretrained one:
#                                  is pretraining doing anything at all
#   ./scripts/ablate.sh mask       mask_ratio 0.90 / 0.95 / 0.98
#   ./scripts/ablate.sh decoder    decoder_depth 1 / 3 / 6
#   ./scripts/ablate.sh probe      probe every pretraining run that has no probe yet
#   ./scripts/ablate.sh report     read the tables back
#   ./scripts/ablate.sh all        noise, signal, mask, decoder, probe, report
#
# Every run is skipped if its output directory already holds a config.yaml, so an
# interrupted sweep resumes by rerunning the same command. Delete a run directory to
# force it.
#
#   EPOCHS=10 ./scripts/ablate.sh mask     short runs while ablating
#   DRY=1 ./scripts/ablate.sh all          print the commands, run nothing
#   RUNS=runs/other ./scripts/ablate.sh    somewhere else
#   PYTHON=... ./scripts/ablate.sh         a different interpreter

set -euo pipefail

# Run from the repository root whatever the caller's directory, so the relative config
# paths below mean the same thing every time.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The project venv rather than whatever `python` happens to be: an IDE terminal often has
# neither the venv activated nor a bare `python` on PATH at all.
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "no interpreter found; set PYTHON=/path/to/python" >&2
        exit 1
    fi
fi

CONFIG_PRETRAIN=${CONFIG_PRETRAIN:-config/experiment_mae.yaml}
CONFIG_PROBE=${CONFIG_PROBE:-config/experiment_linear_probe.yaml}
RUNS=${RUNS:-runs/ablation}
METRIC=${METRIC:-val/macro_map}
EPOCHS=${EPOCHS:-}
DRY=${DRY:-}

run() {
    if [ -n "$DRY" ]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# A run that already wrote its resolved config finished starting, so leave it alone.
done_already() {
    [ -e "$1/config.yaml" ] && { echo "skip  $1"; return 0; }
    return 1
}

epochs_override() {
    [ -n "$EPOCHS" ] && echo "training.epochs=$EPOCHS"
}

pretrain() { # name, overrides...
    local name=$1; shift
    local out="$RUNS/pretrain/$name"
    done_already "$out" && return 0

    echo "pretrain  $name"
    run "$PYTHON" -m sometria.train --config "$CONFIG_PRETRAIN" --output "$out" \
        $(epochs_override) "$@"
}

# The newest version's last.ckpt. `ls -v` so version_10 sorts after version_9.
checkpoint_of() {
    local version
    version=$(ls -vd "$1"/lightning_logs/version_* 2>/dev/null | tail -1) || return 1
    [ -e "$version/checkpoints/last.ckpt" ] || return 1
    echo "$version/checkpoints/last.ckpt"
}

probe() { # name, checkpoint path or the string null
    local name=$1 ckpt=$2
    local out="$RUNS/probe/$name"
    done_already "$out" && return 0

    echo "probe     $name"
    run "$PYTHON" -m sometria.train --config "$CONFIG_PROBE" --output "$out" \
        $(epochs_override) "model.checkpoint=$ckpt"
}

stage_noise() {
    # Same config twice. If the two probe numbers differ by more than an ablation does,
    # the ablation is measuring the seed.
    for seed in 13 14; do
        pretrain "seed_$seed" "training.seed=$seed"
    done
}

stage_signal() {
    # No pretraining at all: the control the whole protocol is read against.
    probe "random_backbone" "null"
}

stage_mask() {
    # Upward, not downward. At 0.90 the reconstruction loss reaches 0.07 against the ~1.0
    # a null prediction scores, and gets there in three epochs: ten percent of the grid is
    # enough to interpolate a joint angle, so the model learns interpolation rather than
    # anything about motion. The question is how much has to be hidden before it cannot.
    for ratio in 0.90 0.95 0.98; do
        pretrain "mask_$ratio" "masking.mask_ratio=$ratio"
    done
}

stage_decoder() {
    for depth in 1 3 6; do
        pretrain "decoder_$depth" "model.decoder_depth=$depth"
    done
}

stage_probe() {
    local ckpt
    for dir in "$RUNS"/pretrain/*/; do
        [ -d "$dir" ] || continue
        if ! ckpt=$(checkpoint_of "$dir"); then
            echo "skip  $(basename "$dir"): no checkpoint yet"
            continue
        fi
        probe "$(basename "$dir")" "$ckpt"
    done
}

stage_report() {
    echo
    echo "=== pretraining ==="
    for column in val/loss val/mse/sin val/mse/tau; do
        "$PYTHON" scripts/results.py "$RUNS/pretrain" "$column" || true
        echo
    done
    echo "=== probes ($METRIC) ==="
    "$PYTHON" scripts/results.py "$RUNS/probe" "$METRIC" || true
}

case "${1:-all}" in
    noise)   stage_noise ;;
    signal)  stage_signal ;;
    mask)    stage_mask ;;
    decoder) stage_decoder ;;
    probe)   stage_probe ;;
    report)  stage_report ;;
    all)     stage_noise; stage_signal; stage_mask; stage_decoder
             stage_probe; stage_report ;;
    *)       sed -n '2,23p' "$0" >&2; exit 2 ;;
esac
