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
#   ./scripts/ablate.sh channels   score the loss on pose only, vs every channel
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
#   PYTHON=python ./scripts/ablate.sh      bypass uv
#
#   CONFIG_PROBE=config/experiment_linear_probe_attentive.yaml PROBE_TAG=_attentive \
#     EPOCHS=10 ./scripts/ablate.sh probe    the same backbones under a different head

set -euo pipefail

# Run from the repository root whatever the caller's directory, so the relative config
# paths below mean the same thing every time.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# uv owns this project's environment, so `uv run` syncs it and picks the interpreter --
# no activated venv required, and an out-of-date lockfile fixes itself rather than
# silently running against stale dependencies.
PYTHON=(${PYTHON:-uv run python})

command -v "${PYTHON[0]}" >/dev/null 2>&1 || {
    echo "${PYTHON[0]} not found; install uv or set PYTHON=/path/to/python" >&2
    exit 1
}

CONFIG_PRETRAIN=${CONFIG_PRETRAIN:-config/experiment_mae.yaml}
CONFIG_PROBE=${CONFIG_PROBE:-config/experiment_linear_probe.yaml}
RUNS=${RUNS:-runs/ablation}
METRIC=${METRIC:-val/macro_map}
# Probes of the same backbones under a different head go in a sibling directory, so both
# survive in one tree and stage_report reads them side by side.
PROBE_TAG=${PROBE_TAG:-}
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
    run "${PYTHON[@]}" -m sometria.train --config "$CONFIG_PRETRAIN" --output "$out" \
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
    local out="$RUNS/probe$PROBE_TAG/$name"
    done_already "$out" && return 0

    echo "probe     $name"
    run "${PYTHON[@]}" -m sometria.train --config "$CONFIG_PROBE" --output "$out" \
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

stage_channels() {
    # Measured at 10 epochs: pose 0.1488, nodyn 0.2343, all five 0.2410. Adding tau to
    # pose is worth +0.086 -- thirteen times the seed floor -- and adding vel and acc on
    # top of that is worth +0.007, which is the floor. So the loss does not need the
    # dynamics channels, but it very much needs torque.
    #
    # tau alone came in at 0.1323 -- below pose, so torque is not the objective on its
    # own. What the arms actually track is how many independent physical quantities the
    # loss scores: sin and cos are one angle twice, vel and acc are its derivatives, and
    # tau is the only channel that is not a function of the trajectory. One quantity
    # scores ~0.14, two score ~0.24.
    #
    # But score is also monotone in head width (8, 16, 24, 40 -> 0.132, 0.149, 0.234,
    # 0.241), so breadth of supervision explains the same numbers. sin_tau separates
    # them: it is pose's width carrying nodyn's two quantities, so it scores like pose
    # if width is what matters and like nodyn if independence is.
    pretrain "channels_pose" "model.loss_channels=[0,1]"
    pretrain "channels_nodyn" "model.loss_channels=[0,1,4]"
    pretrain "channels_tau" "model.loss_channels=[4]"
    pretrain "channels_sin_tau" "model.loss_channels=[0,4]"
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
        "${PYTHON[@]}" scripts/results.py "$RUNS/pretrain" "$column" || true
        echo
    done
    for probes in "$RUNS"/probe*/; do
        [ -d "$probes" ] || continue
        echo "=== $(basename "$probes") ($METRIC) ==="
        "${PYTHON[@]}" scripts/results.py "$probes" "$METRIC" || true
        echo
    done
}

case "${1:-all}" in
    noise)   stage_noise ;;
    signal)  stage_signal ;;
    mask)    stage_mask ;;
    decoder)  stage_decoder ;;
    channels) stage_channels ;;
    probe)   stage_probe ;;
    report)  stage_report ;;
    all)     stage_noise; stage_signal; stage_mask; stage_decoder; stage_channels
             stage_probe; stage_report ;;
    *)       sed -n '2,23p' "$0" >&2; exit 2 ;;
esac
