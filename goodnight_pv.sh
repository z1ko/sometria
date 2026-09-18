#!/usr/bin/env bash
# Overnight: fill the new `pv` rung (pose + velocity) across all three objectives on
# amass_clean, probe it, fine-tune it, regenerate both summaries.
#
#   bash goodnight_pv.sh             # everything, ~25h of pretraining
#   DRY=1 bash goodnight_pv.sh       # print the plan and exit
#   MAE_SEEDS=1 bash goodnight_pv.sh # narrow a stage
#
# Why the rung exists: the whole measured loss-axis gain sits in one step, `p`->`pk` at
# +0.0141 (paired p=0.002), and that step adds velocity and acceleration together. `pv`
# splits it. Nothing on disk currently says which of the two bought the gain.
#
# Ordered by what each stage can actually answer, cheapest informative first:
#
#   1. MAE      24 runs, ~9h.   The only arm with an unconfounded loss axis, so the only
#                               one that answers the question.
#   2. JEPA      6 runs, ~2h.   No loss axis, so `in_pv` only splits the input step -- the
#                               axis where velocity is a derivative of pose and already
#                               measures flat. Expect a null; cheap enough to confirm.
#   3. SimMIM   20 runs, ~13h.  Its loss ladder tracks feature rank (1.17/2.03/2.33), not
#                               channel content, so `loss_pv` will land wherever its target
#                               rank lands. Runs last because it cannot corroborate MAE.
#
# Run counts are from `DRY=1` against the tree as it stood, and already net out the cells
# on disk -- including the `in_pk__loss_pv` row run ahead of this script at all four MAE
# seeds. Re-run DRY before launching; the numbers move as cells land.
#
# Fine-tuning is ~5 min a cell against ~23 min (MAE) to ~40 min (SimMIM) for one
# pretraining, so it is a rounding error at the end rather than a stage worth splitting out.
#
# Fails fast on purpose. Every step below skips work already on disk -- pretrain on `done`,
# probe and finetune on `metrics.json` -- so a crashed run is resumed by re-running this
# file rather than needing a cleanup first.
#
# That skip is not a lock. `done` is written after fit returns, so a run still *in flight*
# has no marker and this script will happily start a second training into the same output
# directory. Do not launch it alongside another sweep over the same tree -- at the time of
# writing that means waiting for the JEPA seed3 pretraining to finish.
#
# A DRY plan also undercounts the downstream stages: the probe and finetune passes can only
# see cells that exist, and in a dry run the pretrainings above them never ran.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-runs/goodnight_pv_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
# Everything below lands in the log as well as the terminal: an unattended job that fails
# at 3am is only debuggable if its output outlived the session that started it.
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

# The matrix scripts read DRY themselves; this is for the calls that are not matrix scripts.
maybe() {
    if [ -n "${DRY:-}" ]; then printf '  %q' "$@"; printf '\n'; else "$@"; fi
}

# Seeds per objective, because the pretraining trees do not hold the same ones: MAE and
# JEPA have 1/2/3/42 on amass_clean, SimMIM only 1/2/42. Listing a seed that was never
# pretrained is not an error here -- pretrain_matrix.sh would simply train it, which is a
# far more expensive surprise than a missing column.
MAE_SEEDS=${MAE_SEEDS:-"42"}
JEPA_SEEDS=${JEPA_SEEDS:-"42"}
SIMMIM_SEEDS=${SIMMIM_SEEDS:-"42"}

# Empty means "whatever each objective just pretrained", which is the only default that
# cannot go stale. Pinning this to seed1 while the seed lists above were narrowed to 42 is
# exactly how the first run of this script fine-tuned nothing: it read seed1's tree, found
# only cells that already had a metrics.json, and skipped all of them without failing.
# Set it to override, e.g. FT_SEEDS=1 to fine-tune only the paired seed.
FT_SEEDS=${FT_SEEDS:-}
FT_EXTRA=${FT_EXTRA:-"model.layer_decay=0.5 training.patience=20"}

stage "start"
echo "log: $LOG"
echo "seeds  mae=$MAE_SEEDS  jepa=$JEPA_SEEDS  simmim=$SIMMIM_SEEDS"
echo "finetune seeds  ${FT_SEEDS:-<same as each objective pretrained>}"


stage "pretrain amass_clean medium_100ep (MAE) seeds $MAE_SEEDS"
SEEDS="$MAE_SEEDS" ./scripts/pretrain_matrix.sh

stage "probe amass_clean medium_100ep (MAE) seeds $MAE_SEEDS"
SEEDS="$MAE_SEEDS" ./scripts/probe_matrix.sh


stage "pretrain amass_clean medium_100ep_jepa seeds $JEPA_SEEDS"
BASE=config/pretrain_jepa.yaml SEEDS="$JEPA_SEEDS" ./scripts/pretrain_matrix.sh

stage "probe amass_clean medium_100ep_jepa seeds $JEPA_SEEDS"
OBJECTIVE=jepa SEEDS="$JEPA_SEEDS" ./scripts/probe_matrix.sh


stage "pretrain amass_clean medium_100ep_simmim seeds $SIMMIM_SEEDS"
BASE=config/pretrain_simmim.yaml SEEDS="$SIMMIM_SEEDS" ./scripts/pretrain_matrix.sh

stage "probe amass_clean medium_100ep_simmim seeds $SIMMIM_SEEDS"
OBJECTIVE=simmim SEEDS="$SIMMIM_SEEDS" ./scripts/probe_matrix.sh


# No CELLS filter: the default `in_*` glob picks up the new rung and skips every cell that
# already has a metrics.json. At ~5 min a cell against ~23 min for one pretraining,
# fine-tuning every seed just pretrained costs a rounding error and is what gives the
# fine-tuning report an error bar -- today every cell there is n=1.
for objective in mae jepa simmim; do
    case $objective in
        mae)    seeds=$MAE_SEEDS ;;
        jepa)   seeds=$JEPA_SEEDS ;;
        simmim) seeds=$SIMMIM_SEEDS ;;
    esac
    seeds=${FT_SEEDS:-$seeds}

    stage "finetune amass_clean $objective seeds $seeds"
    OBJECTIVE="$objective" SEEDS="$seeds" EXTRA="$FT_EXTRA" ./scripts/finetune_matrix.sh
done


stage "summarize"
maybe uv run python results/probe/babel_60_convex/gen.py
maybe uv run python results/finetune/babel_60_finetune/gen.py

stage "done"
echo "read: results/probe/babel_60_convex/README.md     (Where along the ladder the gain appears)"
echo "      results/finetune/babel_60_finetune/README.md"
