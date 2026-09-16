#!/usr/bin/env bash
# Can regularization move the fine-tune peak, or is 0.385 the representation's ceiling?
#
#   ./scripts/finetune_regularization.sh          # 5 arms
#   DRY=1 ./scripts/finetune_regularization.sh    # print the plan and exit
#   ARMS="drop0.1 decay0.5" ./scripts/finetune_regularization.sh   # two arms only
#
# THE OBSERVATION
#
# `amass_clean/medium_100ep/seed42/in_pk__loss_pk`, fine-tuned with layer-wise decay 0.75
# at base lr 1e-3, BABEL-60 macro mAP by epoch:
#
#   epoch 13   train 0.0729   val 0.0984   macro mAP 0.3877   <- peak
#   epoch 20   train 0.0570   val 0.1047   macro mAP 0.3791
#   epoch 37   train 0.0278   val 0.1292   macro mAP 0.3637
#
# Train loss falls 2.6x past the peak while val loss rises 31% and the gap quadruples.
# That is overfitting and nothing else. 12,994 training windows against ~5M backbone
# parameters, where MAE's 0.75 layer decay was tuned on 1.28M ImageNet images.
#
# THE QUESTION
#
# Layer-wise decay already bought what it was going to buy -- it reached the old flat-rate
# 50-epoch result (0.3855) at epoch 13 and peaked at 0.3877, so ~4x the convergence speed
# and +0.002 on the score. Against a convex mean-pool probe at 0.3633, fine-tuning is worth
# about +0.02. The question this sweep asks is whether that +0.02 is small because the
# optimization overfits, or because the representation has nothing more to give.
#
# THE ARMS
#
# One knob at a time around the baseline, not a grid. This is a direction-of-effect
# question and a 3x3 would be nine runs to answer it once.
#
#   dropout   regularizes the readout and the blocks. The knob already exists on
#             MotionFinetuneClassifier and is unset; it is set on the modules rather than
#             rebuilt from a spec, so a pretrained backbone takes it without reloading.
#   decay     regularizes by *moving the backbone less*. A smaller layer_decay shrinks
#             every rung below the head, so the lower blocks stay near where pretraining
#             put them. This is the knob that targets "the backbone is memorizing"
#             directly, where dropout targets the symptom.
#
# Predicted if overfitting is the binding constraint: the peak rises and arrives later.
# If every arm peaks at ~0.385 whatever it is regularized with, the ceiling is the
# representation, the fine-tune-over-probe gap really is +0.02, and the next lever is the
# pretraining objective rather than the protocol.
#
# WHAT THE FIRST FIVE ARMS FOUND, seed 42
#
#   arm         macro mAP   peak epoch   val-train gap at peak
#   drop0.2        0.3698       15           0.024
#   drop0.1        0.3782       13           0.021
#   baseline       0.3878       13           0.025
#   decay0.65      0.3893       15           0.029
#   decay0.5       0.3942       23           0.040
#
# The two knobs move in opposite directions, and the gap column says why. Dropout produced
# the *smallest* gap and the worst score -- it suppressed the train/val divergence by
# damaging the representation, and its train loss at peak (0.0776) is the highest of any
# arm, so it underfits and still generalizes worse. Layer decay produced the largest gap
# and the best score. The failure was never a readout too flexible for its data; it was a
# backbone walking away from weights that were already good. Anything preserving them wins.
#
# So the decay axis is the live one, it is monotone over three points, and 0.5 sat at the
# edge of the sweep. `decay0.35` and `decay0.25` extend it. There has to be a turning point:
# `layer_decay` -> 0 is a frozen backbone, which is the probe, and the probe scores 0.3633.
#
# THE `meanpool` ARM IS A DIFFERENT QUESTION
#
# Every other arm compares fine-tunes to each other. `meanpool` compares a fine-tune to the
# *probe*, which the rest cannot: the probe pools with `mean` and adds no parameters, while
# these run `attentive`, whose scoring MLP is ~66k learned parameters deciding which tokens
# count. So part of the 0.3633 -> 0.3942 gap is a bigger readout rather than a trainable
# backbone, and nothing measured so far separates the two.
#
# This arm holds the readout at the probe's and keeps the best decay. Its distance from
# 0.3633 is what the trainable backbone is worth on its own; the rest of the gap is the
# pooler. Expect it to land below `decay0.5` -- the point is where between.
#
# EPOCHS IS NOT A KNOB HERE
#
# `training.epochs` sets the cosine horizon, so shortening it changes the learning-rate
# trajectory of every step and is *not* early stopping. Every arm runs the same horizon or
# the comparison is confounded. Shorten it globally with EPOCHS if you must, never per arm.
# The reported number is always the monitored checkpoint's, not the last epoch's, so an arm
# that overfits after its peak loses compute and not score.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=${LOG:-runs/finetune_regularization_$(date +%Y%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "=== $* :: $(date '+%F %T')"; }

CORPUS=${CORPUS:-amass_clean}
ARCH=${ARCH:-medium}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-42}
# One cell. The baseline it is read against was measured on this one, and nine cells would
# be nine unread runs per arm at ~100 epochs of full-backbone training each.
CELLS=${CELLS:-in_pk__loss_pk}
OBJECTIVE=${OBJECTIVE:-mae}
BENCHMARK=${BENCHMARK:-babel_60_reg}
# The finished arms stay in the default list rather than being pruned: every one of them
# has a metrics.json, so they are skipped in seconds, and a bare run of this script still
# describes the whole experiment instead of only its unfinished tail.
ARMS=${ARMS:-"baseline drop0.1 drop0.2 decay0.65 decay0.5 decay0.35 decay0.25 meanpool"}
DRY=${DRY:-}

# The overrides each arm applies, on top of config/experiment_finetune.yaml. `baseline`
# restates the config's own values rather than passing nothing, so the arm is reproduced
# from this file even if the config's defaults move underneath it.
overrides() {
    case "$1" in
        baseline)  echo "model.dropout=0.0 model.layer_decay=0.75" ;;
        drop0.1)   echo "model.dropout=0.1 model.layer_decay=0.75" ;;
        drop0.2)   echo "model.dropout=0.2 model.layer_decay=0.75" ;;
        decay0.65) echo "model.dropout=0.0 model.layer_decay=0.65" ;;
        decay0.5)  echo "model.dropout=0.0 model.layer_decay=0.5"  ;;
        decay0.35) echo "model.dropout=0.0 model.layer_decay=0.35" ;;
        decay0.25) echo "model.dropout=0.0 model.layer_decay=0.25" ;;
        # Not a point on the decay axis. It moves two knobs at once on purpose: the best
        # decay so far, and the *probe's* readout. See the note below.
        meanpool)  echo "model.dropout=0.0 model.layer_decay=0.5 model.pool=mean" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

stage "start"
echo "log       $LOG"
echo "corpus    $CORPUS   arch $ARCH   epochs $EPOCHS   seeds $SEEDS"
echo "cell      $CELLS"
echo "objective $OBJECTIVE"
echo "arms      $ARMS"
for arm in $ARMS; do printf '  %-10s %s\n' "$arm" "$(overrides "$arm")"; done

for arm in $ARMS; do
    stage "finetune $arm"
    # Each arm gets its own benchmark segment, so the runs sit beside each other rather
    # than overwriting one path -- and so the skip marker is per arm.
    BENCHMARK="$BENCHMARK/$arm" CORPUS=$CORPUS ARCH=$ARCH EPOCHS=$EPOCHS SEEDS="$SEEDS" \
        CELLS="$CELLS" OBJECTIVE=$OBJECTIVE DRY=$DRY EXTRA="$(overrides "$arm")" \
        ./scripts/finetune_matrix.sh
done

stage "summarize"
[ -n "$DRY" ] || ARMS="$ARMS" BENCHMARK="$BENCHMARK" uv run python - <<'PY'
import json, os, pathlib, re
import polars as pl

root = pathlib.Path("runs/finetune")
rows = []
for arm in os.environ["ARMS"].split():
    # Globbed under the arm rather than rebuilt from corpus/arch/epochs: the objective
    # puts a suffix on the arch segment, so a reconstructed path finds nothing whenever
    # OBJECTIVE is not mae. The arm directory is already unique, so there is nothing to
    # disambiguate below it.
    for metrics in sorted((root / os.environ["BENCHMARK"] / arm).rglob("metrics.json")):
        run = metrics.parent
        best = json.loads(metrics.read_text())["metrics"].get("macro_map")

        # The peak's *epoch* is the other half of the answer: regularization that works
        # moves the peak later as well as higher, and a peak still at the end of the run
        # means the horizon, not the regularizer, is what stopped it.
        #
        # Read off the monitored checkpoint's filename, not by sorting the curve. The
        # closing `trainer.validate` pass logs the best checkpoint's scores back into the
        # same CSV under the *last* epoch, so a sort puts every peak at the horizon and the
        # column becomes a very convincing-looking constant.
        saved = json.loads(metrics.read_text()).get("best_checkpoint") or ""
        found = re.search(r"epoch=(\d+)", saved)
        peak_at = int(found.group(1)) if found else None

        curve = run / "lightning_logs/version_0/metrics.csv"
        gap = None
        if curve.exists() and peak_at is not None:
            frame = pl.read_csv(curve)
            # `head(-1)` drops that same appended row before anything is joined on epoch.
            # Narrowed to three columns first: a validation row carries a null `train/loss`
            # of its own, and the join would keep that and suffix the real one.
            val = (
                frame.filter(pl.col("val/macro_map").is_not_null())
                .head(-1)
                .select("epoch", "val/loss", "val/macro_map")
            )
            train = (
                frame.filter(pl.col("train/loss").is_not_null())
                .group_by("epoch").agg(pl.col("train/loss").mean())
            )
            at = val.join(train, on="epoch").filter(pl.col("epoch") == peak_at)
            if not at.is_empty():
                gap = float(at["val/loss"][0]) - float(at["train/loss"][0])
        rows.append(
            {
                "arm": arm,
                "cell": run.name,
                "macro_map": best,
                "peak_epoch": peak_at,
                "val_minus_train": gap,
            }
        )

if not rows:
    print("no metrics.json yet")
else:
    print(pl.DataFrame(rows).to_pandas().to_string(index=False))
PY

stage "done"
cat <<'EOF'

Read these against, same checkpoint and same cell:

    convex probe, mean pool          0.3633   (layer_decay -> 0 is this, in the limit)
    finetune, flat backbone_lr 1e-5  0.3855   (50 epochs)
    finetune, layer_decay 0.75       0.3878   (peak at epoch 13)
    finetune, layer_decay 0.50       0.3942   (peak at epoch 23)

The decay axis is monotone so far. What the new arms decide is where it turns over -- and
whether the turn is a plateau or a peak, since the far end of the axis is the frozen
backbone at 0.3633 and the near end overfits.

`meanpool` is read differently from the rest: against the probe's 0.3633, not against the
other arms. It holds the readout fixed at the probe's, so its gain over 0.3633 is what a
trainable backbone buys, and whatever `decay0.5` has on top of it is what the attentive
pooler buys.

One seed. The 0.75 -> 0.50 gap is +0.0064 against an unmeasured fine-tune seed sd (the
convex probe's is ~0.0025), so the monotone trend across arms is the evidence here, not
any single pair. Whichever arm wins earns seeds 1 and 2 before it is worth quoting.
EOF
