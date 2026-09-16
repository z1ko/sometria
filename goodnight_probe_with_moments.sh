NORM=stats/opensim_sincos_log_vel_acc_tau_v2/pretrain_v1_stats_train_clean.pt

for obj in "" _simmim; do
  uv run python scripts/probe_convex_mae.py \
    --checkpoint runs/pretrain/amass_clean/medium_100ep${obj}/seed42/in_pk__loss_pk \
    --config config/experiment_linear_probe.yaml \
    --output runs/probe/babel_60_moments/amass_clean/medium_100ep${obj}/seed42/in_pk__loss_pk \
    dataloader.normalization=$NORM \
    model.pool=moments
done
