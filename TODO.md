1. Geometry — cheapest, and it runs first because it can invalidate every later table.


SEEDS=1 ./scripts/geometry_matrix.sh          # 10 cells, look at one metrics.json
SEEDS="1 2 3" CKA=1 ./scripts/geometry_matrix.sh
uv run python results/representation/gen.py
Read before continuing: participation ratio came out ~16–18 of 256 on the smoke. If any cell is far below that, its scores elsewhere are explained before you measure them.

2. Segmentation — the real new axis. One cell first, because nothing has run at full scale yet and this is the one that now extracts 20 passes.


CELLS=in_pk__loss_pk SEEDS=1 ./scripts/segmentation_matrix.sh
SEEDS="1 2 3" ./scripts/segmentation_matrix.sh
uv run python scripts/moments_baseline.py --config config/experiment_segmentation.yaml   # the floor
Watch for the check_feature_memory refusal — 20 passes is 1.88 GiB and the guard wants 80% headroom. On a busy CPU it will refuse; that is correct behaviour, not a bug.

3. kNN, then label-efficiency — both window-level, both extraction-bound, both include their moments floor automatically. These can run concurrently with step 2 if you have GPU headroom; different configs, different output trees.


SEEDS="1 2 3" ./scripts/knn_matrix.sh
SEEDS="1 2 3" ./scripts/label_efficiency_matrix.sh
uv run python results/representation/gen.py
4. Reconstruction + forecasting — by far the most expensive: it trains a decoder per cell per mask, so 10 cells × 3 seeds × 2 masks = 60 decoder trainings. Overnight job.


MASKS=random SEEDS=1 ./scripts/reconstruction_matrix.sh    # watch the first cell, then kill
MASKS="random suffix" SEEDS="1 2 3" ./scripts/reconstruction_matrix.sh
uv run python results/reconstruction/gen.py
One thing to check before committing to that overnight run. DECODER_EPOCHS defaults to 10. The smoke at 1 epoch scored 21.97° against a 6.89° zero-velocity floor — an undertrained decoder. If the first full cell still loses to const_velocity, raise DECODER_EPOCHS before spending 60 runs, or the whole table reads "every backbone loses to holding the last frame" and says nothing about the backbones.

Not launching any of these — they write into runs/ and the long one is hours. Say which and I'll start it.