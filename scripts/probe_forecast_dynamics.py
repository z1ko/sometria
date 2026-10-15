#!/usr/bin/env python
"""Forecast torque (and vel, acc, angles) from a frozen backbone, or infer it from motion.

    uv run python scripts/probe_forecast_dynamics.py --mode forecast \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed1/in_pkd__loss_pkd \\
        --output runs/forecast_dynamics/forecast/amass_clean/medium_100ep/seed1/in_pkd__loss_pkd

    --mode forecast   context patches 0..21, decoder fills 22..29 on sin, cos, vel, acc, tau
    --mode infer      encoder sees 0..22 with tau of patch 22 hidden, decoder reads tau there
    --fill zero|hold  how tau is hidden in infer mode; only changes anything for `in_pkd`

Same checkpoint/config contract as `probe_reconstruction.py`; `config.yaml` and
`metrics.json` land in `--output`, which `scripts/forecast_dynamics_matrix.sh` skips on
and `results/forecast_dynamics/gen.py` reads. The design lives in
`sometria.downstream.forecast_dynamics`.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import torch as t
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.forecast_dynamics import (
    CHANNELS,
    CONTEXT_PATCHES,
    DEC_DEPTH,
    DEC_DIM,
    FILLS,
    KINEMATICS,
    MODES,
    DynamicsForecastProbe,
    Scores,
    affine,
    fit_linear,
    frames,
    hide_tau,
)
from sometria.preprocess import TARGET_HZ
from sometria.representation import build_representation

from probe_baseline_mae import latest_checkpoint, load_pretrained


def run(config, checkpoint: Path, arguments) -> dict:
    """Fit the linear floors, train one decoder on one frozen checkpoint, and score it."""

    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    backbone = load_pretrained(checkpoint)
    representation = build_representation(config.dataloader.human)
    channels = representation.indices(*CHANNELS)
    kinematics, tau = representation.indices(*KINEMATICS), representation.indices("tau")[0]

    data = LabelledMotionDataModule(config)
    data.setup("fit")

    # The stats the windows were normalized with, so N m/kg undoes exactly what was done.
    stats = data.train_dataset.motions.stats
    tau_stats = (stats["mean"][0, :, tau], stats["std"][0, :, tau])

    context_patches = config.model.get("context_patches", CONTEXT_PATCHES)
    patch_size = int(backbone.hparams.num_frames_in_patch)
    device = "cuda" if arguments.device.startswith("cuda") else "cpu"
    # One pass over the train loader; the maps are a property of the split, not the
    # checkpoint, but cheap enough to refit per run rather than cache.
    linear = fit_linear(
        data.train_dataloader(), kinematics, tau, context_patches * patch_size, patch_size, device
    )

    model = DynamicsForecastProbe(
        backbone,
        channels,
        linear,
        tau_stats,
        mode=arguments.mode,
        fill=arguments.fill,
        context_patches=context_patches,
        lr=config.model.get("lr", 1e-3),
        dec_depth=config.model.get("dec_depth", DEC_DEPTH),
        dec_dim=config.model.get("dec_dim", DEC_DIM),
    )

    trainer = L.Trainer(
        max_epochs=arguments.epochs,
        accelerator="gpu" if arguments.device.startswith("cuda") else arguments.device,
        devices=1,
        logger=CSVLogger(str(arguments.output), name="lightning_logs"),
        # A measuring instrument, scored at the final epoch: see probe_reconstruction.py.
        enable_checkpointing=False,
        limit_train_batches=arguments.limit_batches or 1.0,
        limit_val_batches=arguments.limit_batches or 1.0,
    )
    trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    scored = trainer.validate(model, data.val_dataloader(), verbose=False)[0]

    horizon_patches = model.time_patches - context_patches if arguments.mode == "forecast" else 1
    return {
        "mode": arguments.mode,
        "fill": arguments.fill if arguments.mode == "infer" else None,
        "context_patches": context_patches,
        "horizon_patches": horizon_patches,
        "patch_seconds": patch_size / TARGET_HZ,
        "dec_depth": model.hparams.dec_depth,
        "dec_dim": model.hparams.dec_dim,
        "epochs": arguments.epochs,
        "metrics": {key.removeprefix("val/"): float(value) for key, value in scored.items()},
        "scores": model.results,
    }


def demo() -> None:
    """Self-check: token layout, tau hiding, the scorer, the linear fit, both index sets."""

    from sometria.models.baseline import patchify

    # --- frames() inverts patchify + flatten ---------------------------------------
    x = t.randn(2, 24, 3, 5)
    tokens = patchify(x, 8).flatten(1, 2)
    assert t.equal(frames(tokens, 3, 8), x.view(2, 3, 8, 3, 5)), "frames() is not patchify's inverse"

    # --- hide_tau touches tau on the target frames and nothing else -----------------
    for fill in FILLS:
        hidden = hide_tau(x, 4, 8, 16, fill)
        assert t.equal(hidden[..., :4], x[..., :4]) and t.equal(hidden[:, :8], x[:, :8])
        assert t.equal(hidden[:, 16:], x[:, 16:])
        expected = t.zeros_like(x[:, 8:16, :, 4]) if fill == "zero" else x[:, 7:8, :, 4].expand(-1, 8, -1)
        assert t.equal(hidden[:, 8:16, :, 4], expected), fill

    # --- Scores: perfect is 1, the validation mean is 0, N m/kg undoes the transform --
    scores = Scores(t.zeros(3), t.ones(3))
    y = t.randn(16, 2, 8, 3)
    mass = t.full((16,), 2.0)
    scores.observe({"tau": y}, mass)
    scores.update("perfect", {"tau": y})
    scores.update("mean", {"tau": y.mean((0, 2), keepdim=True).expand_as(y)})
    scores.update("off", {"tau": t.zeros_like(y)})
    out = scores.compute()
    assert all(abs(v - 1) < 1e-6 for v in out["perfect"]["r2"]["tau"]), out["perfect"]
    assert all(abs(v) < 1e-5 for v in out["mean"]["r2"]["tau"]), out["mean"]
    newton = (y.sign() * y.abs().expm1()).abs() / 2.0                 # (B, P, p, D)
    expected = newton.transpose(0, 1).flatten(1, 2).median(dim=1).values.mean(dim=1)
    assert t.allclose(t.tensor(out["off"]["tau_nmkg"]), expected, atol=1e-5), out["off"]
    # Variance-weighted: a near-constant DOF predicted badly barely moves R^2.
    flat = t.randn(16, 2, 8, 3)
    flat[..., 0] = 1e-4 * flat[..., 0] + 1.0
    scores = Scores(t.zeros(3), t.ones(3))
    scores.observe({"cos": flat}, mass)
    scores.update("near", {"cos": t.cat([flat[..., :1] * 0 + 0.9, flat[..., 1:]], -1)})
    assert min(scores.compute()["near"]["r2"]["cos"]) > 0.99

    # --- fit_linear recovers a linear torque law, both maps --------------------------
    g = t.Generator().manual_seed(0)
    law = t.randn(3 * 4, 3, generator=g)
    batches = []
    for _ in range(4):
        f = t.randn(32, 24, 3, 5, generator=g)
        f[..., 4] = f[..., :4].flatten(2) @ law + 0.5
        batches.append({"features": f})
    weights = fit_linear(batches, (0, 1, 2, 3), 4, 16, 8, "cpu")
    probe = batches[0]["features"]
    for name, x_in in (
        ("id", probe[..., :4].flatten(2)),
        ("id_last", t.cat([probe[:, 16:24, :, :4].flatten(2), probe[:, 15:16, :, 4].expand(-1, 8, -1)], -1)),
    ):
        target = probe[..., 4] if name == "id" else probe[:, 16:24, :, 4]
        assert t.allclose(affine(x_in, weights[name]), target, atol=1e-2), name

    # --- index sets: forecast holds out 22..29, infer reads patch 22 out of 0..22 ------
    from sometria.downstream.reconstruction import suffix_indices

    dofs = 43
    context, targets = suffix_indices(30, dofs, 30 - CONTEXT_PATCHES)
    assert (targets // dofs).min() == CONTEXT_PATCHES and context.shape[1] == CONTEXT_PATCHES * dofs
    context, _ = suffix_indices(30, dofs, 30 - CONTEXT_PATCHES - 1)
    patch = context[:, -dofs:] // dofs
    assert (patch == CONTEXT_PATCHES).all() and context.shape[1] == (CONTEXT_PATCHES + 1) * dofs

    print("probe_forecast_dynamics self-check ok")


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, help="pretrained checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=MODES, default="forecast")
    parser.add_argument("--fill", choices=FILLS, default="zero", help="infer mode: how tau is hidden")
    parser.add_argument("--epochs", type=int, default=40, help="decoder budget, not the backbone's")
    parser.add_argument("--limit-batches", type=int, default=0, help="smoke runs: batches per loop")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    arguments, overrides = parser.parse_known_args()

    if arguments.self_check:
        return demo()
    if arguments.checkpoint is None or arguments.output is None:
        raise SystemExit("--checkpoint and --output are required unless --self-check")

    checkpoint = (
        latest_checkpoint(arguments.checkpoint) if arguments.checkpoint.is_dir() else arguments.checkpoint
    )
    config = OmegaConf.merge(OmegaConf.load(arguments.config), OmegaConf.from_dotlist(overrides))
    OmegaConf.update(config, "model.checkpoint", str(checkpoint))

    arguments.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, arguments.output / "config.yaml")

    result = run(config, checkpoint, arguments)
    (arguments.output / "metrics.json").write_text(json.dumps(result, indent=2))

    for name, value in result["metrics"].items():
        print(f"val/{name}: {value:.4f}")


if __name__ == "__main__":
    main()
