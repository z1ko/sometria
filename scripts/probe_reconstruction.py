#!/usr/bin/env python
"""Decode a frozen backbone's tokens back into joint angles, or into the next ~1 s.

Phases 3 and 4 of `EVALUATION_PLAN.md` are this one script run under two masks. The
backbone is frozen and in eval, a `tinydec` decoder is trained from scratch on top of it,
and the number is the mean per-DOF angular error in **degrees**:

    uv run python scripts/probe_reconstruction.py --mask random \\
        --checkpoint runs/pretrain/amass_clean/medium_100ep/seed1/in_pk__loss_pk \\
        --output runs/reconstruction/random/amass_clean/medium_100ep/seed1/in_pk__loss_pk

    --mask suffix    hold out the last 8 time patches (1.067 s), and report the
                     zero-velocity and constant-velocity forecasts beside the model

Same config/checkpoint contract as `probe_convex_mae.py`: a directory is resolved to its
latest checkpoint, the config is merged with `key=value` overrides, and `config.yaml` plus
`metrics.json` land in `--output` -- the second of which is what
`scripts/reconstruction_matrix.sh` skips on and `results/reconstruction/gen.py` reads.

`--epochs` is the *decoder's* budget and has nothing to do with the pretraining epoch
count in the run path; `config.training.epochs` is the probe's and is deliberately not
read here. `--limit-batches` truncates both loops, for smoke runs only.

Everything about why the decoder is this small, why the target is `sin, cos`, and why the
masks are what they are lives in `sometria.downstream.reconstruction`.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import torch as t
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from sometria.downstream.dataset import LabelledMotionDataModule
from sometria.downstream.reconstruction import (
    MASK_RATIO,
    SUFFIX_PATCHES,
    MotionReconstructionProbe,
    angular_error,
    suffix_indices,
    token_angles,
    velocity_baselines,
    DEC_DEPTH,
    DEC_DIM,
)
from sometria.preprocess import TARGET_HZ
from sometria.representation import Representation, build_representation

from probe_baseline_mae import latest_checkpoint, load_pretrained


def run(config, checkpoint: Path, arguments) -> dict:
    """Train one decoder on one frozen checkpoint and score it."""

    # The mask is drawn per step and the decoder is initialized randomly, so an unseeded
    # run is a different experiment every time. Seeded for the reason probe_convex_mae.py
    # is: the crop draw alone moved macro mAP by ~0.001 there, the size of the effects
    # these tables compare.
    L.seed_everything(config.training.get("seed", 13), workers=True)
    t.set_float32_matmul_precision("high")

    backbone = load_pretrained(checkpoint)
    # Through Representation's channel names, never a literal index: the layout is that
    # class's to own, and `sin, cos` being slots 0 and 1 is a fact about the current
    # version of it rather than about this experiment.
    representation = build_representation(config.dataloader.human)

    model = MotionReconstructionProbe(
        backbone,
        representation.indices("sin", "cos"),
        mask=arguments.mask,
        mask_ratio=config.model.get("mask_ratio", MASK_RATIO),
        suffix_patches=config.model.get("suffix_patches", SUFFIX_PATCHES),
        lr=config.model.get("lr", 1e-3),
        # Overridable so the decoder-capacity ladder is a dotlist override rather than an
        # edited constant. The default stays tiny on purpose -- a probe decoder large
        # enough to compute what the encoder did not is measuring itself. Raise it only to
        # ask whether the default is compressing the cells, and report which rung produced
        # a number: metrics.json already records both.
        dec_depth=config.model.get("dec_depth", DEC_DEPTH),
        dec_dim=config.model.get("dec_dim", DEC_DIM),
    )

    data = LabelledMotionDataModule(config)
    data.setup("fit")

    trainer = L.Trainer(
        max_epochs=arguments.epochs,
        accelerator="gpu" if arguments.device.startswith("cuda") else arguments.device,
        devices=1,
        logger=CSVLogger(str(arguments.output), name="lightning_logs"),
        # Nothing here is reloaded: the decoder is a measuring instrument, thrown away
        # with the number written down. Model selection would also need a monitor, and
        # selecting the decoder on val degrees is selecting on the metric.
        enable_checkpointing=False,
        limit_train_batches=arguments.limit_batches or 1.0,
        limit_val_batches=arguments.limit_batches or 1.0,
    )
    trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    scored = trainer.validate(model, data.val_dataloader(), verbose=False)[0]

    return {
        "mask": arguments.mask,
        "mask_ratio": model.mask_ratio if arguments.mask == "random" else None,
        "suffix_patches": model.suffix_patches if arguments.mask == "suffix" else None,
        # Reported rather than left to a reader: 8 patches is only ~1 s at this patch size
        # and this sample rate, and the first is a property of the checkpoint. TARGET_HZ
        # rather than 60.0 written out, so a corpus resampled to another rate cannot leave
        # a horizon here that says one thing and means another.
        "horizon_seconds": model.horizon / TARGET_HZ if arguments.mask == "suffix" else None,
        "dec_depth": model.hparams.dec_depth,
        "dec_dim": model.hparams.dec_dim,
        "epochs": arguments.epochs,
        "metrics": {key.removeprefix("val/"): float(value) for key, value in scored.items()},
    }


def demo() -> None:
    """Self-check: the degrees conversion, the suffix index set, and the unit assumption."""

    # --- degrees, exact on angles whose answer is known by hand ---------------------
    known = t.deg2rad(t.tensor([0.0, 90.0, 179.0, -90.0]))
    other = t.deg2rad(t.tensor([0.0, 0.0, -179.0, 90.0]))
    expected = t.tensor([0.0, 90.0, 2.0, 180.0])
    measured = angular_error(known, other)
    # 179 vs -179 is the one that matters: a plain difference calls that 358 degrees.
    assert t.allclose(measured, expected, atol=1e-4), measured

    # ...and through the token layout the decoder actually emits. patchify flattens a
    # token as frame * C + channel, so this fails if that inverse is ever mis-stated.
    from sometria.models.baseline import patchify

    angles = t.deg2rad(t.arange(16, dtype=t.float32) * 7.0 - 60.0).view(1, 16, 1, 1)
    features = t.cat([angles.sin(), angles.cos()], dim=-1)          # (1, 16, 1, 2)
    recovered = token_angles(patchify(features, 8).flatten(1, 2), 8)
    assert t.allclose(recovered.reshape(-1), angles.reshape(-1), atol=1e-5), recovered

    # --- the suffix mask is the last 8 time patches, all 43 DOFs --------------------
    time_patches, dofs, suffix = 30, 43, SUFFIX_PATCHES
    context, targets = suffix_indices(time_patches, dofs, suffix)
    assert targets.numel() == suffix * dofs == 344, targets.shape
    # Disjoint and exhaustive: together they name every token exactly once.
    assert t.equal(t.cat([context, targets], dim=1).sort().values, t.arange(time_patches * dofs).unsqueeze(0))
    # And they are the *final* patches under the t * 43 + d flattening -- the check that
    # would fail if the grid were ever laid out d * TP + t instead, which is the failure
    # mode that looks like a working forecaster trained on the wrong half of the window.
    patch_of = targets // dofs
    assert patch_of.min().item() == time_patches - suffix, patch_of.min()
    assert patch_of.max().item() == time_patches - 1, patch_of.max()
    assert sorted(set((targets % dofs).flatten().tolist())) == list(range(dofs))

    # --- the baselines see the context and predict the horizon ----------------------
    # A constant-rate ramp *through* +pi: constant velocity is exact, zero velocity is not,
    # and both go wrong if the differencing does not wrap.
    ramp = t.linspace(2.0, 2.0 + 0.02 * 239, 240).view(1, 240, 1)
    ramp = t.atan2(ramp.sin(), ramp.cos())
    baselines = velocity_baselines(ramp, SUFFIX_PATCHES * 8)
    assert baselines["const_velocity"].max() < 1e-2, baselines["const_velocity"].max()
    assert baselines["zero_velocity"].mean() > 30.0, baselines["zero_velocity"].mean()

    # --- sin/cos really are physical units at the model's input ---------------------
    # The whole degrees story rests on it: Representation compresses and normalizes the
    # derivative channels only, so atan2 on the encoder's own input is the joint angle.
    # If a future layout normalizes pose, this breaks here rather than silently reporting
    # degrees of a standardized quantity.
    representation = Representation(dofs=("hip_flexion_r",), excluded=())
    stats = {"mean": t.full((1, 5), 3.0), "std": t.full((1, 5), 2.0)}
    raw = t.randn(4, 1, 5)
    model_input = representation.to_model(raw, stats)
    assert t.equal(model_input[..., :2], raw[..., :2]), "pose channels were normalized"
    assert not t.equal(model_input[..., 2:], raw[..., 2:]), "derivative channels were not"

    print("probe_reconstruction self-check ok")


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, help="pretrained checkpoint or run directory")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_linear_probe.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask", choices=("random", "suffix"), default="random")
    parser.add_argument("--epochs", type=int, default=10, help="decoder budget, not the backbone's")
    parser.add_argument("--limit-batches", type=int, default=0, help="smoke runs: batches per loop")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    arguments, overrides = parser.parse_known_args()

    if arguments.self_check:
        return demo()
    if arguments.checkpoint is None or arguments.output is None:
        raise SystemExit("--checkpoint and --output are required unless --self-check")

    checkpoint = (
        latest_checkpoint(arguments.checkpoint)
        if arguments.checkpoint.is_dir()
        else arguments.checkpoint
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
