"""Run with: python tests/test_objective.py

The base is exercised through a stub objective rather than through MAE or JEPA: what is
being checked here is the loop, and a test that needs a backbone to reach it cannot say
which of the two broke.
"""

import sys
import tempfile
from pathlib import Path

import lightning as L
import torch as t
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sometria.architecture.encoder import EncoderSpec, MotionTransformerEncoder
from sometria.masking import MaskSpec
from sometria.models.objective import PretextObjective

SPEC = EncoderSpec(num_dofs=6, num_features=5, patch_size=8, window_frames=80, d_model=32, depth=1, num_heads=4)


class Stub(PretextObjective):
    """Everything a subclass has to say: a default mask, a step, and what it trains."""

    DEFAULT_MASK = MaskSpec(mask_ratio=0.25, tau=0.0, score_channels=())
    PROG_BAR = ("spread",)

    def __init__(self, backbone=None, mask=None, *, width=4, lr=1e-3, min_lr_frac=0.5,
                 weight_decay=0.05, warmup_frac=0.05, freeze=False):
        super().__init__(mask, lr=lr, min_lr_frac=min_lr_frac,
                         weight_decay=weight_decay, warmup_frac=warmup_frac)
        backbone = backbone if isinstance(backbone, MotionTransformerEncoder) else MotionTransformerEncoder(SPEC)
        self.save_objective_hyperparameters(backbone, width=width, freeze=freeze)

        self.backbone = backbone
        self.head = nn.Linear(SPEC.d_model, width)
        self.freeze = freeze

    def step(self, batch):
        pooled = self.backbone.embed(batch["features"])
        out = self.head(pooled)
        return out.square().mean(), {"spread": out.std(), "count": float(out.shape[0])}

    def trainable_parameters(self):
        return self.head.parameters() if self.freeze else self.parameters()


class Batches(t.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return {"features": t.randn(80, 6, 5)}


def _fit(model):
    loader = t.utils.data.DataLoader(Batches(), batch_size=2)
    trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, accelerator="cpu")
    trainer.fit(model, loader, loader)
    return trainer


def test_every_metric_a_step_returns_is_logged_under_its_stage():
    trainer = _fit(Stub())
    logged = set(trainer.callback_metrics)

    assert {"train/loss", "train/spread", "train/count"} <= logged
    assert {"val/loss", "val/spread", "val/count"} <= logged


def test_an_unconfigured_objective_gets_its_own_default_mask():
    assert Stub().mask == Stub.DEFAULT_MASK
    assert Stub(mask=MaskSpec(mask_ratio=0.9)).mask.mask_ratio == 0.9
    # a YAML block or a checkpoint hands back plain fields, not the dataclass
    assert Stub(mask={"mask_ratio": 0.75, "tau": 0.0, "score_channels": []}).mask.mask_ratio == 0.75


def test_frozen_parts_stay_out_of_the_optimizer():
    """requires_grad=False stops the gradient; leaving a parameter out stops weight decay."""

    trainer = _fit(Stub(freeze=True))
    optimizer = trainer.optimizers[0]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert optimized == {id(p) for p in trainer.lightning_module.head.parameters()}
    assert not any(id(p) in optimized for p in trainer.lightning_module.backbone.parameters())


def test_the_schedule_warms_up_and_decays_toward_min_lr_frac():
    model = Stub(lr=1.0, min_lr_frac=0.5, warmup_frac=0.5)
    optimizer = t.optim.AdamW(model.parameters(), lr=1.0)
    from sometria.architecture.scheduler import lr_schedule

    schedule = lr_schedule(optimizer, warmup_steps=10, total_steps=20, min_factor=0.5)
    seen = []
    for _ in range(21):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()

    assert abs(seen[0] - 0.5) < 1e-6          # starts at the floor
    assert abs(seen[10] - 1.0) < 1e-6         # peaks at the end of warmup
    assert abs(seen[20] - 0.5) < 1e-6         # decays back to it, not to zero


def test_a_checkpoint_reloads_both_specs_and_the_objective_s_own_fields():
    model = Stub(mask=MaskSpec(mask_ratio=0.4, tau=0.0, score_channels=()), width=7, lr=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stub.ckpt"
        t.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.0.0",
                "loops": {},
            },
            path,
        )
        reloaded = Stub.load_from_checkpoint(path, map_location="cpu")

    assert reloaded.mask == model.mask
    assert reloaded.backbone.spec == SPEC
    assert reloaded.head.out_features == 7
    assert reloaded.lr == 0.5
    # plain fields, not the dataclasses: Lightning refuses to log a frozen dataclass
    assert isinstance(reloaded.hparams.backbone, dict)
    assert isinstance(reloaded.hparams.mask, dict)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
