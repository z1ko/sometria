"""The same readout as the linear probe, with the backbone training under it.

A probe asks what a representation already contains; a finetune asks what it is a good
*starting point* for. They share the head, the loss and the metrics, so this is
:class:`MotionLinearClassifier` with the three things that make it a probe removed:
the backbone stays in train mode, the gradient reaches it, and it is in the optimizer.

Two knobs, and both are the same knob: the backbone moves at ``backbone_lr`` while the
fresh head moves at ``lr``. Pretrained weights are close to where they should be and a
random head is not, so one rate for both either scrambles the backbone or starves the
head. Ten to a hundred times lower is the usual range.

``dropout`` is the one architectural knob, and it has to live here rather than in the
config's ``encoder:`` block: that block is only read when no checkpoint is named, so a
pretrained backbone always arrives at whatever dropout it was pretrained with. Leave it
``None`` until a run actually overfits.

``layer_decay`` replaces both knobs with the scheme the literature actually uses, and is
the recommended setting -- see :func:`layerwise_param_groups`. ``backbone_lr`` stays for the
runs made before it existed, and is ignored whenever ``layer_decay`` is set.
"""

import re

import lightning as L
import torch as t
import torch.nn as nn

from sometria.architecture.encoder import MotionTransformerEncoder
from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.classifier import MotionLinearClassifier


def param_groups(module: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    """Split ``module`` into a decayed and an undecayed group at one learning rate.

    Biases, LayerNorm gains and the positional table are 1-D and are excluded: decay on
    them is not regularization, it is a constant pull of the normalization statistics
    toward zero, and on a pretrained backbone it erases the scales pretraining set.
    """

    decay = [p for p in module.parameters() if p.requires_grad and p.ndim >= 2]
    plain = [p for p in module.parameters() if p.requires_grad and p.ndim < 2]
    # Empty groups dropped: a parameterless pooler ("mean") would otherwise contribute two
    # of them, and every lr log line then carries a rate that steers nothing.
    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for params, wd in ((decay, weight_decay), (plain, 0.0))
        if params
    ]


#: A transformer block inside ``nn.TransformerEncoder``. Both encoder lineages in this
#: repo name their stack ``blocks``, so one pattern covers ``sometria.models.baseline.Encoder``
#: and ``sometria.architecture.encoder.MotionTransformerEncoder`` alike.
BLOCK = re.compile(r"(?:^|\.)blocks\.layers\.(\d+)\.")

#: Never weight-decayed, whatever their rank. A positional table is a learned coordinate,
#: not a weight, and decaying it pulls the grid toward a single point. MAE excludes its
#: ``pos_embed`` by name for the same reason. The 1-D rule `param_groups` uses does not catch
#: these: ``sometria.models.baseline.Encoder`` stores them as ``(1, 1, V, D)``.
NO_DECAY = ("pos_s", "pos_t", "position.")


def depth_of(name: str, num_layers: int) -> int:
    """Which rung of the ladder a parameter sits on. 0 is the input, ``num_layers`` the output.

    Three cases, and the middle one is the only one that needs reading twice. A parameter
    inside block ``i`` is rung ``i + 1``. The stack's *final* norm is named ``blocks.norm``
    with no layer index, and it runs after every block, so it belongs at the top rather than
    at the bottom with the tokenizer. Everything else -- the patch projection and the
    positional tables -- feeds the first block and is rung 0.
    """

    found = BLOCK.search(name)
    if found:
        return int(found.group(1)) + 1
    return num_layers if "blocks." in name else 0


def layerwise_param_groups(
    module: nn.Module, base_lr: float, weight_decay: float, layer_decay: float
) -> list[dict]:
    """Optimizer groups whose learning rate decays with depth, as in BEiT and MAE.

    The rate at rung ``k`` is ``base_lr * layer_decay ** (num_layers - k)``, so the block
    nearest the head moves at ``base_lr * layer_decay`` and the tokenizer crawls. The
    reasoning is that a pretrained network is general at the bottom and specific at the top,
    so a new task should be free to rewrite the top and should barely disturb the bottom --
    where one flat ``backbone_lr`` has to be low enough for the bottom and is then far too
    low for the top.

    MAE (arXiv:2111.06377, Table 9) fine-tunes ViT-B at base lr 1e-3 with ``layer_decay``
    0.75 and no separate head rate at all; the head simply sits at rung ``num_layers``, where
    the scale is 1. This follows that, which is why ``backbone_lr`` has nothing to do here.

    ``num_layers`` is read off the parameter names rather than off a spec, because this
    module is handed whatever adapter the caller wrapped its checkpoint in and may never see
    an ``EncoderSpec``. A backbone with no recognisable blocks lands every parameter on rung
    0 and is uniformly scaled, which is wrong but is wrong quietly in the safe direction --
    so it is asserted instead.
    """

    depths = [int(found.group(1)) for name, _ in module.named_parameters() if (found := BLOCK.search(name))]
    if not depths:
        raise ValueError(
            "no transformer blocks found in the backbone's parameter names, so there is no "
            "ladder to decay along; pass layer_decay=None to use a flat backbone_lr instead"
        )
    num_layers = max(depths) + 2

    groups: dict[tuple[int, bool], dict] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        # Biases and norm gains are 1-D and stay undecayed, for the reason `param_groups`
        # gives: decay on them is not regularization. The positional tables join them by name.
        decayed = parameter.ndim >= 2 and not any(key in name for key in NO_DECAY)
        depth = depth_of(name, num_layers)
        group = groups.setdefault(
            (depth, decayed),
            {
                "params": [],
                "lr": base_lr * layer_decay ** (num_layers - depth),
                "weight_decay": weight_decay if decayed else 0.0,
            },
        )
        group["params"].append(parameter)

    return [groups[key] for key in sorted(groups)]


class MotionFinetuneClassifier(MotionLinearClassifier):
    """Multi-label action classification over one window, backbone included."""

    def __init__(
        self,
        backbone: MotionTransformerEncoder,
        num_labels: int,
        pool: str = "attentive_factorized",
        min_lr_frac: float = 0.01,
        lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        # None keeps the flat `backbone_lr`, which is what every run before this knob
        # existed used, so an old checkpoint's hparams still describe how it was trained.
        # Set it and `backbone_lr` is ignored: the two are different answers to one question.
        layer_decay: float | None = None,
        weight_decay: float = 0.05,
        dropout: float | None = None,
        warmup: float = 0.05,
    ) -> None:
        super().__init__(
            backbone,
            num_labels,
            pool=pool,
            min_lr_frac=min_lr_frac,
            lr=lr,
            warmup=warmup,
        )

        self.backbone_lr = backbone_lr
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay

        # Undo the probe. The parent froze and eval'd the backbone in its own __init__,
        # which is correct there and is exactly what this class exists not to do.
        self.backbone.requires_grad_(True)
        self.backbone.train()

        # Set on the modules rather than rebuilt from a new spec: dropout is a rate, not a
        # weight, so a pretrained backbone can be regularized without reloading it. The
        # spec is left reporting what pretraining used, which is what it is a record of.
        if dropout is not None:
            for module in self.backbone.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout

    def train(self, mode: bool = True) -> "MotionFinetuneClassifier":
        """Follow the parent module, unlike the probe, which pins the backbone to eval.

        Skipping one level in the MRO rather than calling ``super()``: the method being
        skipped is the freeze itself.
        """

        L.LightningModule.train(self, mode)
        return self

    def forward(self, features: t.Tensor) -> t.Tensor:
        # No torch.no_grad, which is the whole difference: the probe's forward severs the
        # graph at the backbone, so a gradient that reached it would have nowhere to go.
        return self.head(self.pooler(self.backbone.embed_tokens(features)))

    def configure_optimizers(self):  # type: ignore
        backbone = (
            param_groups(self.backbone, self.backbone_lr, self.weight_decay)
            if self.layer_decay is None
            else layerwise_param_groups(
                self.backbone, self.lr, self.weight_decay, self.layer_decay
            )
        )
        # The head and the pooler sit at the top of the ladder, where the depth scale is 1,
        # so they take `lr` under either scheme and this line does not branch.
        groups = [
            *backbone,
            *param_groups(self.pooler, self.lr, self.weight_decay),
            *param_groups(self.head, self.lr, self.weight_decay),
        ]
        # lr comes from each group; passing it here only sets a default for groups that
        # omit it, and none of these do.
        optimizer = t.optim.AdamW(groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # A factor schedule, not an absolute floor: it scales each group by its
                # own base lr, so the backbone at 1e-4 and the head at 1e-3 decay in
                # step instead of the backbone annealing toward a rate above its own.
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    min_factor=self.min_lr_frac,
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
