"""SimMIM over motion tokens: mask inside the encoder, read out with one linear layer.

SimMIM (Xie et al., https://arxiv.org/abs/2111.09886). The third masked objective here and
the one that moves the mask: :class:`~sometria.models.baseline.MAE` drops the held-out
tokens before the encoder runs and rebuilds them in a decoder, while this replaces each
held-out patch embedding with a learned vector and runs the *whole* grid through the
encoder. The held-out positions are then contextualized by the encoder itself, which is
why a single linear layer is enough to read the patch values back out.

That distinction is not a tuning knob, and it is worth being precise about why MAE cannot
simply shrink its decoder to nothing to get here: in MAE the decoder is the *only* path
from a visible token to a masked position, because the encoder never sees a masked
position at all. A zero-depth MAE decoder would predict each patch from its positional
embedding alone. SimMIM moves that routing into the encoder, and only then does the head
become a head rather than the model.

What it costs: the encoder runs over all 1290 tokens rather than 129, so roughly 1.7-2x
MAE per step at these shapes -- less than it sounds, because MAE's decoder already runs
over the full grid.

Taken from the paper:

- The head is a single ``nn.Linear``. Their Table 2 puts it at 82.8 against 82.8 for a
  2-layer MLP at 1.2x the cost, and 82.4-82.5 for inverse-Swin decoders at 1.7-2.3x. There
  is nothing to buy above linear.
- The loss scores the masked tokens only. Their Table 4 puts that at 82.8 against 81.7 for
  reconstructing every patch.

Deliberately *not* taken from the paper, both for comparability with the MAE baseline:

- The loss stays MSE over per-token standardized targets rather than their L1. Their
  Table 5 prices L1 at 82.8 against L2 at 82.7, which is noise, and holding the loss fixed
  keeps the architecture the only difference between this and `baseline.MAE`.
- ``mask_ratio`` stays at 0.90 rather than their 0.60. Their AvgDist analysis finds
  fine-tuning accuracy sits on a ridge in the distance from a masked token to the nearest
  visible one, and 0.90 is outside the band they validate. It is kept anyway because the
  experiment this model exists for is the input/output channel matrix, and that comparison
  needs the masking held fixed against MAE. Worth knowing what it means concretely: at 0.90
  the encoder's input is 1161 copies of one mask token differing only by position, and 129
  tokens carrying real values. If a first cell probes near chance, a 0.60 arm is the thing
  to try before concluding anything about channels.

There is no ``dec_depth`` or ``dec_dim``, because there is no decoder to size. That is why
``config/simmim/`` exists as its own size-overlay tree: ``config/mae/<arch>.yaml`` sets
``dec_depth`` on every file, and rather than accept the argument and quietly ignore it --
leaving a checkpoint whose hyperparameters claim a decoder depth that never existed -- the
sweep picks its size overlay per objective and a mismatched one fails loudly at construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from sometria.architecture.scheduler import lr_schedule
from sometria.models.baseline import Encoder, gather_tokens, patchify, random_mask


class MaskedInputEncoder(Encoder):
    """An :class:`~sometria.models.baseline.Encoder` that reads masked positions too.

    Subclassed rather than folded into the shared encoder on purpose. ``mask_token`` is a
    new ``state_dict`` entry and every MAE and JEPA checkpoint on disk was written without
    one, so adding it upstream would break strict loading for all of them. Here it lands
    under this model's own ``encoder.`` prefix and touches nothing else.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        keep: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``mask`` is ``(B, K)`` flat token indices to hide. None reproduces the parent."""

        time_patches = x.size(1) // self.num_frames_in_patch
        e = self.proj(patchify(x, self.num_frames_in_patch)).flatten(1, 2)

        if mask is not None:
            hidden = torch.zeros(e.shape[:2], dtype=torch.bool, device=e.device)
            hidden = hidden.scatter(1, mask, True)
            # `.to(e.dtype)` for the reason the decoder's scatter needs it: under autocast a
            # Linear hands back bf16 while the parameter stays fp32, and `where` will not
            # mix them.
            e = torch.where(hidden.unsqueeze(-1), self.mask_token.to(e.dtype), e)

        # Position is added *after* the substitution, never before. A mask token still has
        # to know where it is, and substituting over an embedding that already carried its
        # position would throw that away -- every hidden token would then be identical and
        # the encoder could not tell them apart.
        e = e + (self.pos_s + self.pos_t[:, :time_patches]).flatten(1, 2)

        token_pad = self.pad(padding) if padding is not None else None
        if keep is not None:
            e = gather_tokens(e, keep)
            if token_pad is not None:
                token_pad = torch.gather(token_pad, 1, keep)

        return self.blocks(e, src_key_padding_mask=token_pad)


class SimMIM(L.LightningModule):
    def __init__(
        self,
        num_dofs: int = 43,
        num_frames_in_patch: int = 8,
        num_frames: int = 240,
        enc_depth: int = 8,
        mask_ratio: float = 0.90,
        min_lr_ratio: float = 0.5,
        warmup: float = 0.05,
        weight_decay: float = 0.05,
        mlp_ratio: int = 4,
        channels_input: tuple[int, ...] = (0, 1, 2, 3, 4),
        channels_output: tuple[int, ...] = (0, 1, 2, 3, 4),
        num_heads: int = 8,
        dim: int = 256,
        lr: float = 1e-3,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.mask_ratio = mask_ratio
        self.num_frames_in_patch = num_frames_in_patch
        self.channels_output = channels_output
        self.channels_input = channels_input
        self.weight_decay = weight_decay
        self.min_lr_ratio = min_lr_ratio
        self.warmup = warmup
        self.lr = lr

        Te = num_frames // num_frames_in_patch
        self.num_tokens = Te * num_dofs

        self.encoder = MaskedInputEncoder(
            num_dofs, num_frames_in_patch, dim, enc_depth, num_heads, channels_input, mlp_ratio, Te
        )
        # `head`, not `decoder`: the name is the only thing distinguishing a SimMIM
        # checkpoint from an MAE one, which both carry `encoder.` and would otherwise both
        # carry `decoder.`. `scripts/probe_baseline_mae.load_pretrained` dispatches on
        # state_dict prefixes, so this one has to be unique.
        self.head = nn.Linear(dim, num_frames_in_patch * len(channels_output))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, C) with full channels -> (B, N, E). No mask: probes see everything."""
        return self.encoder(x[..., self.channels_input], None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, C) with full channels -> scalar loss"""

        # `keep` is unused: the encoder reads the whole grid. That is the model.
        _, mask_idx = random_mask(self.num_tokens, self.mask_ratio, x.size(0), x.device)

        h = self.encoder(x[..., self.channels_input], None, None, mask_idx)
        prediction = self.head(h)

        target = patchify(x[..., self.channels_output], self.num_frames_in_patch).flatten(1, 2)

        # Per-token normalization: the loss asks for the shape of a patch,
        # not its magnitude. Without this, high-variance channels (acc, tau)
        # dominate the gradient and the model learns noise.
        target_mean = target.mean(dim=-1, keepdim=True)
        target_var = target.var(dim=-1, keepdim=True)
        target_norm = (target - target_mean) / (target_var + 1e-6).sqrt()
        return F.mse_loss(
            gather_tokens(prediction, mask_idx),
            gather_tokens(target_norm, mask_idx),
        )

    def training_step(self, x: dict, _):
        loss = self.forward(x["features"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, x: dict, _):
        self.log("val/loss", self.forward(x["features"]), prog_bar=True)

    def configure_optimizers(self): # type: ignore
        total_steps = int(self.trainer.estimated_stepping_batches)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            lr=self.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    min_factor=self.min_lr_ratio,
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
