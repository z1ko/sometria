"""Can a frozen backbone's tokens be decoded back into joint angles, and into the future?

One rig, two masks. A pretrained backbone is frozen and in ``eval()``, a small decoder is
trained from scratch on top of it, and the score is the **mean per-DOF angular error in
degrees** over the held-out tokens. Which tokens are held out is the only difference
between the two questions this answers:

``mask="random"``
    Uniform masking at ``MASK_RATIO``. Reconstruction, and explicitly a *diagnostic*
    rather than a ranking -- a near-identity encoder passes everything through and
    decodes beautifully while abstracting nothing, so a cell can win here by being worse.
    See ``results/reconstruction/README.md``, where that prediction is written down before
    the table exists.

``mask="suffix"``
    The last ``SUFFIX_PATCHES`` time patches across every DOF, as a fixed index set rather
    than a draw. Forecasting over 8 x 8 = 64 frames = 1.067 s at 60 Hz. Scored against
    zero-velocity and constant-velocity extrapolation in the same unit on the same
    windows, because at 60 Hz much of a 1 s horizon is predictable from the last two
    frames and a bare error table cannot tell a good backbone from an easy horizon.

Three choices carry the experiment, and all three are about *not* measuring something else.

**The decoder is tiny and identical everywhere.** ``dec_depth 1`` / ``dec_dim 64``, the
``tinydec`` rung from ``scripts/decoder_capacity.sh``, about 0.4x the encoder's FLOPs.
This is the confound that has already bitten this project once: ``results/decoder_capacity/
README.md`` documents that ``medium``'s pretraining decoder runs at 8.6x the encoder's
FLOPs and can compute torque from kinematics by itself -- a plain linear map recovers
R^2 = 0.685 -- which is the standing explanation for why the ``pkd`` loss channel looks
inert. A probe decoder that size would rank decoders rather than backbones. The same
decoder serves both masks, so phase 3 and phase 4 differ in the mask and in nothing else.

**The target is ``sin, cos`` only, resolved through :class:`~sometria.representation
.Representation` rather than by literal index.** The matrix varies the *input* channels
across cells (``p``/``pk``/``pkd``), so the pose channels are the only ones every cell
has; anything wider would score the nine cells on nine different targets. Those two
channels are also the only ones that come back as a physical quantity for free:
``Representation`` leaves pose uncompressed and unnormalized (``signed_log`` and the
mean/std statistics apply to the derivative channels only), so ``atan2(sin, cos)`` on the
model's own input is already the joint angle in radians. That is what makes degrees
available without a denormalization round trip, and it is checked in ``demo()``.

**The loss is plain MSE on raw ``sin, cos``, not the per-token standardized target
pretraining uses.** ``norm_targets`` divides each token by its own spread, which is right
when the loss must not be dominated by ``acc`` and ``tau``, and fatal here: a per-token
affine rescaling is exactly the information ``atan2`` needs, so a standardized prediction
cannot be turned back into an angle at all. It also means each cell's own pretraining loss
is not this number and never was -- ``loss_p`` scores 2 channels and ``loss_pkd`` scores 5,
on per-token scales that differ per token.
"""

import lightning as L
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

from sometria.architecture.scheduler import lr_schedule
from sometria.models.baseline import Decoder, gather_tokens, patchify, random_mask

#: Held out under ``mask="random"``. Deliberately off pretraining's 0.90, so the number
#: is not each cell's own training objective wearing a new name, and identical for every
#: cell. Uniform (``MaskSpec.tau <= 0``) rather than motion-aware for the same reason the
#: ratio is fixed: motion-aware masking draws its holes through a per-cell config, and
#: would hand different cells windows of different difficulty.
MASK_RATIO: float = 0.75

#: Held out under ``mask="suffix"``, in time patches. 8 x ``patch_size`` 8 = 64 frames =
#: 1.067 s at 60 Hz, reported as "~1 s"; 1 s exactly is 7.5 patches and does not exist.
#: Taken out of the existing 240-frame window rather than by building a longer one, which
#: would mean a new ``min_frames``, a smaller eligible sample pool and a new collate to
#: buy a horizon the current window already contains.
SUFFIX_PATCHES: int = 8

#: The ``tinydec`` rung, named in ``scripts/decoder_capacity.sh``. See the module
#: docstring for why a bigger decoder would measure the wrong thing.
DEC_DEPTH: int = 1
DEC_DIM: int = 64


def suffix_indices(
    time_patches: int, num_dofs: int, suffix: int, device: t.device | str = "cpu"
) -> tuple[t.Tensor, t.Tensor]:
    """``(context, targets)`` for "hold out the last ``suffix`` time patches".

    Both are ``(1, n)`` and are meant to be expanded over the batch: unlike a Gumbel draw
    every window is masked identically, which is the point -- a forecasting horizon that
    moved per window would not be one horizon.

    The token grid flattens as ``t * num_dofs + d`` (:func:`~sometria.masking.patchify`,
    and the order ``PositionalEncoding.get_flat`` produces), so the final ``suffix`` time
    patches across *all* DOFs are exactly the last ``suffix * num_dofs`` flat indices. The
    split is therefore a slice, not a gather -- which is also what makes it checkable.
    """

    if not 0 < suffix < time_patches:
        raise ValueError(f"suffix of {suffix} patches does not fit in {time_patches}")

    flat = t.arange(time_patches * num_dofs, device=device).unsqueeze(0)
    return flat[:, : (time_patches - suffix) * num_dofs], flat[:, (time_patches - suffix) * num_dofs :]


def token_angles(tokens: t.Tensor, patch_size: int) -> t.Tensor:
    """``(B, N, patch_size * 2)`` holding ``sin, cos`` -> ``(B, N, patch_size)`` radians.

    :func:`~sometria.models.baseline.patchify` flattens a token's frames and channels as
    ``frame * C + channel``, so the unflatten below is its exact inverse and channel 0 is
    whichever channel was named first -- ``sin``, by construction of the caller.

    ``atan2`` reads a direction, so a prediction whose ``(sin, cos)`` is off the unit
    circle still decodes to an angle. That is deliberate: the decoder is never told to
    produce a unit vector, and projecting one out of it is the same operation the metric
    already performs.
    """

    sin, cos = tokens.unflatten(-1, (patch_size, 2)).unbind(-1)
    return t.atan2(sin, cos)


def angular_error(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """Absolute angular difference in degrees, wrapped to ``[0, 180]``.

    Through ``atan2(sin d, cos d)`` rather than by subtracting: an angle recovered from
    ``atan2`` lives on ``(-pi, pi]``, so a raw difference across the branch cut reports
    358 degrees for a two-degree error. That is not a rounding detail -- the wrap is
    crossed by any DOF whose angle passes +-pi mid-window.
    """

    difference = prediction - target
    return t.rad2deg(t.atan2(difference.sin(), difference.cos()).abs())


def velocity_baselines(angles: t.Tensor, horizon: int) -> dict[str, t.Tensor]:
    """Zero- and constant-velocity forecasts of the last ``horizon`` frames, in degrees.

    ``angles`` is ``(B, T, D)`` radians for the *whole* window; the split is the same one
    the suffix mask makes, so the baselines see exactly the frames the encoder sees and
    predict exactly the frames the decoder predicts.

    No model, no fit, nothing to train -- which is why they are computed inside the
    validation loop rather than by a separate script. The velocity is differenced from the
    last two observed frames through the same wrap the metric uses; reading the stored
    ``vel`` channel instead would be a different quantity, signed-log compressed and
    normalized per DOF (the mistake ``extract_motion`` exists to avoid).
    """

    observed, future = angles[:, :-horizon], angles[:, -horizon:]
    last = observed[:, -1:]                                          # (B, 1, D)
    step = t.arange(1, horizon + 1, device=angles.device).view(1, -1, 1)

    difference = observed[:, -1:] - observed[:, -2:-1]
    velocity = t.atan2(difference.sin(), difference.cos())           # per frame, wrapped

    return {
        "zero_velocity": angular_error(last.expand_as(future), future),
        "const_velocity": angular_error(last + velocity * step, future),
    }


class MotionReconstructionProbe(L.LightningModule):
    """A frozen backbone, one freshly-trained ``tinydec``, scored in degrees.

    ``backbone`` is a pretraining ``LightningModule`` as
    :func:`probe_baseline_mae.load_pretrained` returns it -- the encoder and its
    hyperparameters are read off it, so the architecture is never restated here and a
    checkpoint cannot be decoded under a shape it was not trained with.
    """

    def __init__(
        self,
        backbone: L.LightningModule,
        channels_output: tuple[int, ...],
        *,
        mask: str = "random",
        mask_ratio: float = MASK_RATIO,
        suffix_patches: int = SUFFIX_PATCHES,
        dec_depth: int = DEC_DEPTH,
        dec_dim: int = DEC_DIM,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup: float = 0.05,
        min_lr_frac: float = 0.5,
    ) -> None:
        super().__init__()

        if mask not in ("random", "suffix"):
            raise ValueError(f"mask must be 'random' or 'suffix', got {mask!r}")
        if len(channels_output) != 2:
            raise ValueError(
                f"the target is the two pose channels; got {len(channels_output)}. "
                "Pass Representation.indices('sin', 'cos') rather than a literal."
            )

        # Not the backbone: its weights are an input to this experiment, not a result of
        # it. Same reason MotionLinearClassifier ignores it.
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        hp = backbone.hparams
        self.channels_input = list(hp.channels_input)
        self.channels_output = list(channels_output)
        self.patch_size = int(hp.num_frames_in_patch)
        self.num_frames = int(hp.num_frames)
        self.num_dofs = int(hp.num_dofs)
        self.time_patches = self.num_frames // self.patch_size
        self.num_tokens = self.time_patches * self.num_dofs

        self.mask = mask
        self.mask_ratio = mask_ratio
        self.suffix_patches = suffix_patches
        self.horizon = suffix_patches * self.patch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.min_lr_frac = min_lr_frac

        self.decoder = Decoder(
            self.num_dofs,
            self.patch_size,
            self.time_patches,
            tuple(channels_output),
            int(hp.dim),
            dec_depth,
            int(hp.num_heads),
            int(hp.mlp_ratio),
            dec_dim,
        )

        # Buffers, so the fixed index set moves with the module to whatever device
        # Lightning puts it on and is not rebuilt per step.
        context, targets = suffix_indices(self.time_patches, self.num_dofs, suffix_patches)
        self.register_buffer("suffix_context", context, persistent=False)
        self.register_buffer("suffix_targets", targets, persistent=False)

        # Sums over tokens rather than a mean of per-batch means: the tile loader's last
        # batch is short, and a mean of means would weight its windows more heavily.
        self.val_degrees = MeanMetric()
        # Empty under the random mask rather than present-and-unused: a zero-velocity
        # forecast of uniformly scattered holes is not a baseline of anything, and an
        # empty dict logs no column instead of a meaningless one.
        names = ("zero_velocity", "const_velocity") if mask == "suffix" else ()
        self.val_baselines = nn.ModuleDict({name: MeanMetric() for name in names})

    def train(self, mode: bool = True) -> "MotionReconstructionProbe":
        """Keep the backbone in eval mode; Lightning will not do it for you.

        ``requires_grad_(False)`` stops gradients but not dropout, and a backbone that
        still samples is not the fixed feature extractor this measures.
        """

        super().train(mode)
        self.backbone.eval()
        return self

    def split(self, batch_size: int, device: t.device) -> tuple[t.Tensor, t.Tensor]:
        """``(context, targets)`` index sets for this batch, per the configured mask."""

        if self.mask == "suffix":
            return (
                self.suffix_context.expand(batch_size, -1),
                self.suffix_targets.expand(batch_size, -1),
            )
        # Uniform, which is what MaskSpec spells `tau <= 0`; this rig reaches for the
        # encoder's own sampler instead, because it is the one that already returns the
        # flat (keep, mask) index pair the Decoder's scatter consumes.
        return random_mask(self.num_tokens, self.mask_ratio, batch_size, device)

    def forward(self, features: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """``(B, T, D, C)`` -> predicted and true target tokens, ``(B, K, patch * 2)``."""

        if features.shape[1] != self.num_frames:
            raise ValueError(
                f"backbone was pretrained on {self.num_frames}-frame windows, got "
                f"{features.shape[1]}: its positional embeddings do not cover the grid"
            )

        context, targets = self.split(features.shape[0], features.device)
        with t.no_grad():
            tokens = self.backbone.encoder(features[..., self.channels_input], context, None)

        prediction = self.decoder(tokens, context)
        target = patchify(features[..., self.channels_output], self.patch_size).flatten(1, 2)
        return gather_tokens(prediction, targets), gather_tokens(target, targets)

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        prediction, target = self(batch["features"])
        loss = F.mse_loss(prediction, target)
        self.log("train/loss", loss, prog_bar=True, batch_size=target.shape[0])
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        features = batch["features"]
        prediction, target = self(features)
        loss = F.mse_loss(prediction, target)

        degrees = angular_error(
            token_angles(prediction, self.patch_size), token_angles(target, self.patch_size)
        )
        self.val_degrees.update(degrees.mean(), weight=degrees.numel())
        self.log("val/loss", loss, prog_bar=False, batch_size=features.shape[0])
        self.log("val/degrees", self.val_degrees, prog_bar=True, on_epoch=True, batch_size=features.shape[0])

        if self.mask == "suffix":
            # Read off the same tensor the encoder was handed, so a baseline can never be
            # scored on frames the model did not have to predict.
            sin, cos = features[..., self.channels_output].unbind(-1)
            for name, error in velocity_baselines(t.atan2(sin, cos), self.horizon).items():
                metric = self.val_baselines[name]
                metric.update(error.mean(), weight=error.numel())
                self.log(f"val/{name}", metric, on_epoch=True, batch_size=features.shape[0])

        return loss

    def configure_optimizers(self):  # type: ignore
        # Freezing is two things: requires_grad=False stops the gradient, and leaving the
        # backbone out of the optimizer stops weight decay from shrinking weights that are
        # supposed to be fixed.
        optimizer = t.optim.AdamW(
            self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95)
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule(
                    optimizer,
                    warmup_steps=max(1, int(self.warmup * total_steps)),
                    min_factor=self.min_lr_frac,
                    total_steps=total_steps,
                ),
                "interval": "step",
            },
        }
