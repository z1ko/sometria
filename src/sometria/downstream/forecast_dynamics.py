"""Can a frozen backbone forecast joint torque, and infer it from motion it has seen?

The rig of :mod:`~sometria.downstream.reconstruction` -- frozen backbone in ``eval()``, a
fresh ``tinydec`` trained on top, scored at the final epoch -- asked two questions:

``mode="forecast"``
    The encoder sees patches ``0..21``; the decoder fills ``22..29`` (1.067 s) on all five
    channels, ``sin, cos, vel, acc, tau``, under one joint MSE in model space.

``mode="infer"``
    The encoder sees patches ``0..22``, with the ``tau`` channel of patch 22 overwritten;
    the decoder reads ``tau`` out of patch 22 alone. Torque at ``t+1`` given the motion at
    ``t+1`` and everything before it: causal inverse dynamics.

Patch 22 is forecast's first horizon patch, so the two modes differ in one thing --
whether the kinematics of that patch are visible -- and the gap between them is the gap
between *forecasting* torque and *inferring* it.

**Hiding tau** only does anything for encoders that read it (``in_pkd``). A token embeds
all of its channels together, so tau cannot be dropped from a token whose kinematics stay;
it can only be overwritten, and no pretraining run ever saw a token with one channel
overwritten. ``fill="zero"`` writes 0, the per-DOF training mean in z-space;
``fill="hold"`` repeats the last observed torque. Both are off-distribution in different
ways, and the gap between them bounds how much of a ``pkd`` number is that artifact.

**Scores** are per channel and per horizon patch:

* R^2 in model space, centred per DOF on the validation set's own mean and weighted by
  each DOF's variance (``1 - sum SSE / sum SST``). Centred per DOF because ``sin, cos``
  are not normalized, so pooling would count the between-DOF spread as explained;
  variance-weighted because a near-constant ``cos`` has an SST near zero and a uniform
  average of per-DOF R^2 was measured at -897 on it.
* Degrees on ``atan2(sin, cos)``, so the angle number sits beside the published
  reconstruction table as a sanity check.
* Torque error in N m/kg: ``signed_exp`` of the denormalized channel, divided by the
  subject's mass from the catalog metadata, as the **median** absolute error per DOF,
  averaged over DOFs. Not the mean: AMASS torque is heavy-tailed (median 0.04 N m/kg, max
  67 on babel_official val) and ``signed_exp`` turns a modest overshoot in model space
  into thousands of N m/kg, so a mean ranked predictors by their worst frame -- the linear
  floor scored worse than predicting the mean. Per DOF so the many near-zero wrist and
  finger torques do not drown the joints that carry load.

**Floors** are computed in the same validation loop on the same windows: hold the last
frame, constant-velocity extrapolation in model space, the training mean (z = 0), and two
least-squares maps fit on the train split (:func:`fit_linear`) -- a linear inverse
dynamics from same-frame kinematics of every DOF, and the same map given the last observed
torque. In forecast mode the first is fed the *true* future kinematics, so it answers "how
good could torque be if the future motion were known".
"""

from collections import defaultdict

import lightning as L
import torch as t
import torch.nn.functional as F

from sometria.architecture.scheduler import lr_schedule
from sometria.downstream.reconstruction import DEC_DEPTH, DEC_DIM, angular_error, suffix_indices
from sometria.models.baseline import Decoder, gather_tokens

#: The forecast boundary, in time patches: context ``0..21``, horizon ``22..29``. The same
#: split as reconstruction's suffix mask (30 - 8), so the two benchmarks share a horizon.
CONTEXT_PATCHES: int = 22

#: Target channel names, in this order. Resolved to indices through ``Representation``.
CHANNELS: tuple[str, ...] = ("sin", "cos", "vel", "acc", "tau")
KINEMATICS: tuple[str, ...] = CHANNELS[:4]

MODES = ("forecast", "infer")
FILLS = ("zero", "hold")


def frames(tokens: t.Tensor, dofs: int, patch_size: int) -> t.Tensor:
    """``(B, P * dofs, patch_size * C)`` tokens -> ``(B, P, patch_size, dofs, C)``.

    The inverse of :func:`~sometria.models.baseline.patchify` plus its flatten: tokens run
    ``t * dofs + d`` and a token's values run ``frame * C + channel``.
    """

    batch = tokens.shape[0]
    return tokens.view(batch, -1, dofs, patch_size, tokens.shape[-1] // patch_size).transpose(2, 3)


def window(features: t.Tensor, start: int, end: int, patch_size: int) -> t.Tensor:
    """Frames ``start..end`` of ``(B, T, D, C)`` as ``(B, P, patch_size, D, C)``."""

    chunk = features[:, start:end]
    return chunk.view(chunk.shape[0], -1, patch_size, *chunk.shape[2:])


def hide_tau(features: t.Tensor, tau: int, start: int, end: int, fill: str) -> t.Tensor:
    """Overwrite channel ``tau`` on frames ``start..end``, leaving everything else alone."""

    out = features.clone()
    if fill == "zero":
        out[:, start:end, :, tau] = 0.0
    elif fill == "hold":
        out[:, start:end, :, tau] = features[:, start - 1 : start, :, tau]
    else:
        raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
    return out


def signed_exp(x: t.Tensor) -> t.Tensor:
    """Torch twin of :func:`sometria.representation.signed_exp`."""

    return x.sign() * x.abs().expm1()


def affine(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """``x @ W[:-1] + W[-1]``: a least-squares map whose last row is the bias."""

    return x @ weights[:-1].to(x.dtype) + weights[-1].to(x.dtype)


@t.no_grad()
def fit_linear(
    loader, kinematics: tuple[int, ...], tau: int, boundary: int, patch_size: int, device: str, ridge: float = 1e-4
) -> dict[str, t.Tensor]:
    """The two linear floors, by normal equations over one pass of ``loader``.

    ``id``      kinematics of every DOF at frame f -> tau of every DOF at frame f, on every
                frame. Torque couples across joints (a hip carries the leg below it), so
                the map sees all DOFs, not one.
    ``id_last`` the same plus tau at frame ``boundary - 1``, on the frames of the target
                patch only -- the fair floor for an encoder that has torque history.

    Model space in, model space out. Accumulated in float64 so one pass suffices; the ridge
    is relative to the mean diagonal and only there to keep the solve well posed.
    """

    sums = {name: [0.0, 0.0] for name in ("id", "id_last")}

    def add(name: str, x: t.Tensor, y: t.Tensor) -> None:
        x = t.cat([x, t.ones_like(x[:, :1])], dim=1).double()
        sums[name][0] = sums[name][0] + x.T @ x
        sums[name][1] = sums[name][1] + x.T @ y.double()

    for batch in loader:
        features = batch["features"].to(device)
        kin = features[..., kinematics].flatten(2)                    # (B, T, D*4)
        torque = features[..., tau]                                   # (B, T, D)
        add("id", kin.flatten(0, 1), torque.flatten(0, 1))

        target = slice(boundary, boundary + patch_size)
        last = torque[:, boundary - 1 : boundary].expand(-1, patch_size, -1)
        add("id_last", t.cat([kin[:, target], last], dim=-1).flatten(0, 1), torque[:, target].flatten(0, 1))

    weights = {}
    for name, (xtx, xty) in sums.items():
        eye = t.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
        weights[name] = t.linalg.solve(xtx + ridge * xtx.diagonal().mean() * eye, xty).float()
    return weights


class Scores:
    """R^2, degrees and N m/kg per predictor, accumulated over a validation epoch.

    Every predictor hands in a ``{channel: (B, P, patch, D)}`` dict for whichever channels
    it predicts; targets are observed once per batch and shared, so a floor can never be
    scored on frames the model did not have to predict.

    ponytail: plain tensors on one device rather than torchmetrics; add a sync if this is
    ever run multi-GPU.
    """

    def __init__(self, tau_mean: t.Tensor, tau_std: t.Tensor) -> None:
        self.tau_mean, self.tau_std = tau_mean, tau_std
        self.reset()

    def reset(self) -> None:
        self.sum_y: dict = defaultdict(float)
        self.sum_y2: dict = defaultdict(float)
        self.count: dict = defaultdict(float)
        self.sse: dict = defaultdict(float)
        self.degrees: dict = defaultdict(float)                       # name -> (P,)
        self.degrees_count: dict = defaultdict(float)
        self.nmkg: dict = defaultdict(list)                           # name -> [(P, M, D)]

    def newton_per_kg(self, z: t.Tensor, mass: t.Tensor) -> t.Tensor:
        return signed_exp(z * self.tau_std + self.tau_mean) / mass.view(-1, 1, 1, 1)

    def observe(self, target: dict[str, t.Tensor], mass: t.Tensor) -> None:
        self.target, self.mass = target, mass
        for channel, y in target.items():
            y = y.double()
            self.sum_y[channel] = self.sum_y[channel] + y.sum((0, 2))    # (P, D)
            self.sum_y2[channel] = self.sum_y2[channel] + (y**2).sum((0, 2))
            self.count[channel] = self.count[channel] + y.shape[0] * y.shape[2]

    def update(self, name: str, prediction: dict[str, t.Tensor]) -> None:
        for channel, p in prediction.items():
            residual = (p - self.target[channel]).double()
            self.sse[name, channel] = self.sse[name, channel] + (residual**2).sum((0, 2))

        if "tau" in prediction:
            error = (
                self.newton_per_kg(prediction["tau"].float(), self.mass)
                - self.newton_per_kg(self.target["tau"].float(), self.mass)
            ).abs()
            # Kept whole, on the CPU: a median does not accumulate. ~30 MB per predictor.
            self.nmkg[name].append(error.transpose(0, 1).flatten(1, 2).cpu())
        if "sin" in prediction and "cos" in prediction:
            error = angular_error(
                t.atan2(prediction["sin"], prediction["cos"]).float(),
                t.atan2(self.target["sin"], self.target["cos"]).float(),
            )
            self.degrees[name] = self.degrees[name] + error.double().sum((0, 2, 3))
            self.degrees_count[name] = self.degrees_count[name] + error[:, 0].numel()

    def compute(self) -> dict:
        """``{predictor: {"r2": {channel: [per patch]}, "degrees": [...], "tau_nmkg": [...]}}``.

        See the module docstring for why R^2 is variance-weighted and N m/kg is a median.
        """

        out: dict = defaultdict(lambda: defaultdict(dict))
        for (name, channel), sse in self.sse.items():
            sst = self.sum_y2[channel] - self.sum_y[channel] ** 2 / self.count[channel]
            out[name]["r2"][channel] = (1.0 - sse.sum(1) / sst.sum(1)).tolist()
        for name, total in self.degrees.items():
            out[name]["degrees"] = (total / self.degrees_count[name]).tolist()
        for name, errors in self.nmkg.items():
            per_dof = t.cat(errors, dim=1).median(dim=1).values          # (P, D)
            out[name]["tau_nmkg"] = per_dof.nanmean(dim=1).tolist()
        return {name: dict(value) for name, value in out.items()}


class DynamicsForecastProbe(L.LightningModule):
    """A frozen backbone, one ``tinydec``, scored on torque and its kinematic neighbours.

    ``channels`` is ``Representation.indices(*CHANNELS)``. ``linear`` is :func:`fit_linear`'s
    output and ``tau_stats`` the ``(mean, std)`` of the tau channel per DOF, both from the
    train split; they are inputs to the floors and the N m/kg conversion, never trained.
    """

    def __init__(
        self,
        backbone: L.LightningModule,
        channels: tuple[int, ...],
        linear: dict[str, t.Tensor],
        tau_stats: tuple[t.Tensor, t.Tensor],
        *,
        mode: str = "forecast",
        fill: str = "zero",
        context_patches: int = CONTEXT_PATCHES,
        dec_depth: int = DEC_DEPTH,
        dec_dim: int = DEC_DIM,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup: float = 0.05,
        min_lr_frac: float = 0.5,
    ) -> None:
        super().__init__()

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
        if len(channels) != len(CHANNELS):
            raise ValueError(f"pass Representation.indices{CHANNELS}, got {channels}")

        self.save_hyperparameters(ignore=["backbone", "linear", "tau_stats"])

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        hp = backbone.hparams
        self.channels_input = list(hp.channels_input)
        self.patch_size = int(hp.num_frames_in_patch)
        self.num_frames = int(hp.num_frames)
        self.num_dofs = int(hp.num_dofs)
        self.time_patches = self.num_frames // self.patch_size

        self.mode, self.fill = mode, fill
        self.kinematics = list(channels[:4])
        self.tau = channels[4]
        # Forecast predicts every channel; infer sees the kinematics, so only torque is left.
        self.outputs = CHANNELS if mode == "forecast" else ("tau",)
        self.channels_output = list(channels) if mode == "forecast" else [self.tau]
        self.boundary = context_patches * self.patch_size

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.min_lr_frac = min_lr_frac

        self.decoder = Decoder(
            self.num_dofs,
            self.patch_size,
            self.time_patches,
            tuple(self.channels_output),
            int(hp.dim),
            dec_depth,
            int(hp.num_heads),
            int(hp.mlp_ratio),
            dec_dim,
        )

        # forecast: context 0..21, targets 22..29. infer: the encoder sees 0..22 and the
        # target is patch 22 -- the last patch of its own context.
        visible = context_patches if mode == "forecast" else context_patches + 1
        context, targets = suffix_indices(self.time_patches, self.num_dofs, self.time_patches - visible)
        if mode == "infer":
            targets = context[:, -self.num_dofs :]
        self.register_buffer("context", context, persistent=False)
        self.register_buffer("targets", targets, persistent=False)
        self.end = self.num_frames if mode == "forecast" else self.boundary + self.patch_size

        for name, weights in linear.items():
            self.register_buffer(f"linear_{name}", weights, persistent=False)
        self.register_buffer("tau_mean", tau_stats[0], persistent=False)
        self.register_buffer("tau_std", tau_stats[1], persistent=False)
        self.scores: Scores | None = None
        self.results: dict = {}

    def train(self, mode: bool = True) -> "DynamicsForecastProbe":
        """Keep the backbone in eval mode; see ``MotionReconstructionProbe.train``."""

        super().train(mode)
        self.backbone.eval()
        return self

    def encoder(self):
        # JEPA is probed through its EMA teacher, as everywhere else in downstream.
        return getattr(self.backbone, "encoder_teacher", None) or self.backbone.encoder

    def forward(self, features: t.Tensor) -> t.Tensor:
        """``(B, T, D, C)`` -> predicted target frames, ``(B, P, patch, D, outputs)``."""

        if features.shape[1] != self.num_frames:
            raise ValueError(f"backbone was pretrained on {self.num_frames}-frame windows, got {features.shape[1]}")

        batch = features.shape[0]
        context = self.context.expand(batch, -1)
        if self.mode == "infer":
            features = hide_tau(features, self.tau, self.boundary, self.end, self.fill)

        with t.no_grad():
            tokens = self.encoder()(features[..., self.channels_input], context, None)
        prediction = gather_tokens(self.decoder(tokens, context), self.targets.expand(batch, -1))
        return frames(prediction, self.num_dofs, self.patch_size)

    def target(self, features: t.Tensor) -> t.Tensor:
        return window(features[..., self.channels_output], self.boundary, self.end, self.patch_size)

    def training_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        loss = F.mse_loss(self(batch["features"]), self.target(batch["features"]))
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["features"].shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.scores = Scores(self.tau_mean, self.tau_std)

    def validation_step(self, batch: dict, batch_idx: int) -> t.Tensor:
        features = batch["features"]
        prediction, target = self(features), self.target(features)
        loss = F.mse_loss(prediction, target)
        self.log("val/loss", loss, batch_size=features.shape[0])

        scores = self.scores
        scores.observe(dict(zip(self.outputs, target.unbind(-1))), batch["mass"].to(features))
        scores.update("model", dict(zip(self.outputs, prediction.unbind(-1))))
        for name, floor in self.floors(features).items():
            scores.update(name, floor)
        return loss

    def floors(self, features: t.Tensor) -> dict[str, dict[str, t.Tensor]]:
        """The model-free and linear predictions, each ``{channel: (B, P, patch, D)}``."""

        horizon = window(features, self.boundary, self.end, self.patch_size)   # (B, P, p, D, C)
        last = features[:, self.boundary - 1]                                   # (B, D, C)
        kin = horizon[..., self.kinematics].flatten(-2)                         # (B, P, p, D*4)
        tau_last = last[..., self.tau]
        floors: dict[str, dict[str, t.Tensor]] = {}

        if self.mode == "forecast":
            step = t.arange(1, self.end - self.boundary + 1, device=features.device, dtype=features.dtype)
            step = step.view(1, *horizon.shape[1:3], 1, 1)
            hold = last[:, None, None].expand_as(horizon)
            const = last[:, None, None] + (last - features[:, self.boundary - 2])[:, None, None] * step
            channels = list(zip(CHANNELS, (*self.kinematics, self.tau)))
            floors["hold"] = {name: hold[..., i] for name, i in channels}
            floors["const_velocity"] = {name: const[..., i] for name, i in channels}
            # z = 0 is the training mean only where the channel is normalized.
            floors["mean"] = {name: t.zeros_like(hold[..., i]) for name, i in channels[2:]}
            floors["linear_id_future"] = {"tau": affine(kin, self.linear_id)}
        else:
            floors["hold"] = {"tau": tau_last[:, None, None].expand_as(horizon[..., self.tau])}
            floors["mean"] = {"tau": t.zeros_like(horizon[..., self.tau])}
            floors["linear_id"] = {"tau": affine(kin, self.linear_id)}
            with_last = t.cat([kin, tau_last[:, None, None].expand(*kin.shape[:3], -1)], dim=-1)
            floors["linear_id_last"] = {"tau": affine(with_last, self.linear_id_last)}
        return floors

    def on_validation_epoch_end(self) -> None:
        self.results = self.scores.compute()
        tau = self.results["model"]["r2"]["tau"]
        self.log("val/r2_tau", sum(tau) / len(tau), prog_bar=True)
        self.log("val/r2_tau_p1", tau[0])

    def configure_optimizers(self):  # type: ignore
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
