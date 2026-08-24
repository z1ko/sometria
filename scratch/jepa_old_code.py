
from copy import deepcopy

import torch as t
import torch.nn as nn

from omegaconf import DictConfig

class PositionalEncoding(nn.Module):
    def __init__(self, segment_count: int, group_count: int, embed_dim: int):
        super().__init__()

        self.segment_count = segment_count
        self.group_count = group_count
        self.embed_dim = embed_dim

        # Temporal encoding
        self.encode_t = nn.Parameter(t.zeros(1, self.segment_count, 1, self.embed_dim))
        nn.init.trunc_normal_(self.encode_t, std=0.02)

        # Group encoding
        self.encode_g = nn.Parameter(t.zeros(1, 1, self.group_count, self.embed_dim))
        nn.init.trunc_normal_(self.encode_g,  std=0.02)

    def grid(self) -> t.Tensor:
        return self.encode_t + self.encode_g # 1, S, G, E

    def flat(self) -> t.Tensor:
        return self.grid().flatten(1, 2) # 1, SG, E

    def add_flat(self, x: t.Tensor, idx: t.Tensor | None = None) -> t.Tensor:
        if idx is not None:
            return x + self.gather(idx)
        return x + self.flat()
    
    def gather(self, idx: t.Tensor) -> t.Tensor:
        idx = idx.to(device=self.encode_t.device, dtype=t.long)
        pos = self.flat().expand(idx.shape[0], -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        return pos.gather(dim=1, index=idx)

class TokenizeGroups(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        # Store group indices
        self.groups = config.training.groups
        for i, group in enumerate(self.groups):
            self.register_buffer(
                f"group_indices_{i}", 
                t.tensor(self.groups[group], dtype=t.long)
            )

    def forward(self, x: t.Tensor) -> dict[str, t.Tensor]:
        """
        x: tensor of shape ..., D, C
        returns:
            dictionary of groups, each one is a tensor of shape ..., G(i), C
        """

        *_, D, C = x.shape

        groups: dict[str, t.Tensor] = {}
        for i, group in enumerate(self.groups.keys()):

            index: t.Tensor = getattr(self, f"group_indices_{i}")
            if index.max() >= D or index.min() < 0:
                raise ValueError(f"Group {group!r} has indices outside input D={D}")
            
            x_group = x.index_select(dim=-2, index=index)
            groups[group] = x_group

        return groups
    
class TokenizeSegments(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.segment_size = config.architecture.segment_size

    def forward(self, x: t.Tensor) -> t.Tensor:

        B, T, D, C = x.shape
        if T % self.segment_size != 0:
            raise ValueError(
                f"Input tensor of shape {x.shape} is not compatible with segment_size={self.segment_size}"
            )
        
        segment_count = T // self.segment_size
        x = x.reshape(B, segment_count, self.segment_size, D, C)
        return x


@t.no_grad()
def compute_motion_intensity(x: t.Tensor, tokenize_t: TokenizeSegments, tokenize_g: TokenizeGroups, channel: str = "vel") -> t.Tensor:
    """ (B, T, D, C) -> (B, segment_count, group_count) sum-of-|vel| per token """
    velocity = x[..., CHANNELS.index(channel)].unsqueeze(-1)
    velocity_groups: dict[str, t.Tensor] = tokenize_g(tokenize_t(velocity))
    return t.stack([
        group.abs().sum(dim=(2, 3, 4)) for group in velocity_groups.values()
    ], dim=2)


class TokenEmbed(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.embed_dim = config.architecture.embed_dim
        self.segment_size = config.architecture.segment_size
        self.channels = config.architecture.channels
        self.groups = config.training.groups

        # Tokenizers
        self.tokenize_t = TokenizeSegments(config)
        self.tokenize_g = TokenizeGroups(config)

        # Common embedding dimension projections
        self.projections = nn.ModuleDict({
            group: nn.Linear(
                self.segment_size * len(indices) * self.channels,
                self.embed_dim
            ) for group, indices in self.groups.items()
        })

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        # Tokenize input motion
        groups = self.tokenize_g(self.tokenize_t(x))

        # Embed to common dimension
        results: list[t.Tensor] = []
        for group in self.groups.keys():
            x_group = groups[group]

            B, S, T, G, C = x_group.shape
            x_group = x_group.reshape(B, S, -1) # B, S, TGC
            x_group = self.projections[group](x_group) # B, S, E
            results.append(x_group)

        return t.stack(results, dim=2) # B, S, G, E

# Standard transformer encoder
def _encoder(d_model: int, depth: int, heads: int, mlp_ratio: float, dropout: float) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ),
        num_layers=depth
    )

class MotionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.group_count = len(config.training.groups)
        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.embed_dim = config.architecture.embed_dim

        self.embed = TokenEmbed(config)

        # Positional encoding
        self.pos = PositionalEncoding(
            segment_count=self.segment_count,
            group_count=self.group_count,
            embed_dim=self.embed_dim
        )

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.encoder = _encoder(
            d_model=self.embed_dim,
            depth=config.architecture.encoder.depth,
            heads=config.architecture.encoder.heads,
            mlp_ratio=config.architecture.encoder.mlp_ratio,
            dropout=config.architecture.encoder.dropout,
        )
        

    def forward_full_and_gather(
        self,
        x: t.Tensor,
        idx: t.Tensor,
        key_padding_mask: t.Tensor | None = None,
    ) -> t.Tensor:
        # Full (ungathered) mask: the teacher attends over every token
        # (including any padding) before the caller gathers target rows,
        # so padding must be excluded from attention here, not after.
        x = self.forward(x, idx=None, key_padding_mask=key_padding_mask)
        return x.gather(
            index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
            dim=1
        )

    def forward(
        self,
        x: t.Tensor,
        idx: t.Tensor | None = None,
        key_padding_mask: t.Tensor | None = None,
    ) -> t.Tensor:

        tokens = self.embed(x)
        tokens = tokens.flatten(1, 2) # B, SG, E

        if idx is not None:
            # Gather only tokens present in idx
            tokens = tokens.gather(
                index=idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
                dim=1
            )
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.gather(dim=1, index=idx)

        tokens = self.pos.add_flat(tokens, idx)
        x = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(x)
        

class MotionPredictor(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.group_count = len(config.training.groups)
        self.pred_dim = config.architecture.predictor.inner_dim
        self.embed_dim = config.architecture.embed_dim

        self.mask_token = nn.Parameter(t.zeros(1, 1, self.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.i_proj = nn.Linear(self.embed_dim, self.pred_dim)
        self.o_proj = nn.Linear(self.pred_dim, self.embed_dim)

        self.pos = PositionalEncoding(
            segment_count=self.segment_count,
            group_count=self.group_count,
            embed_dim=self.pred_dim
        )

        self.norm = nn.LayerNorm(self.pred_dim, eps=1e-6)
        self.encoder = _encoder(
            d_model=config.architecture.predictor.inner_dim,
            depth=config.architecture.predictor.depth, 
            heads=config.architecture.predictor.heads, 
            mlp_ratio=config.architecture.predictor.mlp_ratio, 
            dropout=config.architecture.predictor.dropout
        )

    def forward(
        self,
        context: t.Tensor,
        context_idx: t.Tensor,
        targets_idx: t.Tensor,
        key_padding_mask: t.Tensor | None = None,
    ) -> t.Tensor:
        B, A = targets_idx.shape

        x_context = self.i_proj(context)
        x_context = self.pos.add_flat(x_context, context_idx)

        x_targets = self.mask_token.expand(B, A, -1)
        x_targets = self.pos.add_flat(x_targets, targets_idx)

        x = t.cat([x_context, x_targets], dim=1)

        full_mask = None
        if key_padding_mask is not None:
            context_mask = key_padding_mask.gather(dim=1, index=context_idx)
            targets_mask = key_padding_mask.gather(dim=1, index=targets_idx)
            full_mask = t.cat([context_mask, targets_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=full_mask)

        y = self.norm(x[:, -A:, :])
        return self.o_proj(y)
    

class MotionJEPA(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.group_count = len(config.training.groups)
        self.channels = config.architecture.channels

        # Dedicated instances for compute_motion_intensity (masking.py's "mamp"
        # strategy) -- decoupled from student_encoder.embed's own tokenizers so
        # this stays reusable (e.g. scripts/viz_masks.py) without needing a full model.
        self._motion_tokenize_t = TokenizeSegments(config)
        self._motion_tokenize_g = TokenizeGroups(config)
        
        self.mamp_target_fraction = config.masking.mamp_target_fraction
        self.mamp_temperature = config.masking.mamp_temperature

        self.student_encoder = MotionEncoder(config)
        self.teacher_encoder = deepcopy(self.student_encoder)
        self._freeze_teacher_encoder()

        self.predictor = MotionPredictor(config)        

    def _freeze_teacher_encoder(self):
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    @t.no_grad()
    def update_teacher(self, ema_momentum: float) -> None:
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters(),
            strict=True,
        ):
            teacher_param.data.mul_(ema_momentum).add_(
                student_param.data,
                alpha=1.0 - ema_momentum,
            )

    def forward(
        self,
        x: t.Tensor,
        valid_segments: t.Tensor,
        masks: MaskIndices | None = None,
    ) -> tuple[t.Tensor, t.Tensor]:

        batch_size = x.shape[0]

        # config.architecture.channels selects how many of the trailing
        # CHANNELS (pos, vel, acc, tau -- see utils.py) the model actually
        # sees, keeping their existing order: 4 is everything, 3 drops tau,
        # etc. Only correct because tau is last in CHANNELS -- dropping a
        # non-trailing channel would need an index_select, not a slice.
        x = x[..., :self.channels]

        # A segment is valid only if every frame in it is real (non-padded);
        # all groups at a given time-segment share that segment's validity,
        # since token_index = s*group_count + g (see masking.py/components.py).
        segment_valid = t.arange(self.segment_count, device=x.device).unsqueeze(0) < valid_segments.unsqueeze(1)
        token_valid = segment_valid.unsqueeze(-1).expand(-1, -1, self.group_count)  # (B, S, G), matches masking.py's scores shape
        key_padding_mask = ~token_valid.flatten(1, 2)  # (B, S*G), matches the flattened token axis used for gather/attention

        # Generate random masks if not provided. mask_mamp only, for now
        # (ablation: train with MAMP-style motion-aware masking exclusively,
        # not blended with the other 4 strategies via mask_mixed).
        if masks is None:
            motion_intensity = compute_motion_intensity(x, self._motion_tokenize_t, self._motion_tokenize_g)
            masks = mask_mamp(
                batch_size=batch_size,
                segment_count=self.segment_count,
                group_count=self.group_count,
                device=x.device,
                motion_intensity=motion_intensity,
                target_fraction=self.mamp_target_fraction,
                temperature=self.mamp_temperature,
                token_valid=token_valid,
            ) # type: ignore

        context = self.student_encoder.forward(x, masks.context, key_padding_mask=key_padding_mask) # B, SG, E
        predict = self.predictor.forward(
            context=context,
            context_idx=masks.context,
            targets_idx=masks.targets,
            key_padding_mask=key_padding_mask,
        )

        with t.no_grad():
            targets = self.teacher_encoder.forward_full_and_gather(
                x, masks.targets, key_padding_mask=key_padding_mask
            )

        return predict, targets