
import torch as t
import torch.nn.functional as f
import torch.nn as nn

class MeanPooling(nn.Module):
    """Standard mean pooling over the sequence dimension."""
    def __init__(self, in_dim: int, **_) -> None:
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.mean(dim=1)


class MeanMaxPooling(nn.Module):
    """Concatenates Mean and Max pooling."""
    def __init__(self, in_dim: int, **_) -> None:
        super().__init__()
        self.out_dim = in_dim * 2

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_mean, x_max = x.mean(dim=1), x.max(dim=1).values
        return t.cat([x_mean, x_max], dim=-1)


class SoftAttentivePooling(nn.Module):
    """Soft/Attentive Pooling via a small scoring network."""
    def __init__(self, in_dim: int, bottleneck_dim: int = 128, **_) -> None:
        super().__init__()
        self.out_dim = in_dim
        self.score_net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1, bias=False)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        # x: (B, S, C) where S = T * D
        logits = self.score_net(x).squeeze(-1) # (B, S)
        w = f.softmax(logits, dim=-1) # (B, S)
        # Multiply weights (B, S) with features (B, S, C) -> sum over S -> (B, C)
        return t.einsum("bs,bsc->bc", w, x)


class FactorizedAttentivePooling(nn.Module):
    """Pool the DOFs of each frame, then the frames.

    One score net per axis rather than one over all ``T * D`` tokens: attention over a
    flat grid has to learn that a token's two coordinates mean different things, and the
    factorization says so instead.
    """

    def __init__(self, in_dim: int, num_d: int, bottleneck_dim: int = 128, **_) -> None:
        super().__init__()

        self.out_dim = in_dim
        self.num_d = num_d

        self.t_score_net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim), nn.Tanh(), nn.Linear(bottleneck_dim, 1, bias=False)
        )
        self.s_score_net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim), nn.Tanh(), nn.Linear(bottleneck_dim, 1, bias=False)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        # T comes from the tokens, not from the spec: a window shorter than the encoder's
        # full length is fewer time patches over the same DOFs.
        B, TD, C = x.shape
        pooled = x.view(B, TD // self.num_d, self.num_d, C)

        # Spatial reduction: aggregate DoFs across space -> (B, T, C)
        s_logits = self.s_score_net(pooled).squeeze(-1) # (B, T, D)
        s_w = f.softmax(s_logits, dim=-1) # (B, T, D)
        pooled = t.einsum("btd,btdc->btc", s_w, pooled)

        # 2. Temporal reduction: aggregate frames across time -> (B, C)
        t_logits = self.t_score_net(pooled).squeeze(-1) # (B, T)
        t_w = f.softmax(t_logits, dim=-1) # (B, T)
        pooled = t.einsum("bt,btc->bc", t_w, pooled)

        return pooled


def get_pooler(pool_type: str, in_dim: int, num_d: int) -> nn.Module:
    pooler = {
        "attentive_factorized": FactorizedAttentivePooling,
        "attentive": SoftAttentivePooling,
        "mean_max": MeanMaxPooling,
        "mean": MeanPooling
    }.get(pool_type)

    if pooler is None:
        raise ValueError(f"Unknown pool type '{pool_type}'.")

    return pooler(in_dim=in_dim, num_d=num_d)
