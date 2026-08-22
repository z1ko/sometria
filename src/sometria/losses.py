
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import roots_jacobi

class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()

        t = T.linspace(0, 3, knots, dtype=T.float32)
        dt = 3 / (knots - 1)

        weights = T.full((knots,), 2 * dt, dtype=T.float32)
        weights[[0, -1]] = dt
        window = T.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):

        A = T.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()