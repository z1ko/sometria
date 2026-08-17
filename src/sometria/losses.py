
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import roots_jacobi

#class EppsPulley(nn.Module):
#    """
#    1-D Epps-Pulley goodness-of-fit statistic for N(0, 1).
#
#    Input:
#        x: [N, S]
#           N samples
#           S random projections
#
#    Output:
#        [S] statistic, one per projection.
#    """
#
#    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
#        super().__init__()
#
#        assert n_points % 2 == 1
#
#        # Quadrature points.
#        t = T.linspace(0.0, t_max, n_points)
#        dt = t_max / (n_points - 1)
#
#        # We integrate only over [0, t_max].
#        # Factor 2 exploits symmetry
#        weights = T.full((n_points,), 2.0 * dt)
#        weights[ 0] = dt
#        weights[-1] = dt
#
#        # Characteristic function of N(0, 1): exp(-t^2 / 2)
#        phi = T.exp(-0.5 * t.square())
#
#        self.register_buffer("phi", phi)
#        self.register_buffer("t", t)
#
#        # Epps-Pulley weighting function.
#        self.register_buffer("weights", weights * phi)
#
#
#    def forward(self, x: T.Tensor) -> T.Tensor:
#        N = x.shape[0]
#
#        xt = x.unsqueeze(-1) * self.t
#
#        real = xt.cos().mean(dim=0)
#        imag = xt.sin().mean(dim=0)
#
#        error = (real - self.phi).square() + imag.square()
#        statistic = (error @ self.weight) * N
#        return statistic
#
#
#class SIGReg(nn.Module):
#    """
#    Sketched Isotropic Gaussian Regularization.
#
#    Forces z ~ N(0, I) by testing many random 1-D
#    projections.
#
#    Input:
#        z: [B, D]
#           or
#           [V, B, D]
#
#    Returns:
#        scalar
#    """
#
#    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17) -> None:
#        super().__init__()
#
#        self.num_slices = num_slices
#        self.ep = EppsPulley(
#            n_points=n_points, 
#            t_max=t_max)
#        
#    def _single_view(self, z: T.Tensor) -> T.Tensor:
#        D = z.shape[-1]
#
#        with T.no_grad():
#            A = T.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
#            A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-9)
#
#        projections = z @ A
#        return self.ep(projections).mean()
#
#    def forward(self, z: T.Tensor):
#        if z.ndim == 2:
#            return self._single_view(z)
#
#        if z.ndim != 3:
#            raise ValueError(
#                "Expected [B,D] or [V,B,D]"
#            )
#
#        # Regularize each augmentation/view distribution.
#        losses = [ self._single_view(view) for view in z ]
#        return T.stack(losses).mean()
#
#
#class SphereMMD(nn.Module):
#    def __init__(self, dim, q=64):
#        super().__init__()
#        a = (dim - 3) / 2
#        x, w = roots_jacobi(q, a, a)
#
#        x = T.tensor(x, dtype=T.float32)
#        w = T.tensor(w, dtype=T.float32)
#        w = w / w.sum()
#
#        self.register_buffer("x2", x.square())
#        self.register_buffer("w", w)
#
#        with T.no_grad():
#            c = (
#                T.exp(
#                    -(1 - x[:, None]) * x.square()[None]
#                )
#                * w[None]
#            ).sum(-1)
#
#            bias = (w * c).sum()
#
#        self.register_buffer("bias", bias)
#
#    def kernel(self, cos):
#        return (
#            T.exp(
#                -(1 - cos[..., None]) * self.x2
#            )
#            * self.w
#        ).sum(-1)
#
#    def forward(self, z):
#        z = F.normalize(z.float(), dim=-1)
#        K = self.kernel((z @ z.T).clamp(-1, 1))
#        return (K.mean() - self.bias) / (1 - self.bias)
#    
#
#class SphereJEPALoss(nn.Module):
#    def __init__(self, dim=256, lam=0.05):
#        super().__init__()
#        self.lam = lam
#        self.mmd = SphereMMD(dim)
#
#    def forward(self, global_z, local_z=None):
#        global_z = F.normalize(global_z, dim=-1)
#
#        if local_z is not None:
#            local_z = F.normalize(local_z, dim=-1)
#            z = T.cat([global_z, local_z], dim=0)
#        else:
#            z = global_z
#
#        target = global_z.mean(0)
#
#        inv = (
#            z - target.unsqueeze(0)
#        ).square().sum(-1).mean()
#
#        reg = T.stack([
#            self.mmd(view)
#            for view in z
#        ]).mean()
#
#        loss = (1 - self.lam) * inv + self.lam * reg
#
#        return loss