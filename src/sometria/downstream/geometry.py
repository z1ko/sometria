"""Geometry of the pooled features the convex probe fits: how many directions they carry,
and whether two backbones carry the same ones.

Both statistics are read off the ``d x d`` second-moment matrix rather than the ``n x n``
gram. The probe's features are tall and thin -- roughly 10k windows by 256 (or 4x that
under ``moments``) -- so the feature-space form is the identical number at 1/1600 of the
memory, and the gram form is the one every CKA reference writes only because vision
papers compare layers of different widths. Nothing here does.

This is new code. CKA existed in this repo as a stored number in
`results/decoder_capacity/evidence.json` and nothing else -- no gram matrix, no HSIC, no
rank statistic in any module -- so the 0.934 cross-cell against 0.959 cross-seed reading
on that page is the only prior art, and it is a number without a function behind it.
"""

import torch as t


def participation_ratio(feats: t.Tensor) -> float:
    """``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the feature covariance.

    Reads as "how many directions the features effectively use": ``d`` when the covariance
    is isotropic, 1 when it is rank-1, and the interpolation in between is variance-weighted
    rather than a rank count, which is the point -- a covariance whose 200th eigenvalue is
    1e-9 has full rank and one useful direction.

    No eigendecomposition. For a symmetric covariance ``sum lambda = tr(C)`` and
    ``sum lambda^2 = ||C||_F^2`` exactly, so ``eigh`` would compute 256 eigenvalues in
    order to discard all but two symmetric functions of them.

    ``float64`` throughout: the features are LayerNorm output, so the covariance diagonal
    is O(1) and its tail is not, and the ratio is a difference of near-equal sums once a
    cell's spectrum concentrates -- which is exactly the near-rank-1 case this is measured
    to detect.
    """

    x = feats.double()
    x = x - x.mean(dim=0)
    covariance = x.T @ x / (len(x) - 1)
    return float(covariance.trace().square() / covariance.square().sum())


def linear_cka(x: t.Tensor, y: t.Tensor) -> float:
    """Linear CKA between two row-aligned feature matrices, in ``[0, 1]``.

    ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)`` on column-centred inputs, which is
    algebraically the centred-gram HSIC form and numerically the sane one here: the gram
    form builds an ``n x n`` matrix (10k windows is 800 MB in float64) to extract a
    quantity that lives in ``d x d``.

    Invariant to isotropic scaling and to orthogonal rotation of either side, which is why
    it is the measurement `results/decoder_capacity/README.md` reaches for after weight
    distance said nothing: two transformers can compute nearly the same function with
    maximally distant weights, and a representation comparison has to be blind to exactly
    those symmetries.

    Row ``i`` of ``x`` and row ``i`` of ``y`` must be the same window. This function cannot
    check that -- the caller matches on ``sample_id``.
    """

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA needs row-aligned matrices, got {tuple(x.shape)} and {tuple(y.shape)}")

    x, y = x.double(), y.double()
    x, y = x - x.mean(dim=0), y - y.mean(dim=0)
    # (y.T @ x) is d_y x d_x, never n x n. The two denominators are self-similarities, so
    # CKA(x, x) is 1 by construction rather than by luck.
    return float((y.T @ x).square().sum() / ((x.T @ x).norm() * (y.T @ y).norm()))
