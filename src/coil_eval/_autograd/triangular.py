"""Independent triangular systems with differentiable device-local solves."""
from __future__ import annotations

import torch


def independent_triangular_solve(matrix, rhs, *, upper):
    """Solve each fitted system independently, retaining its complete VJP.

    CUDA 12.8 with PyTorch 2.7.1 produced incorrect results for transposed
    batched factors at B=16/32 on RTX 5090. Individual solves passed the same
    factor, RHS, and gradient controls. Dispatch independently for every batch
    size so correctness does not depend on a backend-specific size threshold.
    Field evaluation and the surrounding matrix products remain batched.
    """
    if matrix.ndim == 2:
        return torch.linalg.solve_triangular(matrix, rhs, upper=upper)
    if (matrix.ndim != 3 or rhs.ndim != 3 or matrix.shape[0] != rhs.shape[0]
            or matrix.shape[-1] != matrix.shape[-2]
            or rhs.shape[-2] != matrix.shape[-1] or not matrix.shape[0]):
        raise ValueError('independent triangular systems require matching nonempty [query,n,n] and [query,n,k] arrays')
    return torch.stack([torch.linalg.solve_triangular(a, b, upper=upper)
                        for a, b in zip(matrix.unbind(0), rhs.unbind(0))])
