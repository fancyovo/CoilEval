"""Double-precision column norms without a double copy of the full matrix."""
import torch


class _ColumnNorms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, chunk):
        # FP64 conversion and its product coexist in backward. Bound each
        # temporary by element count, including all independent queries.
        rows = matrix.numel() // matrix.shape[-1]
        chunk = min(chunk, max(1, (1 << 22) // max(1, rows)))
        norms = torch.empty((*matrix.shape[:-2], matrix.shape[-1]),
                            dtype=torch.float64, device=matrix.device)
        for start in range(0, matrix.shape[-1], chunk):
            stop = min(start + chunk, matrix.shape[-1])
            norms[..., start:stop] = torch.linalg.vector_norm(
                matrix[..., start:stop], dim=-2, dtype=torch.float64)
        ctx.save_for_backward(matrix, norms)
        ctx.chunk = chunk
        return norms

    @staticmethod
    def backward(ctx, adjoint):
        matrix, norms = ctx.saved_tensors
        gradient = torch.empty_like(matrix)
        for start in range(0, matrix.shape[-1], ctx.chunk):
            stop = min(start + ctx.chunk, matrix.shape[-1])
            norm = norms[..., start:stop]
            factor = torch.where(norm > 0, adjoint[..., start:stop] / norm, 0.)
            gradient[..., start:stop] = (
                matrix[..., start:stop].double() * factor.unsqueeze(-2)).to(matrix.dtype)
        return gradient, None


def column_norms(matrix, *, chunk=128):
    """Return FP64 L2 norms along the row axis of a matrix or matrix batch."""
    if matrix.ndim < 2 or chunk < 1:
        raise ValueError('column norms require a matrix and a positive chunk size')
    return _ColumnNorms.apply(matrix, chunk)
