"""Active geometric metric derivatives."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch
from .. import _binding as native

def split_tokens(tokens):
    if tokens.ndim<1:
        raise ValueError('tokens require three odd Fourier coefficient blocks and current')
    width=tokens.shape[-1]
    coefficients=(width-1)//3
    if coefficients<1 or width!=3*coefficients+1 or coefficients%2!=1:
        raise ValueError('tokens require three odd Fourier coefficient blocks and current')
    return (tokens[...,:coefficients],tokens[...,coefficients:2*coefficients],
            tokens[...,2*coefficients:3*coefficients],tokens[...,-1])

class _EngineeringMetrics(torch.autograd.Function):
    """Native active-pair/active-curvature exact analytical derivatives."""
    @staticmethod
    def forward(ctx, physical_tokens, library_path, nfp):
        host = physical_tokens.detach().double().cpu().numpy()
        if host.ndim!=2 or host.shape[0]<1:
            raise ValueError('engineering metrics require nonempty [coil,3*ncoeff+1] tokens')
        coils,width=host.shape
        coefficients=(width-1)//3
        out = native.coil_metric_gradient_native(library_path, *split_tokens(host), nfp)
        # Native axes: metric, coordinate, coil, Fourier coefficient.
        deriv = np.asarray(out['derivatives']).transpose(0, 2, 1, 3).reshape(6, coils, 3*coefficients)
        full = np.zeros((3, coils, width), dtype=np.float64)
        # Return gap, max curvature, mean length in loss order.
        full[:, :, :-1] = deriv[[3, 2, 0]]
        ctx.save_for_backward(torch.as_tensor(full, device=physical_tokens.device, dtype=physical_tokens.dtype))
        names = ('min_intercoil_distance', 'curvature_max', 'length_mean')
        return physical_tokens.new_tensor([out['values'][name] for name in names])

    @staticmethod
    def backward(ctx, adj_metrics):
        (jacobian,) = ctx.saved_tensors
        return torch.einsum('m,mci->ci', adj_metrics, jacobian), None, None


def _capture_engineering_batch(host, library_path, nfp):
    """CPU-only native work; keep values and derivatives in one upload buffer."""
    queries, coils, width = host.shape
    coefficients = (width-1)//3
    packed = np.zeros((queries, 3, 1+coils*width), dtype=np.float64)
    names = ('min_intercoil_distance', 'curvature_max', 'length_mean')
    for query, tokens in enumerate(host):
        out = native.coil_metric_gradient_native(library_path, *split_tokens(tokens), nfp)
        packed[query, :, 0] = [out['values'][name] for name in names]
        deriv = np.asarray(out['derivatives']).transpose(0, 2, 1, 3).reshape(6, coils, 3*coefficients)
        full = packed[query, :, 1:].reshape(3, coils, width)
        full[:, :, :-1] = deriv[[3, 2, 0]]
    return packed


class _PreparedEngineering(torch.autograd.Function):
    """Attach the independently computed native Jacobians to physical tokens."""
    @staticmethod
    def forward(ctx, physical_tokens, packed):
        uploaded = torch.as_tensor(packed, device=physical_tokens.device, dtype=physical_tokens.dtype)
        jacobian = uploaded[:, :, 1:].reshape(len(physical_tokens), 3, *physical_tokens.shape[1:])
        ctx.save_for_backward(jacobian)
        return uploaded[:, :, 0]

    @staticmethod
    def backward(ctx, adj_metrics):
        (jacobian,) = ctx.saved_tensors
        return torch.einsum('bm,bmci->bci', adj_metrics, jacobian), None


class _EngineeringBatchWork:
    def __init__(self, future, shape):
        self._future, self._shape = future, shape

    def tensor(self, physical_tokens):
        """Join on the caller, then perform the sole device upload there."""
        if tuple(physical_tokens.shape) != self._shape:
            raise ValueError('prepared engineering tokens must match the captured batch shape')
        if physical_tokens.dtype not in (torch.float32, torch.float64):
            raise ValueError('prepared engineering tokens require FP32 or FP64')
        return _PreparedEngineering.apply(physical_tokens, self._future.result())


@contextmanager
def prepare_engineering_batch(host, library_path, nfp=4):
    """Overlap one owned CPU worker with the caller's GPU stages.

    ``host`` must be the same physical snapshot used by ``work.tensor(xs)``.
    The worker never calls Torch or CUDA. A private contiguous copy protects it
    from caller mutation, and context exit joins it even if another stage fails.
    Existing scalar/CPU engineering behavior is retained by _EngineeringMetrics.
    """
    snapshot = np.array(host, dtype=np.float64, order='C', copy=True)
    if snapshot.ndim != 3 or min(snapshot.shape[:2]) < 1:
        raise ValueError('engineering batch requires nonempty [query,coil,3*ncoeff+1] tokens')
    split_tokens(snapshot)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='coil-eval-engineering') as executor:
        future = executor.submit(_capture_engineering_batch, snapshot, library_path, nfp)
        yield _EngineeringBatchWork(future, snapshot.shape)
