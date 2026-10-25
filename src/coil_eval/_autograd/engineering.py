"""Active geometric metric derivatives."""
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
