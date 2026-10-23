"""Active geometric metric derivatives."""
import numpy as np
import torch
from .. import _binding as native

def split_tokens(tokens):
    return tokens[..., :9], tokens[..., 9:18], tokens[..., 18:27], tokens[..., 27]

class _EngineeringMetrics(torch.autograd.Function):
    """Native active-pair/active-curvature exact analytical derivatives."""
    @staticmethod
    def forward(ctx, physical_tokens, library_path, nfp):
        host = physical_tokens.detach().double().cpu().numpy()
        out = native.coil_metric_gradient_native(library_path, *split_tokens(host), nfp)
        # Native axes: metric, coordinate, coil, Fourier coefficient.
        deriv = np.asarray(out['derivatives']).transpose(0, 2, 1, 3).reshape(6, 4, 27)
        full = np.zeros((3, 4, 28), dtype=np.float64)
        # Return gap, max curvature, mean length in loss order.
        full[:, :, :27] = deriv[[3, 2, 0]]
        ctx.save_for_backward(torch.as_tensor(full, device=physical_tokens.device, dtype=physical_tokens.dtype))
        names = ('min_intercoil_distance', 'curvature_max', 'length_mean')
        return physical_tokens.new_tensor([out['values'][name] for name in names])

    @staticmethod
    def backward(ctx, adj_metrics):
        (jacobian,) = ctx.saved_tensors
        return torch.einsum('m,mci->ci', adj_metrics, jacobian), None, None
