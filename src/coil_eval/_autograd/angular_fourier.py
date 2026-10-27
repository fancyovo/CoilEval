"""Bounded-memory real angular Fourier evaluation and first derivatives."""
import torch
from torch.autograd.function import once_differentiable


# A tile holds at most 256 MiB per FP64 feature array. Larger tiles reduce
# launch overhead while compact saved state keeps the reverse memory bounded.
_MAX_FEATURE_ELEMENTS = 2**25
_MAX_MODE_TILE = 4096


class _AngularFourier(torch.autograd.Function):
    """Save angles and coefficients; recompute trigonometric tiles in reverse."""
    @staticmethod
    def forward(ctx, m, k, cosine, sine, theta, phi):
        ctx.save_for_backward(m, k, cosine, sine, theta, phi)
        ctx.set_materialize_grads(False)
        shape = theta.shape
        th, ph = theta.reshape(-1), phi.reshape(-1)
        modes = len(m)
        mode_tile = max(1, min(modes, _MAX_MODE_TILE))
        point_tile = max(1, _MAX_FEATURE_ELEMENTS//mode_tile)
        output = theta.new_zeros((th.numel(), 3))
        for first in range(0, modes, mode_tile):
            last = min(first+mode_tile, modes)
            mm, kk, cc, ss = (v[first:last] for v in (m, k, cosine, sine))
            cw = torch.stack((cc, ss*mm, ss*kk), -1)
            sw = torch.stack((ss, -cc*mm, -cc*kk), -1)
            for start in range(0, th.numel(), point_tile):
                end = min(start+point_tile, th.numel())
                phase = th[start:end, None]*mm + ph[start:end, None]*kk
                output[start:end] += phase.cos()@cw + phase.sin()@sw
        return tuple(output[:, column].reshape(shape) for column in range(3))

    @staticmethod
    @once_differentiable
    def backward(ctx, adj_value, adj_theta, adj_phi):
        m, k, cosine, sine, theta, phi = ctx.saved_tensors
        th, ph = theta.reshape(-1), phi.reshape(-1)
        adjoints = torch.stack([theta.new_zeros(th.numel()) if value is None else
                                value.reshape(-1) for value in (adj_value, adj_theta, adj_phi)], -1)
        dc = torch.zeros_like(cosine) if ctx.needs_input_grad[2] else None
        ds = torch.zeros_like(sine) if ctx.needs_input_grad[3] else None
        dt = torch.zeros_like(th) if ctx.needs_input_grad[4] else None
        dp = torch.zeros_like(ph) if ctx.needs_input_grad[5] else None
        modes = len(m)
        mode_tile = max(1, min(modes, _MAX_MODE_TILE))
        point_tile = max(1, _MAX_FEATURE_ELEMENTS//mode_tile)
        for first in range(0, modes, mode_tile):
            last = min(first+mode_tile, modes)
            mm, kk, cc, ss = (v[first:last] for v in (m, k, cosine, sine))
            if dt is not None or dp is not None:
                cw = torch.stack((ss*mm, ss*kk, cc*mm*mm, cc*mm*kk, cc*kk*kk), -1)
                sw = torch.stack((-cc*mm, -cc*kk, ss*mm*mm, ss*mm*kk, ss*kk*kk), -1)
            for start in range(0, th.numel(), point_tile):
                end = min(start+point_tile, th.numel())
                phase = th[start:end, None]*mm + ph[start:end, None]*kk
                co, si = phase.cos(), phase.sin()
                adj = adjoints[start:end]
                if dc is not None or ds is not None:
                    co_adj, si_adj = co.T@adj, si.T@adj
                    if dc is not None:
                        dc[first:last] += co_adj[:, 0]-mm*si_adj[:, 1]-kk*si_adj[:, 2]
                    if ds is not None:
                        ds[first:last] += si_adj[:, 0]+mm*co_adj[:, 1]+kk*co_adj[:, 2]
                if dt is not None or dp is not None:
                    moment = co@cw+si@sw
                    if dt is not None:
                        dt[start:end] += adj[:, 0]*moment[:, 0]-adj[:, 1]*moment[:, 2]-adj[:, 2]*moment[:, 3]
                    if dp is not None:
                        dp[start:end] += adj[:, 0]*moment[:, 1]-adj[:, 1]*moment[:, 3]-adj[:, 2]*moment[:, 4]
        return None, None, dc, ds, (dt.reshape(theta.shape) if dt is not None else None), (dp.reshape(phi.shape) if dp is not None else None)


def angular_fourier_values(data, theta, phi):
    """Return value, d/dtheta and d/dphi with compact first-order AD state.

    Integer-frequency modes are fixed metadata. Coefficients and both query
    angles remain differentiable. The reverse evaluates the exact finite
    series in bounded tiles, including second angular derivatives needed by
    cotangents of the derivative outputs; it does not approximate the series.
    """
    m, k, cosine, sine = data
    theta, phi = torch.broadcast_tensors(theta, phi)
    dtype = theta.dtype
    for value in (phi, m, k, cosine, sine):
        dtype = torch.promote_types(dtype, value.dtype)
    values = [value.to(dtype=dtype) for value in (m, k, cosine, sine, theta, phi)]
    return _AngularFourier.apply(*values)


class _TransportDesign(torch.autograd.Function):
    """Fourier transport matrix; only beta depends on physical parameters."""
    @staticmethod
    def forward(ctx, beta, m, k, theta, phi):
        if any(value.requires_grad for value in (m, k, theta, phi)):
            raise ValueError('transport design requires a fixed angular grid and frequencies')
        ctx.save_for_backward(m, k, theta, phi)
        modes, points = len(m), beta.numel()
        mode_tile = max(1, min(modes, _MAX_MODE_TILE))
        point_tile = max(1, _MAX_FEATURE_ELEMENTS//mode_tile)
        design = beta.new_empty((points, 2*modes+1))
        design[:, -1] = -1
        for first in range(0, modes, mode_tile):
            last = min(first+mode_tile, modes)
            mm, kk = m[first:last], k[first:last]
            for start in range(0, points, point_tile):
                end = min(start+point_tile, points)
                phase = theta[start:end, None]*mm + phi[start:end, None]*kk
                frequency = kk + beta[start:end, None]*mm
                design[start:end, first:last] = -frequency*phase.sin()
                design[start:end, modes+first:modes+last] = frequency*phase.cos()
        return design

    @staticmethod
    @once_differentiable
    def backward(ctx, adjoint):
        m, k, theta, phi = ctx.saved_tensors
        modes, points = len(m), theta.numel()
        mode_tile = max(1, min(modes, _MAX_MODE_TILE))
        point_tile = max(1, _MAX_FEATURE_ELEMENTS//mode_tile)
        beta_gradient = theta.new_zeros(points)
        for first in range(0, modes, mode_tile):
            last = min(first+mode_tile, modes)
            mm, kk = m[first:last], k[first:last]
            for start in range(0, points, point_tile):
                end = min(start+point_tile, points)
                phase = theta[start:end, None]*mm + phi[start:end, None]*kk
                beta_gradient[start:end] += ((
                    -phase.sin()*adjoint[start:end, first:last]
                    +phase.cos()*adjoint[start:end, modes+first:modes+last])*mm).sum(-1)
        return beta_gradient, None, None, None, None


def transport_fourier_design(beta, m, k, theta, phi):
    """Build [−(k+mβ)sin, (k+mβ)cos, −1] with a compact exact beta VJP."""
    return _TransportDesign.apply(beta, m, k, theta, phi)
