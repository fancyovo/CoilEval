"""Select flux quadrature using detached, owned formal-center references.

Only the quadrature choice is discrete. The selected flux coefficients are
always recomputed from live geometry and field tensors by the caller.
"""
from __future__ import annotations

import math
import torch


def stokes_check(flux, level, cache, tolerance):
    """Return a fallback decision and small, serializable diagnostics.

    A captured formal center supplies an area-integral polynomial and its
    surface level. Compare the edge flux and the derivative over 33 normalized
    flux positions. The derivative error is scaled by the largest reference
    derivative, avoiding singular relative errors near zero crossings.
    """
    info = dict(method='stokes', validation='disabled', fallback=False,
                tolerance=tolerance, edge_relative_error=None,
                derivative_relative_error=None)
    if tolerance is None:
        return False, info
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Stokes validation tolerance must be finite and positive')
    reference = cache.get('reference_flux_coefficients')
    reference_level = cache.get('reference_flux_level')
    if reference is None or reference_level is None:
        # Internal callers can still use snapshots predating formal references.
        # TensorEvaluator validates and supplies both fields for public calls.
        info['validation'] = 'reference_unavailable'
        return False, info
    with torch.no_grad():
        actual = flux['coefficients'].detach().double()
        reference = torch.as_tensor(reference, device=actual.device,
                                    dtype=torch.float64).detach()
        live_level = torch.as_tensor(level, device=actual.device,
                                     dtype=torch.float64).detach().reshape(())
        reference_level = torch.as_tensor(reference_level, device=actual.device,
                                          dtype=torch.float64).detach().reshape(())
        if actual.ndim != 1 or actual.shape != reference.shape:
            raise ValueError('formal flux polynomial shape differs from the live polynomial')
        if not bool(torch.isfinite(reference).all() & torch.isfinite(reference_level)
                    & (reference_level > 0)):
            raise ValueError('formal flux reference must be finite with a positive level')
        powers = torch.arange(1, actual.numel()+1, device=actual.device,
                              dtype=torch.float64)
        positions = torch.linspace(0, 1, 33, device=actual.device, dtype=torch.float64)
        derivative = ((actual*powers)[None, :]
                      * (live_level*positions[:, None]).pow(powers-1)).sum(-1)
        reference_derivative = ((reference*powers)[None, :]
                                * (reference_level*positions[:, None]).pow(powers-1)).sum(-1)
        reference_edge = (reference*reference_level.pow(powers)).sum()
        edge = flux['edge_flux'].detach().double()
        edge_error = (edge-reference_edge).abs()/reference_edge.abs().clamp_min(1e-30)
        derivative_error = ((derivative-reference_derivative).abs().max()
                            / reference_derivative.abs().max().clamp_min(1e-30))
        errors = torch.stack((edge_error, derivative_error)).cpu().tolist()
    bad = any(not math.isfinite(value) or value > tolerance for value in errors)
    info.update(method='area' if bad else 'stokes', validation='checked', fallback=bad,
                edge_relative_error=errors[0] if math.isfinite(errors[0]) else None,
                derivative_relative_error=errors[1] if math.isfinite(errors[1]) else None)
    return bad, info
