"""Target-surface straight-angle correction in an existing angular chart."""
from functools import lru_cache
import math

import torch
from torch.autograd.function import once_differentiable


class _RidgeLeastSquares(torch.autograd.Function):
    """FP64 normal solve with an O(rows*columns) first-derivative contraction."""
    @staticmethod
    def forward(ctx, design, right, ridge):
        gram = design.T@design
        gram.diagonal().add_(ridge)
        factor, pivots = torch.linalg.lu_factor(gram)
        solution = torch.linalg.lu_solve(factor, pivots, (design.T@right)[:, None])[:, 0]
        residual = right-design@solution
        ctx.save_for_backward(design, solution, residual, factor, pivots)
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, adjoint):
        design, solution, residual, factor, pivots = ctx.saved_tensors
        v = torch.linalg.lu_solve(factor, pivots, adjoint[:, None], adjoint=True)[:, 0]
        projected = design@v
        d_design = (residual[:, None]*v[None, :] - projected[:, None]*solution[None, :]
                    if ctx.needs_input_grad[0] else None)
        return d_design, projected if ctx.needs_input_grad[1] else None, None


@lru_cache(maxsize=16)
def _angular_modes(order):
    # One signed mode per real conjugate pair, with no constant lambda gauge.
    return tuple((m, n) for m in range(order+1)
                 for n in (range(1, order+1) if m == 0 else range(-order, order+1)))


def refine_surface_alpha(geometry, level, alpha_data, iota, field, *, nfp,
                         order, nphi, ntheta, ridge=1e-12, root_options=None,
                         coordinates='straight'):
    """Solve for delta(theta_star,phi) and iota without moving the psi surface.

    B is projected onto the live surface tangent basis, giving beta=B^theta /
    B^phi. The Fourier fit solves ``delta_phi + beta*delta_theta - iota=-beta``.
    Its zero-mean gauge and explicit column-scaled ridge define a local solve.
    In ``straight`` coordinates the result composes delta with the existing
    angle exactly. In ``geometric`` coordinates it replaces that correction
    on this surface. Neither route projects onto the original radial basis.
    """
    from .boozer import ComposedAngularCorrection, straight_surface, surface_geometry
    from .angular_fourier import angular_fourier_values, transport_fourier_design
    from .least_squares import ridge_least_squares
    if order < 1 or min(nphi, ntheta) <= 2*order or not math.isfinite(ridge) or ridge < 0:
        raise ValueError('invalid target-alpha fit order, grid or regularization')
    phi = torch.arange(nphi, device=level.device, dtype=torch.float64)*2*math.pi/(nfp*nphi)
    theta = torch.arange(ntheta, device=level.device, dtype=torch.float64)*2*math.pi/ntheta
    pp, tt = torch.meshgrid(phi, theta, indexing='ij')
    if coordinates == 'straight':
        xyz, xt, xp, original_diagnostics = straight_surface(
            geometry, level, alpha_data, tt, pp, **(root_options or {}))
    elif coordinates == 'geometric':
        xyz, xt, xp = surface_geometry(geometry, level, tt, pp)
        original_diagnostics = {'alpha_jacobian_min': level*0+1}
    else:
        raise ValueError('target-alpha coordinates must be straight or geometric')
    magnetic = field(xyz.reshape(-1, 3)).reshape_as(xyz).double()
    gtt, gtp, gpp = (xt*xt).sum(-1), (xt*xp).sum(-1), (xp*xp).sum(-1)
    determinant = gtt*gpp-gtp.square()
    if not bool(torch.isfinite(determinant).all() & (determinant > 0).all()):
        raise ValueError('degenerate tangent metric in target-alpha fit')
    bt, bp = (magnetic*xt).sum(-1), (magnetic*xp).sum(-1)
    contravariant_theta = (bt*gpp-bp*gtp)/determinant
    contravariant_phi = (bp*gtt-bt*gtp)/determinant
    if (not bool(torch.isfinite(contravariant_phi).all())
            or bool((contravariant_phi.abs() < 1e-12).any())
            or not bool((contravariant_phi > 0).all() | (contravariant_phi < 0).all())):
        raise ValueError('toroidal field vanishes or reverses in target-alpha fit')
    beta = (contravariant_theta/contravariant_phi).reshape(-1)
    modes = level.new_tensor(_angular_modes(order))
    m, k = modes[:, 0], nfp*modes[:, 1]
    design = transport_fourier_design(beta, m, k, tt.reshape(-1), pp.reshape(-1))
    scales = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-14)
    scaled = design/scales
    scaled_solution, solver_diagnostics = ridge_least_squares(scaled, -beta, ridge)
    solution = scaled_solution/scales
    c, s, fitted_iota = solution[:len(m)], solution[len(m):2*len(m)], solution[-1]
    # Diagnostics do not contribute to metric derivatives. Keep their matrix
    # contractions off the tape; the fitted coefficients and iota stay live.
    with torch.no_grad():
        residual = design@solution+beta
        correction_jacobian = 1+angular_fourier_values((m, k, c, s), tt, pp)[1]
    threshold = (root_options or {}).get('min_jacobian', 1e-6)
    if not bool(torch.isfinite(solution).all() & (correction_jacobian > threshold).all()):
        raise ValueError('noninvertible target-alpha correction')
    correction = (ComposedAngularCorrection(alpha_data, (m, k, c, s))
                  if coordinates == 'straight' else (m, k, c, s))
    with torch.no_grad():
        rhs_norm = (scaled.T@beta).norm().clamp_min(1e-30)
        diagnostics = dict(
            surface_alpha_order=order, surface_alpha_coordinates=coordinates,
            surface_alpha_fit_nphi=nphi, surface_alpha_fit_ntheta=ntheta,
            surface_alpha_columns=design.shape[-1], surface_alpha_ridge=ridge,
            surface_alpha_solver=solver_diagnostics,
            surface_alpha_previous_iota=iota.detach(), surface_alpha_iota=fitted_iota.detach(),
            surface_alpha_transport_residual_rms=residual.square().mean().sqrt(),
            surface_alpha_transport_relative_rms=residual.norm()/beta.norm().clamp_min(1e-30),
            surface_alpha_correction_jacobian_min=correction_jacobian.amin(),
            surface_alpha_base_jacobian_min=original_diagnostics['alpha_jacobian_min'].detach(),
            surface_alpha_stationarity_relative=(scaled.T@residual+ridge*scaled_solution).norm()/rhs_norm,
            surface_alpha_ridge_rhs_relative=(ridge*scaled_solution).norm()/rhs_norm)
    return correction, fitted_iota, diagnostics
