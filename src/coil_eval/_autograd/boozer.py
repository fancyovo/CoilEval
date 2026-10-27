"""Differentiable target-surface Boozer sampling from a live native fit.

Angles are radians.  The input poloidal angle is the negative geometric polar
angle, matching the native Clebsch fit.  Scalar root iterations run without a
tape and their converged implicit derivatives are attached explicitly.  The
magnetic axis, psi coefficients, target level, lambda, iota, nu and field values
all remain live; no frozen surface interpolation enters this path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from operator import index

import torch

from .downstream import _alpha_radial_values, clebsch_modes
from .angular_fourier import angular_fourier_values


@dataclass(frozen=True)
class ComposedAngularCorrection:
    """lambda_base(theta,phi)+delta(theta+lambda_base(theta,phi),phi)."""
    base: object
    correction: object


def _scalar(value, reference):
    result = torch.as_tensor(value, device=reference.device, dtype=torch.float64)
    if result.numel() != 1 or not bool(torch.isfinite(result).all()):
        raise ValueError('Boozer surface inputs require finite scalar parameters')
    return result.reshape(())


@lru_cache(maxsize=32)
def _angular_mode_map(radial_order, poloidal_order, toroidal_order, radial_basis):
    m, n, kind, radial = clebsch_modes(radial_order, poloidal_order,
                                     toroidal_order, radial_basis)
    angular = tuple(sorted(set(zip(m, n))))
    mapping = {mode: i for i, mode in enumerate(angular)}
    indices = tuple(2 * mapping[(mm, nn)] + kk for mm, nn, kk in zip(m, n, kind))
    return angular, indices, radial


def alpha_angular_coefficients(coefficients, config, *, rho=1.0):
    """Contract the native radial basis at one flux radius, preserving its VJP."""
    coefficients = coefficients.double()
    modes, indices, radial = _angular_mode_map(
        config.alpha_radial_order, config.alpha_poloidal_order,
        config.alpha_toroidal_order, config.alpha_radial_basis)
    if coefficients.ndim != 1 or coefficients.numel() != len(indices):
        raise ValueError('lambda coefficient count differs from the Clebsch basis')
    r = _scalar(rho, coefficients)
    basis = torch.as_tensor(radial, device=coefficients.device, dtype=torch.float64)
    values = _alpha_radial_values(r, basis, config.alpha_radial_basis)
    destination = torch.as_tensor(indices, device=coefficients.device, dtype=torch.long)
    collapsed = coefficients.new_zeros(2 * len(modes)).scatter_add(
        0, destination, coefficients * values).reshape(-1, 2)
    mode = coefficients.new_tensor(modes)
    return mode[:, 0], -config.nfp * mode[:, 1], collapsed[:, 0], collapsed[:, 1]


def fourier_values(data, theta, phi):
    """Value and both angular derivatives of a finite real Fourier series."""
    if isinstance(data, ComposedAngularCorrection):
        base, bt, bp = fourier_values(data.base, theta, phi)
        correction, ct, cp = fourier_values(data.correction, theta+base, phi)
        return base+correction, bt+ct*(1+bt), bp*(1+ct)+cp
    return angular_fourier_values(data, theta, phi)


def invert_alpha(data, theta_star, phi, *, max_iterations=30,
                 tolerance=1e-11, min_jacobian=1e-6):
    """Solve theta+lambda(theta,phi)=theta_star with its implicit derivative."""
    with torch.no_grad():
        theta = theta_star.clone()
        for iteration in range(max_iterations):
            lam, lt, _ = fourier_values(data, theta, phi)
            residual = theta + lam - theta_star
            jacobian = 1 + lt
            if not bool(torch.isfinite(jacobian).all() & (jacobian > min_jacobian).all()):
                raise ValueError('noninvertible alpha coordinate map')
            theta = theta - (residual / jacobian).clamp(-.5, .5)
            if float(residual.abs().max()) <= tolerance:
                break
        lam, lt, _ = fourier_values(data, theta, phi)
        residual = theta + lam - theta_star
        if not bool(torch.isfinite(residual).all()) or float(residual.abs().max()) > tolerance * 10:
            raise ValueError('alpha coordinate inversion did not converge')
    lam, lt, _ = fourier_values(data, theta, phi)
    attached = theta - (theta + lam - theta_star) / (1 + lt)
    return attached, dict(alpha_iterations=iteration + 1,
                         alpha_inversion_max_rad=residual.abs().max(),
                         alpha_jacobian_min=(1 + lt).amin())


def surface_geometry(geometry, level, theta, phi):
    """Cartesian surface and tangents in native negative-polar coordinates."""
    theta, phi = torch.broadcast_tensors(theta, phi)
    radius = geometry.solve_rays(geometry.rays(phi, -theta), level)
    ar, az, arp, azp = geometry.axis_eval(phi)
    ct, st, cp, sp = theta.cos(), theta.sin(), phi.cos(), phi.sin()
    R, Z = ar + radius * ct, az - radius * st
    _, gr, gz, gp = geometry.psi_eval(R, Z, phi)
    pr = gr * ct - gz * st
    if not bool(torch.isfinite(pr).all() & (pr > 0).all()):
        raise ValueError('nonpositive psi radial derivative on Boozer surface')
    rt = radius * (gr * st + gz * ct) / pr
    rp = -(gp + gr * arp + gz * azp) / pr
    Rt, Zt = rt * ct - radius * st, -rt * st - radius * ct
    Rp, Zp = arp + rp * ct, azp - rp * st
    xyz = torch.stack((R * cp, R * sp, Z), -1)
    xt = torch.stack((Rt * cp, Rt * sp, Zt), -1)
    xp = torch.stack((Rp * cp - R * sp, Rp * sp + R * cp, Zp), -1)
    return xyz, xt, xp


def straight_surface(geometry, level, alpha_data, theta_star, phi, **root_options):
    theta, diagnostics = invert_alpha(alpha_data, theta_star, phi, **root_options)
    xyz, xt, xp = surface_geometry(geometry, level, theta, phi)
    _, lt, lp = fourier_values(alpha_data, theta, phi)
    xt_star = xt / (1 + lt)[..., None]
    xp_star = xp - xt_star * lp[..., None]
    return xyz, xt_star, xp_star, diagnostics


def fit_nu(target, iota, nfp, order, *, resonance_tolerance=1e-10):
    """Solve (d_phi+iota*d_theta)nu=target on a uniform angular grid."""
    nphi, ntheta = target.shape
    kp = torch.fft.fftfreq(nphi, d=1 / nphi, device=target.device, dtype=target.dtype)
    kt = torch.fft.fftfreq(ntheta, d=1 / ntheta, device=target.device, dtype=target.dtype)
    k, m = torch.meshgrid(kp * nfp, kt, indexing='ij')
    band = (kp[:, None].abs() <= order) & (kt[None, :].abs() <= order)
    nonzero = (m != 0) | (k != 0)
    spectrum = torch.fft.fft2(target, norm='forward')
    divisor = k + iota * m
    active = band & nonzero & (divisor.detach().abs() > resonance_tolerance)
    safe_divisor = torch.where(active, divisor, torch.ones_like(divisor))
    coefficients = torch.where(active, spectrum / (1j * safe_divisor), torch.zeros_like(spectrum))
    represented = torch.fft.ifft2(coefficients * (1j * divisor), norm='forward').real
    resonance = band & nonzero & ~active
    # The input is real and the odd grid has no unpaired Nyquist modes. Each
    # conjugate pair contributes 2*real(c)*cos - 2*imag(c)*sin, so one canonical
    # half-plane evaluates the identical series with half the trigonometry.
    half = band & ((m > 0) | ((m == 0) & (k > 0)))
    data = (m[half], k[half], 2 * coefficients.real[half], -2 * coefficients.imag[half])
    diagnostics = dict(
        nu_transport_residual_rms=(represented - target).square().mean().sqrt(),
        nu_target_rms=target.square().mean().sqrt(),
        nu_resonant_mode_count=resonance.sum(),
        nu_resonant_spectrum_l2=spectrum[resonance].abs().square().sum().sqrt(),
        nu_min_nonzero_divisor=divisor[band & nonzero].abs().amin(),
        nu_omitted_spectrum_l2=spectrum[~(band & nonzero)].abs().square().sum().sqrt())
    return data, diagnostics


def invert_nu(data, theta_b, phi_b, iota, *, max_iterations=30,
              tolerance=1e-11, min_jacobian=1e-6):
    """Preserve alpha while solving phi+nu(theta_B+iota*(phi-phi_B),phi)=phi_B."""
    with torch.no_grad():
        phi = phi_b.clone()
        for iteration in range(max_iterations):
            theta_star = theta_b + iota * (phi - phi_b)
            nu, nt, np = fourier_values(data, theta_star, phi)
            residual = phi + nu - phi_b
            jacobian = 1 + np + iota * nt
            if not bool(torch.isfinite(jacobian).all() & (jacobian > min_jacobian).all()):
                raise ValueError('noninvertible nu coordinate map')
            phi = phi - (residual / jacobian).clamp(-.5, .5)
            if float(residual.abs().max()) <= tolerance:
                break
        theta_star = theta_b + iota * (phi - phi_b)
        nu, _, _ = fourier_values(data, theta_star, phi)
        residual = phi + nu - phi_b
        if not bool(torch.isfinite(residual).all()) or float(residual.abs().max()) > tolerance * 10:
            raise ValueError('nu coordinate inversion did not converge')
    theta_star = theta_b + iota * (phi - phi_b)
    nu, nt, np = fourier_values(data, theta_star, phi)
    jacobian = 1 + np + iota * nt
    attached_phi = phi - (phi + nu - phi_b) / jacobian
    attached_theta = theta_b + iota * (attached_phi - phi_b)
    return attached_theta, attached_phi, dict(
        nu_iterations=iteration + 1, nu_inversion_max_rad=residual.abs().max(),
        nu_jacobian_min=jacobian.amin())


def sample_boozer_surface(geometry, level, physics, field, config, *,
                          nphi=65, ntheta=67, nu_order=12,
                          max_iterations=30, tolerance=1e-11,
                          min_jacobian=1e-6, resonance_tolerance=1e-10,
                          field_derivatives=False, nu_method='transport',
                          alpha_order=0, alpha_ridge=1e-12,
                          alpha_coordinates='straight', alpha_fit_nphi=None,
                          alpha_fit_ntheta=None):
    """Sample |B| at uniform Boozer angles on the live target psi surface.

    This vacuum-field approximation has I=0.  It returns the coordinate and
    magnetic residuals separately from metric values; an approximate fit is not
    an independently certified invariant magnetic surface.  The sampler never
    damps lambda to manufacture an invertible map.
    """
    for value in (nphi, ntheta, nu_order, max_iterations):
        if isinstance(value, bool):
            raise ValueError('Boozer grid, order and iteration counts must be integers')
        index(value)
    if min(nphi, ntheta) < 5 or nphi % 2 == 0 or ntheta % 2 == 0:
        raise ValueError('Boozer sampling requires odd angular grids of size at least five')
    if nu_order < 1 or min(nphi, ntheta) <= 2 * nu_order or max_iterations < 1:
        raise ValueError('Boozer grid must resolve positive nu order and iteration count')
    if min(tolerance, min_jacobian, resonance_tolerance) <= 0:
        raise ValueError('Boozer root and resonance tolerances must be positive')
    level = level.double()
    alpha_data = physics.get('alpha_angular_data')
    if alpha_data is None:
        alpha_data = alpha_angular_coefficients(physics['alpha_coefficients'], config)
    iota = physics['iota_coefficients'].double().sum()
    phi = torch.arange(nphi, device=level.device, dtype=torch.float64) * (2 * math.pi / (config.nfp * nphi))
    theta = torch.arange(ntheta, device=level.device, dtype=torch.float64) * (2 * math.pi / ntheta)
    pp, tt = torch.meshgrid(phi, theta, indexing='ij')
    options = dict(max_iterations=max_iterations, tolerance=tolerance, min_jacobian=min_jacobian)
    alpha_diagnostics = {}
    if alpha_order:
        from .surface_alpha import refine_surface_alpha
        fit_nphi = max(nphi, 4*alpha_order+1) if alpha_fit_nphi is None else alpha_fit_nphi
        fit_ntheta = max(ntheta, 4*alpha_order+3) if alpha_fit_ntheta is None else alpha_fit_ntheta
        alpha_data, iota, alpha_diagnostics = refine_surface_alpha(
            geometry, level, alpha_data, iota, field, nfp=config.nfp, order=alpha_order,
            nphi=fit_nphi, ntheta=fit_ntheta, ridge=alpha_ridge, root_options=options,
            coordinates=alpha_coordinates)
    xyz, xt, xp, first_diagnostics = straight_surface(
        geometry, level, alpha_data, tt, pp, **options)
    B = field(xyz.reshape(-1, 3)).reshape_as(xyz).double()
    tangent = xp + iota * xt
    local_G = (B * tangent).sum(-1)
    G = local_G.mean()
    if not bool(torch.isfinite(G)) or float(G.detach().abs()) < 1e-20:
        raise ValueError('zero or nonfinite covariant G on Boozer surface')
    if nu_method == 'transport':
        nu_data, nu_diagnostics = fit_nu(local_G / G - 1, iota, config.nfp, nu_order,
                                       resonance_tolerance=resonance_tolerance)
    elif nu_method == 'potential':
        from .boozer_potential import fit_nu_potential
        nu_data, nu_diagnostics = fit_nu_potential((B*xt).sum(-1), (B*xp).sum(-1),
                                                   G, config.nfp, nu_order)
    else:
        raise ValueError('nu_method must be transport or potential')
    ts, po, inversion_diagnostics = invert_nu(nu_data, tt, pp, iota, **options)
    mapped, mt, mp, final_diagnostics = straight_surface(
        geometry, level, alpha_data, ts, po, **options)
    if field_derivatives:
        magnetic, field_jacobian = field(mapped.reshape(-1, 3), gradient=True)
        magnetic = magnetic.reshape_as(mapped).double()
        field_jacobian = field_jacobian.reshape(*mapped.shape[:-1], 3, 3).double()
    else:
        magnetic = field(mapped.reshape(-1, 3)).reshape_as(mapped).double()
    magnitude = torch.linalg.vector_norm(magnetic, dim=-1)
    if not bool(torch.isfinite(magnitude).all() & (magnitude > 0).all()):
        raise ValueError('nonfinite or zero magnetic field on Boozer surface')
    _, nt, np = fourier_values(nu_data, ts, po)
    along_jacobian = 1 + np + iota * nt
    if not bool((along_jacobian > min_jacobian).all()):
        raise ValueError('noninvertible final nu coordinate map')
    # Invert the full angular Jacobian, not only its along-field contraction.
    # iota is constant on this selected flux surface, but remains live in AD.
    theta_tangent = ((1 + np)[..., None] * mt - nt[..., None] * mp) / along_jacobian[..., None]
    phi_tangent = ((1 + iota * nt)[..., None] * mp - (iota * np)[..., None] * mt) / along_jacobian[..., None]
    boozer_tangent = phi_tangent + iota * theta_tangent
    equation = G * magnetic - magnitude.square()[..., None] * boozer_tangent
    normal = torch.linalg.cross(mt, mp)
    normal_length = torch.linalg.vector_norm(normal, dim=-1)
    if not bool((normal_length > 0).all()):
        raise ValueError('degenerate target-surface tangent basis')
    diagnostics = dict(
        **alpha_diagnostics, **first_diagnostics, **nu_diagnostics, **inversion_diagnostics,
        final_alpha_jacobian_min=final_diagnostics['alpha_jacobian_min'],
        final_alpha_inversion_max_rad=final_diagnostics['alpha_inversion_max_rad'],
        relative_boozer_equation_rms=(equation.square().sum(-1) /
                                    (G.square() * magnitude.square())).mean().sqrt(),
        relative_normal_B_rms=((magnetic * normal).sum(-1) /
                              (magnitude * normal_length)).square().mean().sqrt(),
        relative_covariant_theta_rms=((magnetic * theta_tangent).sum(-1) /
                                      G).square().mean().sqrt(),
        relative_covariant_phi_rms=(((magnetic * phi_tangent).sum(-1) - G) /
                                    G).square().mean().sqrt(),
        coordinate_method='implicit_alpha_nu', radial_location='target_surface',
        nphi=nphi, ntheta=ntheta, nu_order=nu_order, nu_method=nu_method)
    result = dict(B_magnitude=magnitude, xyz=mapped, B=magnetic,
                iota=iota, G=G, I=G * 0, major_radius=physics['major_radius'],
                theta_b=tt, phi_b=pp, x_theta=theta_tangent, x_phi=phi_tangent,
                diagnostics=diagnostics)
    if field_derivatives:
        if not bool(torch.isfinite(field_jacobian).all()):
            raise ValueError('nonfinite magnetic-field Jacobian on Boozer surface')
        grad_magnitude = torch.einsum('...ij,...i->...j', field_jacobian, magnetic) / magnitude[..., None]
        result.update(dBdx=field_jacobian,
                      parallel_derivative=(magnetic * grad_magnitude).sum(-1))
    return result
