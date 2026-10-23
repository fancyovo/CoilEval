"""Differentiable surface, flux, alpha and QS metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Callable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DownstreamConfig:
    nfp: int = 4
    axis_sample_count: int = 240
    surface_theta_count: int = 128
    surface_trace_steps: int = 100
    flux_level_count: int = 11
    flux_phi_count: int = 8
    flux_theta_count: int = 256
    flux_radial_quadrature: int = 24
    flux_polynomial_degree: int = 4
    volume_point_count: int = 100000
    volume_phi_count: int = 96
    volume_theta_count: int = 32
    volume_rho_min: float = .08
    alpha_fit_point_count: int = 30000
    alpha_radial_order: int = 12
    alpha_poloidal_order: int = 12
    alpha_toroidal_order: int = 12
    iota_degree: int = 3
    radial_bin_count: int = 10
    alpha_ridge: float = 1e-7
    alpha_iterations: int = 4
    alpha_chunk_columns: int = 128
    target_M: int = 1
    target_N: int = 4
    target_quality_threshold: float = .01
    target_quality_temperature: float = .0025
    target_quality_weight: float = 10000.
    stokes_flux: bool = True

    @classmethod
    def from_mapping(cls, source: Mapping):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in source.items() if k in names})


def _arange(n: int, ref: Tensor):
    return torch.arange(n, device=ref.device, dtype=ref.dtype)


def _xyz(R, Z, phi):
    return torch.stack((R * phi.cos(), R * phi.sin(), Z), -1)


def polynomial(coefficients: Tensor, x: Tensor, *, constant=False):
    """Evaluate coefficients in ascending order, with optional constant term."""
    result = torch.zeros_like(x)
    for c in coefficients.flip(-1).unbind(-1):
        result = result * x + c
    return result if constant else result * x


def flux_calibration(geometry, level: Tensor, field: Callable,
                     potential: Callable | None, config: DownstreamConfig):
    """Full 8 x 11 x 256 calibration; signed flux/(2*pi) as in native."""
    c = config
    phi = _arange(c.flux_phi_count, level) * (2 * math.pi / (c.nfp * c.flux_phi_count))
    theta = _arange(c.flux_theta_count, level) * (2 * math.pi / c.flux_theta_count)
    pp, tt = torch.meshgrid(phi, theta, indexing='ij')
    ray = geometry.rays(pp, tt)
    fractions = (_arange(c.flux_level_count, level) + 1) / c.flux_level_count
    levels = level * fractions.square()
    radii = geometry.solve_rays(ray[:, None, :, :], levels[None, :, None])
    ap, az, _, _ = geometry.axis_eval(pp)
    ct, st = tt.cos()[:, None, :], tt.sin()[:, None, :]
    R = ap[:, None, :] + radii * ct
    Z = az[:, None, :] + radii * st
    ph = pp[:, None, :].expand_as(R)
    if c.stokes_flux:
        if potential is None:
            raise ValueError('Stokes flux requires a differentiable vector potential callback')
        _, gr, gz, _ = geometry.psi_eval(R, Z, ph)
        pr = gr * ct + gz * st
        # Native Stokes also rejects nonpositive psi_r, rather than flooring it.
        if not bool(torch.all(torch.isfinite(pr) & (pr > 0)).detach()):
            raise ValueError('nonpositive/nonfinite radial derivative on Stokes boundary')
        rt = radii * (gr * st - gz * ct) / pr
        Rt, Zt = rt * ct - radii * st, rt * st + radii * ct
        points = _xyz(R, Z, ph)
        tangent = _xyz(Rt, Zt, ph)
        A = potential(points.reshape(-1, 3)).reshape_as(points)
        sections = -(A.double() * tangent.double()).sum(-1).mean(-1)
    else:
        nodes_np, weights_np = np.polynomial.legendre.leggauss(c.flux_radial_quadrature)
        nodes = torch.as_tensor(nodes_np, device=level.device, dtype=level.dtype)
        weights = torch.as_tensor(weights_np, device=level.device, dtype=level.dtype)
        r = .5 * radii[..., None] * (nodes + 1)
        rw = .5 * radii[..., None] * weights
        RR = ap[:, None, :, None] + r * ct[..., None]
        ZZ = az[:, None, :, None] + r * st[..., None]
        ph4 = ph[..., None].expand_as(RR)
        points = _xyz(RR, ZZ, ph4)
        B = field(points.reshape(-1, 3)).reshape_as(points)
        bp = -B[..., 0] * ph4.sin() + B[..., 1] * ph4.cos()
        sections = (bp.double() * r.double() * rw.double()).sum(-1).mean(-1)
    mean = sections.mean(0)
    # Normalized polynomial coordinates avoid the badly scaled normal matrix
    # at small levels; this is algebraically the native no-constant LS problem.
    powers = torch.arange(1, c.flux_polynomial_degree + 1, device=level.device)
    t = fractions.double().square()
    matrix = t[:, None].pow(powers)
    normalized_coeff = torch.linalg.solve(matrix.T @ matrix, matrix.T @ mean)
    coeff = (normalized_coeff / level.double().pow(powers)).to(level.dtype)
    edge = normalized_coeff.sum().to(level.dtype)
    boundary = radii[:, -1, :]
    volume = (2 * math.pi) ** 2 * (
        .5 * ap.double() * boundary.double().square()
        + tt.cos().double() * boundary.double().pow(3) / 3
    ).mean()
    return dict(coefficients=coeff, edge_flux=edge, surface_volume=volume,
                major_radius=ap.double().mean(), sections=sections, radii=radii,
                flux_fit_relative_rms=((matrix @ normalized_coeff - mean).square().mean().sqrt()
                                       / mean[-1].abs().clamp_min(1e-30)))


def volume_points(geometry, level: Tensor, flux: Mapping, config: DownstreamConfig):
    """Native local endpoint cloud, including each point's moving coordinates."""
    c = config
    rays = c.volume_phi_count * c.volume_theta_count
    nr = (math.ceil(c.volume_point_count * 1.25) + rays - 1) // rays
    candidate_count = rays * nr
    source = torch.div(torch.arange(c.volume_point_count, device=level.device) * candidate_count,
                       c.volume_point_count, rounding_mode='floor')
    ir = source % nr
    it = (source // nr) % c.volume_theta_count
    ip = source // (nr * c.volume_theta_count)
    pi = _arange(c.volume_phi_count, level)[:, None]
    ti = _arange(c.volume_theta_count, level)[None, :]
    phi_ray = ((pi + .417) * (2 * math.pi / (c.nfp * c.volume_phi_count))).expand(-1, c.volume_theta_count)
    shifted = ti + .613 + .217 * pi
    theta_ray = shifted.remainder(c.volume_theta_count) * (2 * math.pi / c.volume_theta_count)
    boundary_ray = geometry.solve_rays(geometry.rays(phi_ray, theta_ray), level)
    boundary = boundary_ray[ip, it]
    phi, theta = phi_ray[ip, it], theta_ray[ip, it]
    fraction = (c.volume_rho_min ** 2 + (1 - c.volume_rho_min ** 2)
                * (ir.to(level.dtype) + .371) / nr).sqrt()
    radius = boundary * fraction
    ar, az, arp, azp = geometry.axis_eval(phi)
    R, Z = ar + radius * theta.cos(), az + radius * theta.sin()
    s, gr, gz, gp = geometry.psi_eval(R, Z, phi)
    fs = polynomial(flux['coefficients'], s)
    degrees = torch.arange(1, flux['coefficients'].numel() + 1, device=s.device, dtype=s.dtype)
    fd = polynomial(flux['coefficients'] * degrees, s, constant=True)
    rho = (fs / flux['edge_flux']).clamp_min(0).sqrt()
    cp, sp = phi.cos(), phi.sin()
    gs = torch.stack((gr * cp - gp / R * sp, gr * sp + gp / R * cp, gz), -1)
    r2 = radius.square().clamp_min(1e-30)
    tr, tz = radius * theta.sin() / r2, -radius * theta.cos() / r2
    tp = (radius * theta.cos() * azp - radius * theta.sin() * arp) / r2
    gt = torch.stack((tr * cp - tp / R * sp, tr * sp + tp / R * cp, tz), -1)
    gphi = torch.stack((-sp / R, cp / R, torch.zeros_like(R)), -1)
    return dict(xyz=_xyz(R, Z, phi), grad_s=gs, grad_theta=gt, grad_phi=gphi,
                rho=rho, theta=-theta, phi=phi, volume_weight=R * boundary.square(),
                flux_derivative=fd, boundary=boundary_ray, s=s)


@lru_cache(maxsize=16)
def clebsch_modes(radial_order, poloidal_order, toroidal_order):
    """Exactly score_pipeline.cu::build_clebsch_modes_native ordering."""
    ms, ns, kinds, radial = [], [], [], []
    for m in range(min(radial_order, poloidal_order) + 1):
        for n in range(1 if m == 0 else -toroidal_order, toroidal_order + 1):
            for ell in range(m, radial_order + 1, 2):
                for kind in range(2):
                    ms.append(m); ns.append(n); kinds.append(kind)
                    row = [0.] * (radial_order + 1)
                    for k in range((ell - m) // 2 + 1):
                        row[ell - 2 * k] = ((-1) ** k * math.factorial(ell - k)
                            / (math.factorial(k) * math.factorial((ell + m) // 2 - k)
                               * math.factorial((ell - m) // 2 - k)))
                    radial.append(row)
    return tuple(ms), tuple(ns), tuple(kinds), tuple(tuple(r) for r in radial)


def _design_chunk(rho, theta, phi, bt, bp, weight, scale, metadata, nfp, start, end, iota_degree):
    m, n, kind, radial_coeff = metadata
    mode_count = m.numel()
    stop = min(end, mode_count)
    blocks = []
    if start < stop:
        radial = torch.zeros((rho.numel(), stop - start), device=rho.device, dtype=rho.dtype)
        for power in range(radial_coeff.shape[1] - 1, -1, -1):
            radial = radial * rho[:, None] + radial_coeff[start:stop, power]
        phase = theta[:, None] * m[start:stop] - phi[:, None] * n[start:stop] * nfp
        trig_derivative = torch.where(kind[start:stop] == 0, -phase.sin(), phase.cos())
        # d_phi basis = -n*nfp/m * d_theta when m != 0, with no division at m=0.
        tangent_factor = bt[:, None] * m[start:stop] - bp[:, None] * n[start:stop] * nfp
        blocks.append(weight[:, None] * radial * trig_derivative * tangent_factor)
    if end > mode_count:
        begin_iota = max(start, mode_count) - mode_count
        powers = torch.arange(begin_iota, end - mode_count, device=rho.device)
        blocks.append(-weight[:, None] * bp[:, None] * rho[:, None].pow(2 * powers))
    return torch.cat(blocks, -1) / scale[start:end]


class _AlphaDesign(torch.autograd.Function):
    """Store one dense design; recompute bounded feature tiles during VJP.

    Unlike saving every polynomial/trigonometric intermediate, memory is O(NC)
    with a small O(N*chunk) extra. All six point-feature cotangents are returned.
    """
    @staticmethod
    def forward(ctx, rho, theta, phi, bt, bp, weights, scale, metadata, nfp, iota_degree, chunk):
        ctx.save_for_backward(rho, theta, phi, bt, bp, weights, scale)
        ctx.mode_data, ctx.nfp, ctx.iota_degree, ctx.chunk = metadata, nfp, iota_degree, chunk
        cols = metadata[0].numel() + iota_degree + 1
        output = torch.empty((rho.numel(), cols), device=rho.device, dtype=rho.dtype)
        for lo in range(0, cols, chunk):
            hi = min(lo + chunk, cols)
            output[:, lo:hi] = _design_chunk(rho, theta, phi, bt, bp, weights, scale,
                                           metadata, nfp, lo, hi, iota_degree)
        return output

    @staticmethod
    def backward(ctx, adj_matrix):
        source = ctx.saved_tensors
        features = [x.detach().requires_grad_(True) for x in source[:6]]
        scale = source[6]
        grads = [torch.zeros_like(x) for x in features]
        cols = adj_matrix.shape[1]
        with torch.enable_grad():
            for lo in range(0, cols, ctx.chunk):
                hi = min(lo + ctx.chunk, cols)
                matrix = _design_chunk(*features, scale, ctx.mode_data, ctx.nfp, lo, hi, ctx.iota_degree)
                values = torch.autograd.grad(matrix, features, adj_matrix[:, lo:hi], allow_unused=True)
                for i, value in enumerate(values):
                    if value is not None:
                        grads[i] += value
        return (*grads, None, None, None, None, None)


def frozen_pcgls(A: Tensor, b: Tensor, cache: Mapping, ridge: float, iterations=4):
    """Reverse the actual fixed-step native recurrence; no converged-solve substitution."""
    upper = cache['upper'].to(device=A.device, dtype=A.dtype).detach()
    x = cache['scaled_solution'].to(device=A.device, dtype=A.dtype).detach()
    def tri(v, transposed=False):
        return torch.linalg.solve_triangular(upper.T if transposed else upper,
                   v[:, None], upper=not transposed).squeeze(-1)
    residual = b - A @ x
    normal = tri(A.T @ residual - ridge * x, True)
    direction = normal
    gamma = normal.square().sum()
    for iteration in range(iterations):
        transformed = tri(direction)
        image = A @ transformed
        delta = image.square().sum() + ridge * transformed.square().sum()
        step = gamma / delta.clamp_min(1e-30)
        x = x + step * transformed
        residual = residual - step * image
        if iteration + 1 < iterations:
            normal = tri(A.T @ residual - ridge * x, True)
            new_gamma = normal.square().sum()
            direction = normal + new_gamma / gamma.clamp_min(1e-30) * direction
            gamma = new_gamma
    return x, residual


def alpha_fit(points: Mapping, B: Tensor, cache: Mapping, config: DownstreamConfig):
    c = config
    n = points['rho'].numel()
    nf = min(c.alpha_fit_point_count, n)
    index = torch.div(torch.arange(nf, device=B.device) * n, nf, rounding_mode='floor')
    # Native alpha is FP32 even though magnetic-axis integration is mixed64.
    # These are live casts, so coordinate/psi cotangents still flow upstream.
    rho, theta, phi = (points[name][index].to(B.dtype) for name in ('rho', 'theta', 'phi'))
    field, gs, gt, gp = (t[index].to(B.dtype) for t in (B, points['grad_s'], points['grad_theta'], points['grad_phi']))
    field_norm2 = field.square().sum(-1)
    normal_coeff = (field * gs).sum(-1) / gs.square().sum(-1).clamp_min(1e-30)
    normal_B = normal_coeff[:, None] * gs
    tangent = field - normal_B
    bt, bp = (tangent * gt).sum(-1), (tangent * gp).sum(-1)
    bins = (rho.detach() * c.radial_bin_count).to(torch.long).clamp(0, c.radial_bin_count - 1)
    counts = torch.bincount(bins, minlength=c.radial_bin_count)
    weight = counts[bins].clamp_min(1).to(B.dtype).rsqrt() / field_norm2.clamp_min(1e-30).sqrt()
    weight = weight * (nf / weight.double().square().sum()).sqrt().to(weight.dtype)
    mode_lists = clebsch_modes(c.alpha_radial_order, c.alpha_poloidal_order, c.alpha_toroidal_order)
    metadata = tuple(torch.as_tensor(v, device=B.device, dtype=B.dtype) for v in mode_lists)
    mode_count = len(mode_lists[0])
    scale = cache['scale'].to(device=B.device, dtype=B.dtype).detach()
    if scale.shape != (mode_count + c.iota_degree + 1,):
        raise ValueError(f'alpha cache has {scale.shape}, expected {mode_count + c.iota_degree + 1} columns')
    A = _AlphaDesign.apply(rho, theta, phi, bt, bp, weight, scale, metadata,
                          c.nfp, c.iota_degree, c.alpha_chunk_columns)
    rhs = -weight * bt
    solution, residual = frozen_pcgls(A, rhs, cache, c.alpha_ridge, c.alpha_iterations)
    iota = solution[mode_count:] / scale[mode_count:]
    return dict(iota_coefficients=iota, iota_boundary=iota.sum(),
                normal_B_relative_l2=(normal_B.double().square().sum() / field.double().square().sum()).sqrt(),
                relative_l2=residual.double().norm() / rhs.double().norm(),
                bin_counts=counts.detach(), alpha_column_count=scale.numel())


def vacuum_G(geometry, ref: Tensor, edge_flux: Tensor, field: Callable, config: DownstreamConfig):
    phi = _arange(config.axis_sample_count, ref) * (2 * math.pi / (config.nfp * config.axis_sample_count))
    ar, az, arp, azp = geometry.axis_eval(phi)
    points = _xyz(ar, az, phi)
    tangent = torch.stack((arp * phi.cos() - ar * phi.sin(),
                           arp * phi.sin() + ar * phi.cos(), azp), -1)
    B = field(points)
    circulation_over_2pi = (B.double() * tangent.double()).sum(-1).mean()
    return circulation_over_2pi.abs() * edge_flux.detach().sign()


def qs_metric(points: Mapping, B: Tensor, gradB: Tensor, iota: Tensor, G: Tensor,
              config: DownstreamConfig):
    """All cotangents, including grad_s, flux', rho and quadrature weight, are live."""
    magnitude = B.square().sum(-1).clamp_min(1e-30).sqrt()
    grad_magnitude = torch.einsum('nic,ni->nc', gradB, B) / magnitude[:, None]
    # The native point kernel consumes FP32 geometry, while the mixed axis
    # construction may produce FP64 tensors. Preserve that precision boundary
    # with differentiable casts, including all moving-point cotangents.
    grad_flux = points['flux_derivative'].to(B.dtype)[:, None] * points['grad_s'].to(B.dtype)
    aa = (torch.linalg.cross(B, grad_flux) * grad_magnitude).sum(-1)
    cc = (B * grad_magnitude).sum(-1)
    iv = polynomial(iota, points['rho'].to(B.dtype).square(), constant=True)
    residual = ((config.target_M * iv.double() - config.target_N) * aa.double()
                - config.target_M * G.double() * cc.double()) / magnitude.double().pow(3).clamp_min(1e-30)
    weight = points['volume_weight'].to(B.dtype).double()
    rms = ((weight * residual.square()).sum() / weight.sum()).sqrt()
    norm = max(math.hypot(config.target_M, config.target_N), 1.)
    return dict(raw_qs=rms / norm, qs_global_error=rms, qs_vacuum_G=G,
                qs_loss=torch.log(rms / norm + 1e-8))


def surface_quality(geometry, level: Tensor, trace: Callable, config: DownstreamConfig,
                    *, promote_geometry=True, return_arrays=False):
    # Native FAST quality is double throughout root construction and endpoint
    # psi evaluation (including the optional native GPU roots/drift kernels).
    # The field itself remains FP32 inside its mixed64 trace callback. This
    # live view preserves coefficient/axis responses without changing flux or
    # alpha precision. The toggle exists for controlled numerical diagnostics.
    if promote_geometry and hasattr(geometry, 'coefficients'):
        geometry = type(geometry)(geometry.axis_samples.double(), geometry.coefficients.double(),
            geometry.nfp, geometry.a, geometry.degree, geometry.mtor,
            maximum_radius_scale=geometry.maximum_radius_scale,
            root_iterations=geometry.root_iterations, root_tolerance=geometry.root_tolerance)
        level = level.double()
    theta = _arange(config.surface_theta_count, level) * (2 * math.pi / config.surface_theta_count)
    phi = torch.zeros_like(theta)
    radius = geometry.solve_rays(geometry.rays(phi, theta), level)
    ar, az, _, _ = geometry.axis_eval(phi)
    initial = torch.stack((ar + radius * theta.cos(), az + radius * theta.sin()), -1)
    endpoint = trace(initial, 0., 2 * math.pi / config.nfp, config.surface_trace_steps)
    R, Z = endpoint.unbind(-1)
    s, gr, gz, gp = geometry.psi_eval(R, Z, phi + 2 * math.pi / config.nfp)
    norm = (gr.square() + gz.square() + (gp / R).square()).sqrt()
    drift = (s - level).abs() / (norm * radius.mean()).clamp_min(1e-14)
    quality = torch.quantile(drift.double(), .95)
    smooth = config.target_quality_temperature * F.softplus(
        (quality - config.target_quality_threshold) / config.target_quality_temperature)
    penalty = config.target_quality_weight * smooth.square()
    result = dict(target_quality_error=quality, quality_penalty=penalty,
                  surface_drift_relative_p95=quality, surface_relative_drift=drift)
    if return_arrays:
        result.update(surface_radius=radius, surface_initial=initial, surface_endpoint=endpoint,
                      surface_signed_psi_residual=s - level, surface_gradient_norm=norm,
                      surface_sorted_indices=torch.argsort(drift.detach()))
    return result


def downstream(geometry, level: Tensor, alpha_cache: Mapping, field: Callable,
               trace: Callable, potential: Callable | None, config: DownstreamConfig,
               *, return_arrays=False):
    flux = flux_calibration(geometry, level, field, potential, config)
    points = volume_points(geometry, level, flux, config)
    B, gradB = field(points['xyz'], gradient=True)
    alpha = alpha_fit(points, B, alpha_cache, config)
    G = vacuum_G(geometry, level, flux['edge_flux'], field, config)
    qs = qs_metric(points, B, gradB, alpha['iota_coefficients'], G, config)
    quality = surface_quality(geometry, level, trace, config)
    result = {**qs, **quality, **alpha, 'surface_volume': flux['surface_volume'],
              'surface_level': level, 'flux_edge': flux['edge_flux'],
              'flux_coefficients': flux['coefficients'], 'major_radius': flux['major_radius']}
    if return_arrays:
        result.update(points=points, B=B, gradB=gradB, flux_sections=flux['sections'])
    return result



