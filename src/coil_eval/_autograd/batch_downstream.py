"""Batched surface, flux, alpha and QS metrics."""
from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from .column_norms import column_norms
from .triangular import independent_triangular_solve

from .downstream import (DownstreamConfig, _arange, _xyz, _alpha_radial_values,
    clebsch_modes, iota_profile_extrema, outer_qs_rms, volume_points,smooth_alpha_weights)


def _configs(configs, count):
    if isinstance(configs, (DownstreamConfig, Mapping)):
        configs = [configs] * count
    result = [c if isinstance(c, DownstreamConfig) else DownstreamConfig.from_mapping(c)
              for c in configs]
    if len(result) != count or not count:
        raise ValueError('one downstream config is required per query')
    independent = {'volume_rho_min', 'target_quality_threshold',
                   'target_quality_temperature', 'target_quality_weight',
                   'stokes_validation_rtol'}
    for name in (f.name for f in fields(DownstreamConfig) if f.name not in independent):
        if any(getattr(c, name) != getattr(result[0], name) for c in result[1:]):
            raise ValueError(f'batch downstream requires shared {name}')
    return result


def _stack_dict(items):
    return {key: torch.stack([value[key] for value in items]) for key in items[0]}


def _slice_dict(values, i):
    return {key: value[i] for key, value in values.items()}


def batch_flux_calibration(geometries, levels, field, configs):
    """One batch potential/field call for all independent flux calibrations."""
    configs = _configs(configs, len(geometries))
    c = configs[0]
    phi = _arange(c.flux_phi_count, levels) * (2 * math.pi / (c.nfp * c.flux_phi_count))
    theta = _arange(c.flux_theta_count, levels) * (2 * math.pi / c.flux_theta_count)
    pp, tt = torch.meshgrid(phi, theta, indexing='ij')
    fractions = (_arange(c.flux_level_count, levels) + 1) / c.flux_level_count
    ct, st = tt.cos()[:, None, :], tt.sin()[:, None, :]
    radii, axes, points, tangents, area_data = [], [], [], [], []
    for geometry, level in zip(geometries, levels.unbind()):
        ray = geometry.rays(pp, tt)
        radius = geometry.solve_rays(ray[:, None, :, :], (level * fractions.square())[None, :, None])
        ar, az, _, _ = geometry.axis_eval(pp)
        R, Z = ar[:, None, :] + radius * ct, az[:, None, :] + radius * st
        ph = pp[:, None, :].expand_as(R)
        radii.append(radius); axes.append(ar)
        if c.stokes_flux:
            _, gr, gz, _ = geometry.psi_eval(R, Z, ph)
            pr = gr * ct + gz * st
            if not bool(torch.all(torch.isfinite(pr) & (pr > 0)).detach()):
                raise ValueError('nonpositive/nonfinite radial derivative on Stokes boundary')
            rt = radius * (gr * st - gz * ct) / pr
            tangents.append(_xyz(rt * ct - radius * st, rt * st + radius * ct, ph))
            points.append(_xyz(R, Z, ph))
        else:
            nodes_np, weights_np = np.polynomial.legendre.leggauss(c.flux_radial_quadrature)
            nodes = torch.as_tensor(nodes_np, device=level.device, dtype=level.dtype)
            weights = torch.as_tensor(weights_np, device=level.device, dtype=level.dtype)
            r = .5 * radius[..., None] * (nodes + 1)
            rw = .5 * radius[..., None] * weights
            ph4 = ph[..., None].expand_as(r)
            points.append(_xyz(ar[:, None, :, None] + r * ct[..., None],
                               az[:, None, :, None] + r * st[..., None], ph4))
            area_data.append((r, rw, ph4))
    points = torch.stack(points)
    shape = points.shape
    if c.stokes_flux:
        A = field.potential(points.reshape(len(geometries), -1, 3)).reshape(shape)
        tangent = torch.stack(tangents)
        sections = -(A.double() * tangent.double()).sum(-1).mean(-1)
    else:
        B = field(points.reshape(len(geometries), -1, 3)).reshape(shape)
        r, rw, ph = (torch.stack([a[k] for a in area_data]) for k in range(3))
        bp = -B[..., 0] * ph.sin() + B[..., 1] * ph.cos()
        sections = (bp.double() * r.double() * rw.double()).sum(-1).mean(-1)
    mean = sections.mean(1)
    powers = torch.arange(1, c.flux_polynomial_degree + 1, device=levels.device)
    matrix = fractions.double().square()[:, None].pow(powers)
    normalized_coeff = torch.linalg.solve(matrix.T @ matrix, (mean @ matrix).T).T
    coeff = (normalized_coeff / levels.double()[:, None].pow(powers)).to(levels.dtype)
    edge = normalized_coeff.sum(-1).to(levels.dtype)
    radius, axis = torch.stack(radii), torch.stack(axes)
    boundary = radius[:, :, -1, :].double()
    volume = (2 * math.pi) ** 2 * (.5 * axis.double() * boundary.square()
              + tt.cos().double() * boundary.pow(3) / 3).mean((-1, -2))
    return dict(coefficients=coeff, edge_flux=edge, surface_volume=volume,
                major_radius=axis.double().mean((-1, -2)), sections=sections, radii=radius,
                flux_fit_relative_rms=((normalized_coeff @ matrix.T - mean).square().mean(-1).sqrt()
                                      / mean[:, -1].abs().clamp_min(1e-30)))


def _batch_design_chunk(rho, theta, phi, bt, bp, weight, scale, metadata,
                        nfp, start, end, iota_degree, radial_basis='zernike'):
    m, n, kind, radial_coeff = metadata
    mode_count = m.numel()
    stop = min(end, mode_count)
    blocks = []
    if start < stop:
        radial = _alpha_radial_values(rho, radial_coeff[start:stop], radial_basis)
        phase = theta[..., None] * m[start:stop] - phi[..., None] * n[start:stop] * nfp
        derivative = torch.where(kind[start:stop] == 0, -phase.sin(), phase.cos())
        tangent = bt[..., None] * m[start:stop] - bp[..., None] * n[start:stop] * nfp
        blocks.append(weight[..., None] * radial * derivative * tangent)
    if end > mode_count:
        powers = torch.arange(max(start, mode_count) - mode_count, end - mode_count, device=rho.device)
        blocks.append(-weight[..., None] * bp[..., None] * rho[..., None].pow(2 * powers))
    return torch.cat(blocks, -1) / scale[:, None, start:end]


class _BatchAlphaDesign(torch.autograd.Function):
    """Bounded-column feature recomputation, with every moving feature VJP."""
    @staticmethod
    def forward(ctx, rho, theta, phi, bt, bp, weight, scale, metadata, nfp, iota_degree, chunk,
                radial_basis='zernike'):
        # Bound recomputation temporaries as the batch grows. Retaining the
        # graph for component VJPs keeps the full fit state alive concurrently.
        chunk = min(chunk, max(1, (1 << 22) // max(1, rho.numel())))
        ctx.save_for_backward(rho, theta, phi, bt, bp, weight, scale)
        ctx.mode_data, ctx.nfp, ctx.iota_degree, ctx.chunk = metadata, nfp, iota_degree, chunk
        ctx.radial_basis = radial_basis
        columns = metadata[0].numel() + iota_degree + 1
        matrix = torch.empty((*rho.shape, columns), device=rho.device, dtype=rho.dtype)
        for lo in range(0, columns, chunk):
            hi = min(lo + chunk, columns)
            matrix[..., lo:hi] = _batch_design_chunk(rho, theta, phi, bt, bp, weight,
                scale, metadata, nfp, lo, hi, iota_degree, radial_basis)
        return matrix

    @staticmethod
    def backward(ctx, adjoint):
        source = ctx.saved_tensors
        features = [t.detach().requires_grad_(True) for t in source[:6]]
        gradients = [torch.zeros_like(t) for t in features]
        with torch.enable_grad():
            for lo in range(0, adjoint.shape[-1], ctx.chunk):
                hi = min(lo + ctx.chunk, adjoint.shape[-1])
                matrix = _batch_design_chunk(*features, source[6], ctx.mode_data,
                    ctx.nfp, lo, hi, ctx.iota_degree, ctx.radial_basis)
                values = torch.autograd.grad(matrix, features, adjoint[..., lo:hi], allow_unused=True)
                for j, value in enumerate(values):
                    if value is not None:
                        gradients[j] += value
        return (*gradients, *(None for _ in range(len(ctx.needs_input_grad) - 6)))


def batch_frozen_pcgls(A, b, cache, ridge, iterations=4, *, live_column_scale=False):
    upper = cache['upper'].to(A).detach()
    x = cache['scaled_solution'].to(A).detach()
    penalty=(column_norms(A).clamp_min(1e-20).to(A.dtype).square()
             if live_column_scale else 1.)
    def tri(v, transposed=False):
        return independent_triangular_solve(upper.transpose(-1, -2) if transposed else upper,
                    v[..., None], upper=not transposed).squeeze(-1)
    def apply(v):
        return torch.bmm(A, v[..., None]).squeeze(-1)
    def transpose(v):
        return torch.bmm(A.transpose(1, 2), v[..., None]).squeeze(-1)
    residual = b - apply(x)
    normal = tri(transpose(residual) - ridge * penalty * x, True)
    direction = normal
    gamma = normal.square().sum(-1, keepdim=True)
    for k in range(iterations):
        transformed = tri(direction)
        image = apply(transformed)
        ridge_norm=((penalty*transformed.square()).sum(-1,keepdim=True) if live_column_scale else
                    transformed.square().sum(-1,keepdim=True))
        delta = image.square().sum(-1, keepdim=True) + ridge * ridge_norm
        step = gamma / delta.clamp_min(1e-30)
        x = x + step * transformed
        residual = residual - step * image
        if k + 1 < iterations:
            normal = tri(transpose(residual) - ridge * penalty * x, True)
            new_gamma = normal.square().sum(-1, keepdim=True)
            direction = normal + new_gamma / gamma.clamp_min(1e-30) * direction
            gamma = new_gamma
    return x, residual


def batch_alpha_fit(points, B, cache, config, *, volume_rho_min=None):
    c = config
    nq, n = points['rho'].shape
    nf = min(c.alpha_fit_point_count, n)
    index = torch.div(torch.arange(nf, device=B.device) * n, nf, rounding_mode='floor')
    rho, theta, phi = (points[name][:, index].to(B.dtype) for name in ('rho', 'theta', 'phi'))
    field, gs, gt, gp = (t[:, index].to(B.dtype)
                         for t in (B, points['grad_s'], points['grad_theta'], points['grad_phi']))
    field_norm2 = field.square().sum(-1)
    normal = (field * gs).sum(-1) / gs.square().sum(-1).clamp_min(1e-30)
    normal_B = normal[..., None] * gs
    tangent = field - normal_B
    bt, bp = (tangent * gt).sum(-1), (tangent * gp).sum(-1)
    if c.alpha_radial_weighting=='smooth':
        weight,counts=smooth_alpha_weights(rho,field_norm2,c)
    elif c.alpha_radial_weighting=='histogram':
        bins = (rho.detach() * c.radial_bin_count).long().clamp(0, c.radial_bin_count - 1)
        counts = torch.zeros((nq, c.radial_bin_count), device=B.device, dtype=torch.long)
        counts.scatter_add_(1, bins, torch.ones_like(bins))
        weight = counts.gather(1, bins).clamp_min(1).to(B.dtype).rsqrt() / field_norm2.clamp_min(1e-30).sqrt()
        weight = weight * (nf / weight.double().square().sum(-1, keepdim=True)).sqrt().to(B.dtype)
    else:raise ValueError('alpha_radial_weighting must be histogram or smooth')
    modes = clebsch_modes(c.alpha_radial_order, c.alpha_poloidal_order, c.alpha_toroidal_order,
                         c.alpha_radial_basis)
    metadata = tuple(torch.as_tensor(v, device=B.device, dtype=B.dtype) for v in modes)
    mode_count = len(modes[0])
    scale = cache['scale'].to(B).detach()
    if scale.shape != (nq, mode_count + c.iota_degree + 1):
        raise ValueError('alpha cache shape does not match batch and complete basis')
    A = _BatchAlphaDesign.apply(rho, theta, phi, bt, bp, weight, scale, metadata,
                               c.nfp, c.iota_degree, c.alpha_chunk_columns, c.alpha_radial_basis)
    rhs = -weight * bt
    solution, residual = batch_frozen_pcgls(A, rhs, cache, c.alpha_ridge, c.alpha_iterations,
        live_column_scale=c.alpha_radial_weighting=='smooth')
    iota = solution[:, mode_count:] / scale[:, mode_count:]
    return dict(iota_coefficients=iota, iota_boundary=iota.sum(-1),
        **iota_profile_extrema(iota,c.volume_rho_min if volume_rho_min is None else volume_rho_min),
        normal_B_relative_l2=(normal_B.double().square().sum((-1, -2))
                              / field.double().square().sum((-1, -2))).sqrt(),
        relative_l2=torch.linalg.vector_norm(residual.double(), dim=-1)
                    / torch.linalg.vector_norm(rhs.double(), dim=-1),
        bin_counts=counts.detach(), alpha_column_count=scale.shape[-1])


def batch_surface_quality(geometries, levels, field, configs, *, return_arrays=False):
    configs = _configs(configs, len(geometries))
    c = configs[0]
    promoted, radii, initial = [], [], []
    levels = levels.double()
    theta = _arange(c.surface_theta_count, levels) * (2 * math.pi / c.surface_theta_count)
    phi = torch.zeros_like(theta)
    for geometry, level in zip(geometries, levels.unbind()):
        if hasattr(geometry, 'coefficients'):
            geometry = type(geometry)(geometry.axis_samples.double(), geometry.coefficients.double(),
                geometry.nfp, geometry.a, geometry.degree, geometry.mtor,
                maximum_radius_scale=geometry.maximum_radius_scale,
                root_iterations=geometry.root_iterations, root_tolerance=geometry.root_tolerance)
        promoted.append(geometry)
        radius = geometry.solve_rays(geometry.rays(phi, theta), level)
        ar, az, _, _ = geometry.axis_eval(phi)
        initial.append(torch.stack((ar + radius * theta.cos(), az + radius * theta.sin()), -1))
        radii.append(radius)
    initial, radii = torch.stack(initial), torch.stack(radii)
    endpoint = field.trace(initial, 0., 2 * math.pi / c.nfp, c.surface_trace_steps)
    values = []
    for geometry, end in zip(promoted, endpoint.unbind()):
        R, Z = end.unbind(-1)
        s, gr, gz, gp = geometry.psi_eval(R, Z, phi + 2 * math.pi / c.nfp)
        values.append((s, (gr.square() + gz.square() + (gp / R).square()).sqrt()))
    s, norm = (torch.stack([v[k] for v in values]) for k in range(2))
    signed = s - levels[:, None]
    drift = signed.abs() / (norm * radii.mean(-1, keepdim=True)).clamp_min(1e-14)
    quality = torch.quantile(drift.double(), .95, dim=-1)
    threshold, temperature, weight = (levels.new_tensor([getattr(v, name) for v in configs])
        for name in ('target_quality_threshold', 'target_quality_temperature', 'target_quality_weight'))
    smooth = temperature * F.softplus((quality - threshold) / temperature)
    result = dict(target_quality_error=quality, quality_penalty=weight * smooth.square(),
                  surface_drift_relative_p95=quality, surface_relative_drift=drift)
    if return_arrays:
        result.update(surface_radius=radii, surface_initial=initial, surface_endpoint=endpoint,
                      surface_signed_psi_residual=signed, surface_gradient_norm=norm,
                      surface_sorted_indices=torch.argsort(drift.detach(), dim=-1))
    return result


def batch_qs_metric(points, B, gradB, iota, G, config, *, volume_rho_min=None):
    """Full volume reductions over points, independently for every query."""
    magnitude = B.square().sum(-1).clamp_min(1e-30).sqrt()
    grad_magnitude = torch.einsum('bnic,bni->bnc', gradB, B) / magnitude[..., None]
    grad_flux = points['flux_derivative'].to(B.dtype)[..., None] * points['grad_s'].to(B.dtype)
    aa = (torch.linalg.cross(B, grad_flux) * grad_magnitude).sum(-1)
    cc = (B * grad_magnitude).sum(-1)
    rho2 = points['rho'].to(B.dtype).square()
    iv = torch.zeros_like(rho2)
    for coefficient in iota.flip(-1).unbind(-1):
        iv = iv * rho2 + coefficient[:, None]
    residual = ((config.target_M * iv.double() - config.target_N) * aa.double()
                - config.target_M * G.double()[:, None] * cc.double()) / magnitude.double().pow(3).clamp_min(1e-30)
    weight = points['volume_weight'].to(B.dtype).double()
    rms = ((weight * residual.square()).sum(-1) / weight.sum(-1)).sqrt()
    norm = max(math.hypot(config.target_M, config.target_N), 1.)
    outer=outer_qs_rms(residual,weight,points['rho'],config,volume_rho_min=volume_rho_min)
    return dict(raw_qs=rms/norm, raw_qs_outer=outer/norm,
                qs_global_error=rms, qs_edge_error=outer, qs_vacuum_G=G,
                qs_loss=torch.log(rms/norm+1e-8))


def batch_downstream(front, alpha_caches, field, configs, *, return_arrays=False):
    """All outputs have a leading query dimension; no query is normalized away.

    ``front.geometry(i)`` must return live views. ``alpha_caches`` may be a list
    of per-query dictionaries or a dictionary of stacked detached tensors.
    """
    levels = front.levels
    nq = len(levels)
    configs = _configs(configs, nq)
    c = configs[0]
    geometries = [front.geometry(i) for i in range(nq)]
    cache = _stack_dict(alpha_caches) if not isinstance(alpha_caches, Mapping) else alpha_caches
    flux = batch_flux_calibration(geometries, levels, field, configs)
    from .flux_validation import stokes_check
    flux_validation, area_queries = [], []
    for i, config in enumerate(configs):
        if config.stokes_flux:
            use_area, info = stokes_check(_slice_dict(flux, i), levels[i],
                                          _slice_dict(cache, i), config.stokes_validation_rtol)
        else:
            use_area, info = False, dict(method='area', validation='not_requested', fallback=False)
        flux_validation.append(info)
        if use_area:
            area_queries.append(i)
    if area_queries:
        indices = torch.tensor(area_queries, device=levels.device, dtype=torch.int64)
        area = batch_flux_calibration(
            [geometries[i] for i in area_queries], levels.index_select(0, indices),
            field.select_queries(indices),
            [replace(configs[i], stokes_flux=False) for i in area_queries])
        # index_copy removes the rejected Stokes VJP on selected rows. Both
        # their geometry and field contributions flow through live area work.
        flux = {name: value.index_copy(0, indices, area[name]) for name, value in flux.items()}
    points = _stack_dict([volume_points(g, levels[i], _slice_dict(flux, i), configs[i],
                           source_indices=(cache['volume_source_indices'][i]
                                           if 'volume_source_indices' in cache else None),
                           sampling_shape=(cache['volume_sampling_shape'][i]
                                           if 'volume_sampling_shape' in cache else None))
                          for i, g in enumerate(geometries)])
    phi = _arange(c.axis_sample_count, levels) * (2 * math.pi / (c.nfp * c.axis_sample_count))
    axis_points, axis_tangents = [], []
    for geometry in geometries:
        ar, az, arp, azp = geometry.axis_eval(phi)
        axis_points.append(_xyz(ar, az, phi))
        axis_tangents.append(torch.stack((arp * phi.cos() - ar * phi.sin(),
                                         arp * phi.sin() + ar * phi.cos(), azp), -1))
    # Axis points add only 240 rows to the 100k point field call, avoiding a
    # second field launch while keeping their coordinate derivatives intact.
    combined = torch.cat((points['xyz'], torch.stack(axis_points)), 1)
    B_all, grad_all = field(combined, gradient=True)
    nv = c.volume_point_count
    B, gradB = B_all[:, :nv], grad_all[:, :nv]
    G = (B_all[:, nv:].double() * torch.stack(axis_tangents).double()).sum(-1).mean(-1).abs()
    G = G * flux['edge_flux'].detach().sign()
    lower_radii=[config.volume_rho_min for config in configs]
    alpha = batch_alpha_fit(points, B, cache, c, volume_rho_min=lower_radii)
    qs = batch_qs_metric(points, B, gradB, alpha['iota_coefficients'], G, c,
                         volume_rho_min=lower_radii)
    quality = batch_surface_quality(geometries, levels, field, configs, return_arrays=return_arrays)
    result = {**qs, **quality, **alpha, 'surface_volume': flux['surface_volume'],
              'surface_level': levels, 'flux_edge': flux['edge_flux'],
              'flux_coefficients': flux['coefficients'], 'major_radius': flux['major_radius']}
    result['flux_validation'] = flux_validation
    if return_arrays:
        result.update(points=points, B=B, gradB=gradB, flux_sections=flux['sections'])
    return result

