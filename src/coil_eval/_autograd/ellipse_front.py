"""Local derivative of the axis, the moving ellipse lattice and its scalar fit."""
from __future__ import annotations

import math

import torch

from .._two_stage.ellipse import AXIS_NODES, VOLUME_FACTOR, ellipse_chart, field_linearization
from .batch_front import BatchFrontResult, batch_four_step_pcgls, differentiable_batch_axis
from .front import (FrontResult, PsiGeometry, differentiable_axis, four_step_pcgls,
                    periodic_hermite, predict_target_level)
from .two_stage_front import (BatchSectionPsiOperator, SectionPsiOperator,
                             _axis_local_value, _batch_axis_local_value)


def ellipse_sampling_geometry(axis, field, samplings, configs, *, local_value=False):
    """Differentiate the linearized field, periodic invariant and moving rows."""
    nfp = int(configs[0]['nfp'])
    period = 2 * math.pi / nfp
    options = dict(device=axis.device, dtype=axis.dtype)
    phi = torch.arange(AXIS_NODES, **options) * (period / AXIS_NODES)
    values = axis[..., :2].transpose(-1, -2)
    slopes = axis[..., 2:].transpose(-1, -2)
    rz, _ = periodic_hermite(phi, values, slopes, nfp)
    r, z = rz.unbind(-2)
    xyz = torch.stack((r * phi.cos(), r * phi.sin(), z), -1)
    magnetic, gradient = field(xyz, gradient=True)
    linear = field_linearization(r, phi, magnetic, gradient, torch)
    volume = torch.as_tensor([cfg['target_volume'] for cfg in configs], **options) * VOLUME_FACTOR
    chart, derivative, _ = ellipse_chart(linear, r, period, volume[0] if axis.ndim == 2 else volume, torch)
    if axis.ndim == 2:
        chart, derivative, values, slopes = (x.unsqueeze(0) for x in (chart, derivative, values, slopes))
    domains = []
    for index, (sampling, cfg) in enumerate(zip(samplings, configs)):
        phis = torch.as_tensor(sampling['phis'], **options)
        disk = torch.as_tensor(sampling['disk'], **options)
        native_map = torch.as_tensor(sampling['transform'], **options)
        center, _ = periodic_hermite(phis, values[index], slopes[index], nfp)
        mapping, _ = periodic_hermite(phis, chart[index].movedim(0, -1),
                                      derivative[index].movedim(0, -1), nfp)
        mapping = mapping.movedim(-1, 0)
        if not local_value:
            mapping = native_map + (mapping - mapping.detach())
        offsets = torch.einsum('pij,kj->kpi', mapping, disk)
        rz = offsets + center.T
        if local_value:
            limit = cfg['psi_a'] * cfg['surface_max_radius_scale']
            if not bool(torch.isfinite(rz).all() & (rz[..., 0] > 0.).all() &
                        (offsets.square().sum(-1) <= limit**2).all()):
                raise ValueError('local ellipse exceeds the physical training domain')
        section = torch.arange(len(phis), device=axis.device).repeat(len(disk))
        domains.append((rz.reshape(-1, 2), phis, section))
    return domains


def build_ellipse_front(field, ellipse_field, centers, *, batch=False, local_value=False, value_only=False):
    """Reuse the original Taylor basis and target-volume derivative after one fit."""
    configs = [center['config'] for center in centers]
    cfg = configs[0]
    nfp = int(cfg['nfp'])
    degree, mtor = int(cfg['psi_poly_degree']), int(cfg['psi_m_tor'])
    axis_settings = dict(nfp=nfp, trace_steps=int(cfg['axis_trace_steps']),
                         sample_count=int(cfg['axis_sample_count']))
    axes = torch.stack([center['axis_samples'] for center in centers]) if batch else centers[0]['axis_samples']
    with torch.enable_grad():
        if batch:
            axis_builder = _batch_axis_local_value if local_value else differentiable_batch_axis
            axis, diagnostics = axis_builder(field, axes[:, 0, :2], **axis_settings)
        else:
            axis_builder = _axis_local_value if local_value else differentiable_axis
            axis, diagnostics = axis_builder(field, field.trace, axes[0, :2], **axis_settings)
    if not local_value:
        axis = axes + (axis - axis.detach())
    if value_only:
        axis = axis.detach()
    domains = ellipse_sampling_geometry(axis, ellipse_field,
        [center['ellipse_sampling'] for center in centers], configs, local_value=local_value)
    radii = [float(c['psi_a']) for c in configs]
    operator_options = dict(nfp=nfp, degree=degree, mtor=mtor, dtype=torch.float64)
    fit_options = dict(iterations=4, live_column_scale=True)

    def frozen(center, key):
        value = center[key]
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value, device=axis.device, dtype=axis.dtype)

    fit_names = ('psi_coefficients', 'psi_scale', 'psi_upper')
    if batch:
        rz, phis, sections = zip(*domains)
        operator = BatchSectionPsiOperator(field, axis, rz, phis, sections, radii, **operator_options)
        coefficients, _ = batch_four_step_pcgls(operator,
            *(torch.stack([frozen(center, name) for center in centers]) for name in fit_names),
            ridge=[float(c['psi_ridge']) for c in configs], **fit_options)
    else:
        operator = SectionPsiOperator(field, axis, *domains[0], a=radii[0], **operator_options)
        coefficients, _ = four_step_pcgls(operator, *(frozen(centers[0], name) for name in fit_names),
                                          ridge=float(cfg['psi_ridge']), **fit_options)

    def geometry(samples, fit, config):
        return PsiGeometry(samples, fit, nfp, float(config['psi_a']), degree, mtor,
            maximum_radius_scale=float(config['surface_max_radius_scale']),
            root_iterations=int(config['surface_newton_iters']), root_tolerance=float(config['surface_newton_tolerance']),
            root_library=field.library)

    diagnostics.update(psi_fit_method='near_axis_ellipse', psi_fit_stages=1,
                       ellipse_training_points=[len(domain[0]) for domain in domains],
                       ellipse_axis_nodes=AXIS_NODES)
    if batch:
        geometries = [geometry(samples, fit, c) for samples, fit, c in zip(axis, coefficients, configs)]
        levels = torch.stack([predict_target_level(g, center) for g, center in zip(geometries, centers)])
        return BatchFrontResult(axis, coefficients, levels, operator.residual_rms(coefficients),
                                diagnostics, operator, geometries)
    result = geometry(axis, coefficients, cfg)
    return FrontResult(result, predict_target_level(result, centers[0]),
                       operator.residual_rms(coefficients), diagnostics, operator)
