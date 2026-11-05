"""Local derivative of the S2-selected target-volume training interior."""
from __future__ import annotations

import math
import torch

from .front import PsiGeometry


def _volume(polygons):
    following = torch.roll(polygons, -1, 1)
    cross = polygons[..., 0]*following[..., 1]-following[..., 0]*polygons[..., 1]
    return 2*math.pi*((polygons[..., 0]+following[..., 0])*cross).sum(1).mean()/6


def target_sampling_geometry(geometry, metadata, *, reference_axis, local_value=False):
    """Differentiate fixed-volume roots, capacity clipping and area weights.

    The capacity-selection branch and quadrature nodes are fixed locally, as
    for S2. Root and volume equations have implicit derivatives; the S2 shape,
    axis motion and (on the clipped branch) capacity all remain live.
    """
    from .two_stage_front import _tensor, _value_with_response, two_stage_sampling_geometry
    boundary, sampling = metadata['boundary'], metadata['sampling']
    reference = geometry.coefficients
    phi, theta, radii = [_tensor(boundary[key], reference) for key in ('phis', 'theta', 'radii')]
    reference_coeff = _tensor(metadata['stage2_coefficients'], reference)
    reference_geometry = PsiGeometry(_tensor(reference_axis, reference), reference_coeff,
        geometry.nfp, geometry.a, geometry.degree, geometry.mtor)
    live = PsiGeometry(geometry.axis_samples.double(), reference.double(),
        geometry.nfp, geometry.a, geometry.degree, geometry.mtor)
    powers = torch.arange(geometry.degree+1, device=reference.device, dtype=torch.float64)
    polynomial = live.rays(phi[:, None], theta[None, :])
    captured = reference_geometry.rays(phi[:, None], theta[None, :])
    reference_value = (captured*radii[..., None]**powers).sum(-1)
    derivative = (captured[..., 1:]*powers[1:]*radii[..., None]**(powers[1:]-1)).sum(-1)
    if not bool(torch.isfinite(derivative).all() & (derivative > 1e-12).all()):
        raise ValueError('target-refit boundary has a singular radial root')
    value = (polynomial*radii[..., None]**powers).sum(-1)
    response = _value_with_response(torch.zeros_like(value), value, reference_value, local_value)
    fixed_roots = radii-response/derivative.detach()
    axis_R, axis_Z, _, _ = live.axis_eval(phi)

    def polygons(roots):
        return torch.stack((axis_R[:, None]+roots*theta.cos(),
                            axis_Z[:, None]+roots*theta.sin()), -1)

    fixed_volume = _volume(polygons(fixed_roots))
    reference_volume = _tensor(boundary['volume_m3'], reference)
    delta_fixed = _value_with_response(reference_volume*0, fixed_volume, reference_volume, local_value)
    delta_target = reference_volume*0
    if boundary['diagnostics']['capacity_clipped']:
        capacity_metadata = dict(stage1_coefficients=metadata['stage2_coefficients'],
            boundary=metadata['capacity_boundary'], sampling=sampling)
        capacity = two_stage_sampling_geometry(live, capacity_metadata,
            reference_axis=reference_axis, local_value=local_value)
        capacity_volume = _volume(capacity['polygons_rz'])
        capacity_ref = _tensor(metadata['capacity_boundary']['volume_m3'], reference)
        delta_target = .5*_value_with_response(capacity_ref*0, capacity_volume, capacity_ref, local_value)
    dvolume = _tensor(boundary['diagnostics']['volume_level_derivative'], reference)
    if not bool(torch.isfinite(dvolume) & (dvolume > 0)):
        raise ValueError('target-refit volume has a singular level derivative')
    delta_level = (delta_target-delta_fixed)/dvolume
    roots = fixed_roots+delta_level/derivative.detach()
    polygon = polygons(roots)
    following = torch.roll(polygon, -1, 1)
    areas = .5*(polygon[..., 0]*following[..., 1]-following[..., 0]*polygon[..., 1]).sum(-1)
    indices = _tensor(sampling['theta_indices'], reference, dtype=torch.long)
    u = _tensor(sampling['radial_u'], reference)
    radial_weights = _tensor(sampling['radial_weights'], reference)
    angles, edge = theta[indices], roots[:, indices]
    radius = edge[..., None]*u.sqrt()
    R = axis_R[:, None, None]+radius*angles.cos()[None, :, None]
    Z = axis_Z[:, None, None]+radius*angles.sin()[None, :, None]
    weights = (edge[..., None].square()*radial_weights).reshape(-1)
    section = torch.arange(len(phi), device=reference.device).repeat_interleave(len(indices)*len(u))
    return dict(kind='polar_quadrature', phis=phi, radii=roots, polygons_rz=polygon,
        level=_tensor(boundary['level'], reference)+delta_level,
        area=areas.sum(), section_areas=areas, volume=_volume(polygon), spacing=None,
        rz=torch.stack((R, Z), -1).reshape(-1, 2), section_index=section,
        row_weights=weights/weights.mean(),
        derivative_scope='fixed capacity-clipping branch and polar nodes; live axis/S2/volume-root/boundary/weights')
