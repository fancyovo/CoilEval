"""Near-axis ellipse sampling in physical R-Z coordinates."""
from __future__ import annotations

import math

import numpy as np


AXIS_NODES = 1024
VOLUME_FACTOR = 1.2


class EllipseDomainError(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.diagnostics = dict(failure_stage='ellipse_domain', failure_reason=reason, **details)


def disk_points(n_r, n_z, inner_radius):
    x, y = np.meshgrid(np.linspace(-1., 1., n_r), np.linspace(-1., 1., n_z), indexing='ij')
    radius = np.hypot(x, y)
    keep = (radius > inner_radius) & (radius <= 1.)
    return np.column_stack((x[keep], y[keep]))


def field_linearization(radius, phi, field, gradient, xp):
    """Derivative of (R B_R/B_phi, R B_Z/B_phi) at fixed toroidal angle."""
    cosine, sine = xp.cos(phi), xp.sin(phi)
    zero = xp.zeros_like(phi)
    radial = xp.stack((cosine, sine, zero), -1)
    vertical = xp.stack((zero, zero, xp.ones_like(phi)), -1)
    toroidal = xp.stack((-sine, cosine, zero), -1)
    transverse = xp.stack((radial, vertical), -1)
    b_phi = (field * toroidal).sum(-1)
    if xp is np and (not np.isfinite(field).all() or not np.isfinite(gradient).all()
                     or np.any(np.abs(b_phi) < 1e-12)):
        raise EllipseDomainError('axis_field_linearization_unavailable')
    spatial = gradient @ transverse
    db_phi = (spatial * toroidal[..., None]).sum(-2)
    slope = (field[..., None] * transverse).sum(-2) / b_phi[..., None]
    matrix = (radius / b_phi)[..., None, None] * (
        xp.swapaxes(transverse, -1, -2) @ spatial - slope[..., :, None] * db_phi[..., None, :])
    return matrix + xp.stack((slope, xp.zeros_like(slope)), -1)


def _det(matrix):
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _inverse(matrix, xp):
    a, b, c, d = (matrix[..., i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return xp.stack((d, -b, -c, a), -1).reshape(matrix.shape) / _det(matrix)[..., None, None]


def _linear_flow(linear, period, identity, xp):
    """RK4 transfer maps on the periodic cubic interpolant of the sampled field."""
    count = linear.shape[-3]
    step = period / count
    # The periodic cubic spline equations are circulant on an equally spaced grid.
    frequencies = np.cos(2 * np.pi * np.arange(count) / count)
    symbol = 6 * (2 * frequencies - 2) / (4 + 2 * frequencies)
    if xp is not np:
        symbol = xp.as_tensor(symbol, dtype=linear.dtype, device=linear.device)
    curvature = xp.fft.ifft(xp.fft.fft(linear, axis=-3) * symbol[:, None, None], axis=-3).real
    following = xp.roll(linear, -1, -3)
    midpoint = (linear + following) / 2 - (curvature + xp.roll(curvature, -1, -3)) / 16
    k2 = midpoint @ (identity + (step / 2) * linear)
    k3 = midpoint @ (identity + (step / 2) * k2)
    k4 = following @ (identity + step * k3)
    maps = identity + (step / 6) * (linear + 2 * k2 + 2 * k3 + k4)
    # Each prefix is the ordered product of the interval maps up to that node.
    stride = 1
    while stride < count:
        maps = xp.concatenate((maps[..., :stride, :, :],
                               maps[..., stride:, :, :] @ maps[..., :-stride, :, :]), -3)
        stride *= 2
    first = xp.broadcast_to(identity, maps[..., :1, :, :].shape)
    return xp.concatenate((first, maps), -3)


def ellipse_chart(linear, radius, period, volume, xp):
    """Periodic quadratic invariant and its inverse-square-root ellipse map."""
    identity = (np.eye(2, dtype=linear.dtype) if xp is np else
                xp.eye(2, dtype=linear.dtype, device=linear.device))
    flow = _linear_flow(linear, period, identity, xp)
    end = flow[..., -1, :, :]
    a, b, c, d = (end[..., i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    off_diagonal = (d - a) / 2
    # For a two-dimensional elliptic return map, J(M - tr(M)I/2) is the
    # invariant symmetric form up to scale, with sign chosen positive.
    q0 = xp.stack((c, off_diagonal, off_diagonal, -b), -1).reshape(end.shape)
    q0 = q0 * xp.sign(c - b)[..., None, None]
    det_q0 = _det(q0)
    diagnostics = None
    if xp is np:
        if not np.isfinite(flow).all():
            raise EllipseDomainError('axis_linear_integration_failed')
        basis = np.array([[[1., 0.], [0., 0.]], [[0., 1.], [1., 0.]], [[0., 0.], [0., 1.]]])
        system = (end.T @ basis @ end - basis)[:, np.triu_indices(2)[0], np.triu_indices(2)[1]].T
        singular = np.linalg.svd(system, compute_uv=False)
        diagnostics = dict(return_matrix=end.tolist(), determinant=float(_det(end)), trace=float(a + d),
                           axis_elliptic_margin=float(2 - abs(a + d)), singular_values=singular.tolist(),
                           invariant_separation=float(singular[-2] / max(singular[0], 1e-30)))
        if not det_q0 > 0 or not np.trace(q0) > 0:
            raise EllipseDomainError('axis_quadratic_not_positive_definite', **diagnostics)
    q0 = q0 / xp.sqrt(det_q0)[..., None, None]
    inverse = _inverse(flow, xp)
    transported = xp.swapaxes(inverse, -1, -2) @ q0[..., None, :, :] @ inverse
    metric = transported[..., :-1, :, :]
    if xp is np:
        closure = np.linalg.norm(transported[-1] - transported[0]) / np.linalg.norm(transported[0])
        diagnostics.update(metric_periodic_relative_error=float(closure),
                           q0_eigenvalues=np.linalg.eigvalsh(q0).tolist())
        if closure > 1e-4 or abs(diagnostics['determinant'] - 1.) > 1e-4:
            raise EllipseDomainError('axis_metric_periodicity_unresolved', **diagnostics)
        if diagnostics['axis_elliptic_margin'] < 1e-5 or diagnostics['invariant_separation'] < 1e-6:
            raise EllipseDomainError('axis_metric_near_resonance', **diagnostics)
    det = _det(metric)
    root_det = xp.sqrt(det)
    trace = metric[..., 0, 0] + metric[..., 1, 1]
    # Closed-form positive square root avoids differentiating eigenvectors.
    inv_sqrt = _inverse(metric + root_det[..., None, None] * identity, xp)
    inv_sqrt = inv_sqrt * xp.sqrt(trace + 2 * root_det)[..., None, None]
    scale = xp.sqrt(volume / (2 * math.pi**2 * (radius / root_det).mean(-1)))
    direction = _inverse(inv_sqrt[..., 0, :, :], xp)[..., :, 0]
    direction = direction / xp.sqrt((direction**2).sum(-1))[..., None]
    u, v = direction[..., 0], direction[..., 1]
    rotation = xp.stack((u, -v, v, u), -1).reshape(end.shape)
    chart = (inv_sqrt @ rotation[..., None, :, :]) * scale[..., None, None, None]
    frequencies = np.fft.fftfreq(linear.shape[-3], d=period / (2 * np.pi * linear.shape[-3]))
    if xp is not np:
        frequencies = xp.as_tensor(frequencies, dtype=linear.dtype, device=linear.device)
    derivative = xp.fft.ifft(xp.fft.fft(chart, axis=-3) * (1j * frequencies[:, None, None]), axis=-3).real
    if diagnostics is not None:
        eig = np.linalg.eigvalsh(metric)
        diagnostics.update(scale_b=float(scale), ellipse_semiaxis_min=float(scale / np.sqrt(eig).max()),
                           ellipse_semiaxis_max=float(scale / np.sqrt(eig).min()),
                           elongation_max=float(np.sqrt(eig[:, 1] / eig[:, 0]).max()),
                           training_volume_requested=float(volume))
    return chart, derivative, diagnostics


def sample_ellipse(axis, field, cfg, nfp):
    """Build the one-fit lattice and retain the small center map for derivatives."""
    from ..simsopt.state import _hermite

    period = 2 * np.pi / nfp
    phi = np.arange(AXIS_NODES) * period / AXIS_NODES
    radius, height = (_hermite(phi, axis[:, i], axis[:, i + 2], nfp) for i in range(2))
    xyz = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height))
    magnetic, gradient = field(xyz, gradient=True)
    linear = field_linearization(radius, phi, magnetic, gradient, np)
    chart, derivative, diagnostics = ellipse_chart(linear, radius, period, VOLUME_FACTOR * cfg.target_volume, np)
    phis = np.arange(cfg.psi_n_phi) * period / cfg.psi_n_phi
    radius, height = (_hermite(phis, axis[:, i], axis[:, i + 2], nfp) for i in range(2))
    transform = np.stack([_hermite(phis, chart[:, i, j], derivative[:, i, j], nfp)
                          for i in range(2) for j in range(2)], -1).reshape(-1, 2, 2)
    disk = disk_points(cfg.psi_n_r, cfg.psi_n_z, cfg.psi_rho_min / cfg.psi_a)
    if not len(disk):
        raise EllipseDomainError('ellipse_training_lattice_empty')
    offsets = np.einsum('pij,kj->kpi', transform, disk)
    radius_limit = cfg.psi_a * cfg.surface_max_radius_scale
    maximum_radius = float(np.linalg.norm(offsets, axis=-1).max())
    rz = offsets + np.stack((radius, height), -1)
    if maximum_radius > radius_limit or rz[..., 0].min() <= 0.:
        raise EllipseDomainError('predicted_training_domain_exceeds_original_physical_cap',
                                 max_radius_m=maximum_radius, radial_cap_m=radius_limit,
                                 minimum_R_m=float(rz[..., 0].min()))
    diagnostics['maximum_training_radius_m'] = maximum_radius
    return (np.ascontiguousarray(rz.reshape(-1, 2)), np.tile(phis, len(disk)),
            dict(phis=phis, disk=disk, transform=transform), diagnostics)
