"""A fixed-volume interior training domain selected from the second scalar.

Only the scalar, axis and requested volume enter this selection. The first
axis-connected capacity guard stays active. No reference surface is consulted.
"""
from __future__ import annotations

import numpy as np

from .boundary import MaxClosedBoundaryError, max_closed_boundary
from .polynomial import ray_polynomials
from .quadrature import sample_polar_quadrature


def polygon_volume(polygons, tangent=None):
    """Full-device volume and optionally its directional derivative."""
    following = np.roll(polygons, -1, axis=1)
    cross = polygons[..., 0]*following[..., 1]-following[..., 0]*polygons[..., 1]
    radius_sum = polygons[..., 0]+following[..., 0]
    volume = float(2*np.pi*np.mean(np.sum(radius_sum*cross, axis=1)/6))
    if tangent is None:
        return volume
    next_tangent = np.roll(tangent, -1, axis=1)
    dcross = (tangent[..., 0]*following[..., 1]+polygons[..., 0]*next_tangent[..., 1]
              -next_tangent[..., 0]*polygons[..., 1]-following[..., 0]*tangent[..., 1])
    derivative = float(2*np.pi*np.mean(np.sum(
        (tangent[..., 0]+next_tangent[..., 0])*cross+radius_sum*dcross, axis=1)/6))
    return volume, derivative


def _values(polynomial, u):
    value = polynomial[..., -1].copy()
    derivative = np.zeros_like(value)
    for j in range(polynomial.shape[-1]-2, -1, -1):
        derivative = derivative*u+value
        value = value*u+polynomial[..., j]
    return value, derivative


def _first_roots(polynomial, level, upper):
    """Safeguarded Newton inside a previously certified first-root bracket."""
    low = np.zeros_like(upper)
    high = upper.copy()
    capacity, _ = _values(polynomial, upper)
    u = upper*np.sqrt(np.minimum(level/capacity, 1.))
    tolerance = max(abs(level)*2e-13, np.finfo(float).tiny)
    for _ in range(64):
        value, derivative = _values(polynomial, u)
        residual = value-level
        done = abs(residual) <= tolerance
        if done.all():
            return u, derivative
        low = np.where((residual < 0) & ~done, u, low)
        high = np.where((residual >= 0) & ~done, u, high)
        safe = np.where(derivative > 0, derivative, 1.)
        newton = u-residual/safe
        candidate = np.where((derivative > 0) & (newton > low) & (newton < high),
                             newton, .5*(low+high))
        u = np.where(done, u, candidate)
    raise MaxClosedBoundaryError('target-refit first-root solve did not converge')


def target_training_domain(model, phis, target_volume, *, factor=1.2,
                           target_count=82368, ntheta=256, theta_count=128,
                           workers=None):
    """Return full-interior polar samples and all local derivative metadata.

    Domain volume is min(factor*target, (target+capacity)/2). A capacity below
    target is a failure; the physical requested volume is never reduced.
    """
    target_volume, factor = float(target_volume), float(factor)
    if not np.isfinite(target_volume) or target_volume <= 0 or not np.isfinite(factor) or factor <= 1:
        raise ValueError('positive target volume and target-refit factor >1 required')
    capacity = max_closed_boundary(model, phis, ntheta=ntheta, max_radius=model.a, workers=workers)
    available = capacity['volume_m3']
    if available <= target_volume:
        raise MaxClosedBoundaryError('S2 first-root capacity does not exceed the target-refit volume',
            dict(failure_reason='target_refit_capacity_below_target',
                 capacity_volume=available, target_volume=target_volume))
    clipped = factor*target_volume > .5*(target_volume+available)
    requested = .5*(target_volume+available) if clipped else factor*target_volume
    theta = capacity['theta']
    polynomial = ray_polynomials(model, capacity['phis'], theta)
    axis_R, axis_Z, _, _ = model.axis_at(capacity['phis'])
    upper_u = capacity['radii']/model.a
    low, high = 0., capacity['level']
    level = high*requested/available
    for iteration in range(40):
        u, derivative = _first_roots(polynomial, level, upper_u)
        if not np.isfinite(derivative).all() or np.any(derivative <= 0):
            raise MaxClosedBoundaryError('target-refit boundary has a singular radial root')
        radii = model.a*u
        polygons = np.stack((axis_R[:, None]+radii*np.cos(theta),
                             axis_Z[:, None]+radii*np.sin(theta)), -1)
        tangent = model.a/derivative
        tangent = tangent[..., None]*np.stack((np.cos(theta), np.sin(theta)), -1)
        volume, dvolume = polygon_volume(polygons, tangent)
        residual = volume-requested
        if abs(residual) <= 2e-11*requested:
            break
        if not np.isfinite(dvolume) or dvolume <= 0:
            raise MaxClosedBoundaryError('target-refit volume has a nonpositive level derivative')
        if residual < 0:
            low = level
        else:
            high = level
        candidate = level-residual/dvolume
        level = candidate if low < candidate < high else .5*(low+high)
    else:
        raise MaxClosedBoundaryError('target-refit volume solve did not converge')
    if not np.isfinite(polygons).all() or np.any(polygons[..., 0] <= 0):
        raise MaxClosedBoundaryError('target-refit boundary requires finite positive cylindrical R')
    diagnostics = dict(schema='target-local-s3-domain-v1',
        target_volume=target_volume, requested_factor=factor,
        selected_factor=requested/target_volume, capacity_clipped=clipped,
        capacity_volume=available, requested_training_volume=requested,
        training_volume=volume, volume_relative_error=residual/requested,
        volume_level_derivative=dvolume, volume_iterations=iteration+1,
        domain_scope='entire axis-connected interior, not a thin shell',
        reference_geometry_used=False)
    boundary = dict(phis=capacity['phis'], theta=theta, radii=radii,
        polygons_rz=polygons, level=level, volume_m3=volume,
        ds_dr=derivative/model.a, diagnostics=diagnostics)
    sample = sample_polar_quadrature(boundary, target_count=target_count,
                                    theta_count=theta_count, nfp=model.nfp)
    # The capacity boundary is retained even on the un-clipped branch for
    # auditing; only the clipped branch needs its derivative.
    return dict(boundary=boundary, sampling=sample, capacity_boundary=capacity)
