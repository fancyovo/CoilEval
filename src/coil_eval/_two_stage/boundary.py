"""Largest sampled closed first-root s boundary within a physical radius cap.

The axis-connected radial branch ends at its first genuine positive-to-negative
derivative fold or at the physical radius cap. Flat stationary points with no
sign reversal are allowed. The native strict Bernstein certificate is not used.
No magnetic field, fit-error, native validity or magnetic-quality gate is read.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .polynomial import ray_polynomials as _ray_polynomials
from .branches import branch_limits, resolve_workers


class MaxClosedBoundaryError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _signed_value(coefficients, value):
    """Extended-precision polynomial value and a numerical sign error bound."""
    c = np.asarray(coefficients, dtype=np.longdouble)
    x = np.longdouble(value)
    result = np.polynomial.polynomial.polyval(x, c)
    scale = np.polynomial.polynomial.polyval(abs(x), abs(c))
    bound = 64*np.finfo(np.longdouble).eps*scale
    return result, bound


def first_connected_branch(polynomial, maximum_u):
    """Find the first true fold; allow flat nonfold stationary points.

    Returned u is a physical branch endpoint, without an inward margin. Roots
    partition q(u)=p'(u)/u; interval midpoints determine derivative signs. Any
    numerically unresolved critical-root configuration is an explicit failure.
    """
    p = np.asarray(polynomial, dtype=float)
    if p.ndim != 1 or len(p) < 3 or not np.isfinite(p).all() or not maximum_u > 0:
        raise ValueError('finite radial polynomial and positive maximum_u required')
    if p[0] != 0 or p[1] != 0 or p[2] <= 0:
        raise MaxClosedBoundaryError('axis-connected positive s branch requires zero constant/linear terms and positive near-axis quadratic')
    q = np.arange(2, len(p))*p[2:]
    q = np.trim_zeros(q, trim='b')
    roots = np.polynomial.polynomial.polyroots(q) if len(q) > 1 else np.empty(0, dtype=complex)
    candidates, near_complex = [], []
    for root in roots:
        real, imaginary = float(root.real), abs(float(root.imag))
        if not np.isfinite(real) or not np.isfinite(imaginary):
            raise MaxClosedBoundaryError('nonfinite derivative critical root; no branch fallback')
        if real <= 0 or real >= maximum_u:
            continue
        tolerance = 1e-7*max(1., abs(real))
        if imaginary <= tolerance:
            candidates.append(real)
        elif imaginary <= 1e-4*max(1., abs(real)):
            at_real, bound = _signed_value(q, real)
            if at_real <= bound:
                raise MaxClosedBoundaryError('uncertain near-real derivative root; no native-certificate fallback',
                    dict(root_real=real, root_imaginary=imaginary, q_at_real=float(at_real), numerical_bound=float(bound)))
            near_complex.append(dict(real=real, imaginary=imaginary, positive_q_at_real=float(at_real)))
    candidates.sort()
    groups = []
    for root in candidates:
        if groups and abs(root-np.mean(groups[-1])) <= 1e-9*max(1., abs(root)):
            groups[-1].append(root)
        else:
            groups.append([root])
    critical = [float(np.mean(group)) for group in groups]
    for group, root in zip(groups, critical):
        at_root, bound = _signed_value(q, root)
        scale = np.polynomial.polynomial.polyval(abs(root), abs(q))
        if abs(at_root) > 1e-6*max(scale, np.finfo(float).tiny):
            raise MaxClosedBoundaryError('derivative root residual is uncertain; no branch fallback',
                dict(root=root, q_at_root=float(at_root), coefficient_scale=float(scale)))
        if len(group) > 1 and at_root < -bound:
            raise MaxClosedBoundaryError('merged critical roots enclose an unresolved negative derivative interval',
                dict(root_cluster=group, q_at_cluster=float(at_root), numerical_bound=float(bound)))
    boundaries = [0.] + critical + [float(maximum_u)]
    signs = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint = .5*(left+right)
        value, bound = _signed_value(q, midpoint)
        if abs(value) <= bound:
            raise MaxClosedBoundaryError('uncertain derivative sign between critical roots; no branch fallback',
                dict(interval=[left, right], midpoint=midpoint, q_midpoint=float(value), numerical_bound=float(bound)))
        signs.append(1 if value > 0 else -1)
    if signs[0] != 1:
        raise MaxClosedBoundaryError('critical-root partition does not connect to the positive near-axis branch')
    events, first_fold = [], None
    for index, root in enumerate(critical):
        left_sign, right_sign = signs[index], signs[index+1]
        label = 'flat_nonfold' if left_sign == right_sign else ('fold_maximum' if left_sign > 0 else 'fold_minimum')
        events.append(dict(u=root, left_interval_sign=left_sign, right_interval_sign=right_sign,
                           kind=label, numerical_root_multiplicity=len(groups[index])))
        if first_fold is None and left_sign > 0 and right_sign < 0:
            first_fold = root
    cap = float(maximum_u if first_fold is None else first_fold)
    if not cap > 0 or not np.polynomial.polynomial.polyval(cap, p) > 0:
        raise MaxClosedBoundaryError('nonpositive axis-connected branch capacity')
    return dict(maximum_u=cap, reason='radius_cap' if first_fold is None else 'first_genuine_fold',
                first_fold_u=first_fold, critical_points=events,
                flat_nonfold_count=sum(event['kind'] == 'flat_nonfold' and event['u'] < cap for event in events),
                positive_near_real_complex_pairs=near_complex,
                interval_signs=signs, native_bernstein_applied=False)


def max_closed_boundary(model, phis, ntheta=256, max_radius=None, *, workers=None):
    """Freeze the largest common first-root boundary on the supplied phi/ray grid.

    A fractional 1e-6 margin is applied to the shared maximum s level. The cap
    is model.a metres unless max_radius is supplied explicitly; it is unrelated
    to fit error. Physical R>0 is mandatory. Volume is computed from exact
    polygon first-R moments, with the ray trapezoid retained as a cross-check.
    """
    nfp = int(model.nfp)
    workers_requested,workers_effective=resolve_workers(workers)
    a = float(model.a)
    phis = np.array(phis, dtype=float, copy=True)
    maximum = a if max_radius is None else float(max_radius)
    if (nfp < 1 or model.nfp != nfp or not np.isfinite(a) or a <= 0 or
            not np.isfinite(maximum) or maximum <= 0):
        raise ValueError('positive nfp, a and physical max_radius required')
    if (phis.ndim != 1 or len(phis) < 2 or not np.isfinite(phis).all() or
            not np.allclose(np.diff(phis), 2*np.pi/(nfp*len(phis)), rtol=1e-9, atol=1e-12)):
        raise ValueError('uniform geometric phis must span one field period without repeated endpoint')
    if isinstance(ntheta, bool) or int(ntheta) != ntheta or ntheta < 8:
        raise ValueError('ntheta must be integer >= 8')
    ntheta = int(ntheta)
    if len(model.coeffs) != len(model.modes) or not np.isfinite(model.coeffs).all():
        raise ValueError('finite coefficients must match scalar basis')
    if any(mode.a < 0 or mode.b < 0 or mode.a+mode.b < 2 for mode in model.modes):
        raise ValueError('axis scalar basis must contain only total degree >= 2 terms')
    theta = np.arange(ntheta)*2*np.pi/ntheta
    axis_R, axis_Z, _, _ = model.axis_at(phis)
    axis_R, axis_Z = np.asarray(axis_R, dtype=float), np.asarray(axis_Z, dtype=float)
    if (axis_R.shape != phis.shape or axis_Z.shape != phis.shape or not np.isfinite(axis_R).all() or
            not np.isfinite(axis_Z).all() or np.min(axis_R) <= 0):
        raise ValueError('axis positions must be finite, phi-aligned, at positive R')
    polynomials = _ray_polynomials(model, phis, theta)
    shape = (len(phis), ntheta)
    physical_caps=np.full(shape,maximum)
    cosine=np.cos(theta)
    negative=cosine<0
    R_limits=np.nextafter(axis_R[:,None]/(-cosine[None,negative]),0.)
    physical_caps[:,negative]=np.minimum(physical_caps[:,negative],R_limits)
    R_limited=physical_caps<maximum
    try:
        caps,folded,flat_counts,fallback_count=branch_limits(polynomials,physical_caps/a,first_connected_branch,workers=workers_effective)
    except MaxClosedBoundaryError as error:
        if 'flat_ray_index' not in error.diagnostics:
            raise
        iphi,itheta=np.unravel_index(error.diagnostics['flat_ray_index'],shape)
        raise MaxClosedBoundaryError(str(error),dict(phi_index=int(iphi),theta_index=int(itheta),
            phi=float(phis[iphi]),theta=float(theta[itheta]),physical_radius_cap_m=float(physical_caps[iphi,itheta]),
            critical_root_diagnostics=error.diagnostics)) from error
    levels=np.polynomial.polynomial.polyval(caps,polynomials.transpose(2,0,1),tensor=False)
    reasons=np.where(folded,'first_genuine_fold',np.where(R_limited,'cylindrical_R_positive_cap','radius_cap'))
    flat_count=int(flat_counts.sum());fold_count=int(folded.sum());R_cap_count=int((R_limited&~folded).sum())
    limiting = np.unravel_index(int(np.argmin(levels)), shape)
    maximum_level = float(levels[limiting])
    margin = 1e-6
    level = maximum_level*(1-margin)
    if not np.isfinite(level) or level <= 0:
        raise MaxClosedBoundaryError('no positive common closed first-root level')
    low, high = np.zeros(shape), caps.copy()
    # Vectorized Horner evaluation keeps the full 48x256 grid inexpensive.
    for _ in range(48):
        mid = .5*(low+high)
        value = polynomials[:, :, -1].copy()
        for j in range(polynomials.shape[2]-2, -1, -1):
            value = value*mid+polynomials[:, :, j]
        below = value < level
        low = np.where(below, mid, low); high = np.where(below, high, mid)
    normalized = .5*(low+high)
    radii = a*normalized
    residual = polynomials[:, :, -1].copy()
    derivative = (polynomials.shape[2]-1)*polynomials[:, :, -1]
    for j in range(polynomials.shape[2]-2, -1, -1):
        residual = residual*normalized+polynomials[:, :, j]
        if j >= 1: derivative = derivative*normalized+j*polynomials[:, :, j]
    derivative /= a
    residual -= level
    polygons = np.stack((axis_R[:, None]+radii*np.cos(theta), axis_Z[:, None]+radii*np.sin(theta)), axis=-1)
    if (not np.isfinite(polygons).all() or np.min(polygons[:, :, 0]) <= 0 or
            np.any(radii <= 0) or np.max(radii) > maximum*(1+1e-12)):
        raise MaxClosedBoundaryError('selected boundary violates finite positive R/radius geometry')
    following = np.roll(polygons, -1, axis=1)
    cross = polygons[:, :, 0]*following[:, :, 1]-following[:, :, 0]*polygons[:, :, 1]
    areas = .5*np.sum(cross, axis=1)
    moments = np.sum((polygons[:, :, 0]+following[:, :, 0])*cross, axis=1)/6
    if np.min(areas) <= 0 or np.min(moments) <= 0:
        raise MaxClosedBoundaryError('selected star-shaped polygons have invalid area/R moment')
    volume = float(2*np.pi*np.mean(moments))
    ray_volume = float(4*np.pi*np.pi*np.mean(.5*axis_R[:, None]*radii**2+np.cos(theta)*radii**3/3))
    iphi, itheta = (int(index) for index in limiting)
    branch = first_connected_branch(polynomials[limiting],float(physical_caps[limiting]/a))
    digest = hashlib.sha256()
    for array in (phis, polygons): digest.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    diagnostics = dict(
        schema='max-axis-connected-closed-s-boundary-within-a-v1',
        branch_algorithm='real critical roots of q=p_prime/u; midpoint derivative signs; first positive-to-negative fold; flat nonfold points allowed',
        native_bernstein_applied=False, magnetic_quality_gate_applied=False, fit_error_gate_applied=False,
        geometric_phi_count=len(phis), theta_count=ntheta, nfp=nfp,
        configured_max_radius_m=maximum, max_radius_m=maximum, scalar_scale_a_m=a,
        maximum_common_s_level=maximum_level, level=level, inward_s_level_fraction=margin,
        radius_min_m=float(radii.min()), radius_max_m=float(radii.max()),
        branch_cap_radius_min_m=float(a*caps.min()), branch_cap_radius_max_m=float(a*caps.max()),
        fold_limited_ray_count=int(fold_count), flat_nonfold_stationary_points_allowed=int(flat_count),
        batched_root_scalar_fallback_count=fallback_count,
        geometry_workers_requested=workers_requested, geometry_workers_effective=workers_effective,
        cylindrical_R_limited_ray_count=int(R_cap_count), uncertain_critical_root_count=0,
        selected_ds_dr_min=float(derivative.min()), selected_ds_dr_max=float(derivative.max()),
        root_residual_max=float(np.max(abs(residual))),
        full_device_volume_polygon_m3=volume, full_device_volume_ray_m3=ray_volume,
        polygon_minus_ray_relative_volume=float(volume/ray_volume-1),
        section_areas_m2=areas.tolist(), section_first_R_moments_m3=moments.tolist(),
        boundary_sha256=digest.hexdigest(), boundary_rescaled=False,
        limiting_ray=dict(phi_index=iphi, theta_index=itheta, phi=float(phis[iphi]), theta=float(theta[itheta]),
            reason=reasons[limiting], branch_cap_radius_m=float(a*caps[limiting]),
            available_s_level=maximum_level, selected_radius_m=float(radii[limiting]),
            critical_points=branch['critical_points']),
        active_capacity_normalized_radius=float(caps[limiting]),
        active_capacity_polynomial=polynomials[limiting].tolist(),
        active_capacity_source='fold' if folded[limiting] else 'R_positive_cap' if R_limited[limiting] else 'fixed_radius_cap',
        active_capacity_gap=float(np.partition(levels.ravel(),1)[1]-maximum_level),
        numerical_root_policy='unresolved near-real/merged derivative roots or interval signs cause explicit failure; no native fallback',
        topology_scope='positive axis-connected first roots give simple star-shaped sampled section polygons; finite phi/theta grid is not a global magnetic-surface certificate',
        volume_scope='full-device exact polygon first-R moments; independent radial-ray trapezoid; no target-volume adjustment',
    )
    for array in (polygons, radii, phis, theta): array.setflags(write=False)
    return dict(polygons_rz=polygons, radii=radii, phis=phis, theta=theta,
                normalized_branch_caps=caps, available_levels=levels, ds_dr=derivative,
                volume_m3=volume, volume_ray_m3=ray_volume, level=level, diagnostics=diagnostics)
