"""Largest sampled closed first-root s boundary within a physical radius cap.

The axis-connected radial branch ends at its first genuine positive-to-negative
derivative fold or at the physical radius cap. Flat stationary points with no
sign reversal are allowed. The native strict Bernstein certificate is not used.
No magnetic field, fit-error, native validity or magnetic-quality gate is read.
"""


import hashlib
import numpy as np

def _ray_polynomials(model, phis, theta):
    degree = max([2] + [int(mode.a + mode.b) for mode in model.modes])
    polynomials = np.zeros((len(phis), len(theta), degree + 1))
    cosine, sine = np.cos(theta), np.sin(theta)
    cp = np.asarray([cosine**power for power in range(degree + 1)])
    sp = np.asarray([sine**power for power in range(degree + 1)])
    polynomials[:, :, 2] = cosine**2
    for coefficient, mode in zip(model.coeffs, model.modes):
        if coefficient == 0:
            continue
        angle = int(mode.m) * int(model.nfp) * phis
        if mode.kind not in ('cos', 'sin'):
            raise ValueError('mode.kind must be cos or sin')
        trig = np.cos(angle) if mode.kind == 'cos' else np.sin(angle)
        polynomials[:, :, int(mode.a + mode.b)] += (
            float(coefficient) * trig[:, None] * cp[int(mode.a)] * sp[int(mode.b)])
    return polynomials



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


def max_closed_boundary(model, phis, ntheta=256, max_radius=None):
    """Freeze the largest common first-root boundary on the supplied phi/ray grid.

    A fractional 1e-6 margin is applied to the shared maximum s level. The cap
    is model.a metres unless max_radius is supplied explicitly; it is unrelated
    to fit error. Physical R>0 is mandatory. Volume is computed from exact
    polygon first-R moments, with the ray trapezoid retained as a cross-check.
    """
    nfp = int(model.nfp)
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
    caps, levels = np.empty(shape), np.empty(shape)
    reasons, branches = {}, {}
    flat_count = fold_count = R_cap_count = 0
    for iphi in range(len(phis)):
        for itheta, angle in enumerate(theta):
            physical_cap = maximum
            R_limited = False
            if np.cos(angle) < 0:
                R_limit = np.nextafter(axis_R[iphi]/(-np.cos(angle)), 0.)
                if R_limit < physical_cap:
                    physical_cap, R_limited = float(R_limit), True
            try:
                branch = first_connected_branch(polynomials[iphi, itheta], physical_cap/a)
            except MaxClosedBoundaryError as error:
                raise MaxClosedBoundaryError(str(error), dict(phi_index=iphi, theta_index=itheta,
                    phi=float(phis[iphi]), theta=float(angle), physical_radius_cap_m=physical_cap,
                    critical_root_diagnostics=error.diagnostics)) from error
            caps[iphi, itheta] = branch['maximum_u']
            levels[iphi, itheta] = np.polynomial.polynomial.polyval(branch['maximum_u'], polynomials[iphi, itheta])
            reason = branch['reason']
            if reason == 'radius_cap' and R_limited:
                reason = 'cylindrical_R_positive_cap'; R_cap_count += 1
            reasons[iphi, itheta] = reason
            branches[iphi, itheta] = branch
            flat_count += branch['flat_nonfold_count']
            fold_count += branch['first_fold_u'] is not None
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
    branch = branches[limiting]
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
        numerical_root_policy='unresolved near-real/merged derivative roots or interval signs cause explicit failure; no native fallback',
        topology_scope='positive axis-connected first roots give simple star-shaped sampled section polygons; finite phi/theta grid is not a global magnetic-surface certificate',
        volume_scope='full-device exact polygon first-R moments; independent radial-ray trapezoid; no target-volume adjustment',
    )
    for array in (polygons, radii, phis, theta): array.setflags(write=False)
    return dict(polygons_rz=polygons, radii=radii, phis=phis, theta=theta,
                volume_m3=volume, volume_ray_m3=ray_volume, level=level, diagnostics=diagnostics)


"""Deterministic Cartesian cross-section sampling inside a frozen boundary.

This module deliberately knows nothing about the original s fit or magnetic
field.  A caller freezes one polygon at each *geometric* phi and passes those
polygons here.  All subsequent point decisions depend only on these arrays.

The lattice is uniform in physical (R, Z) at uniformly spaced phi.  Equal rows
therefore represent dR dZ dphi, not physical volume R dR dZ dphi.  Both weights
are returned explicitly; choosing either fitting objective is the caller's job.
"""



import hashlib
import struct
from typing import Sequence

import numpy as np


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack("<" + "Q" * value.ndim, *value.shape))
        digest.update(value.tobytes())
    return digest.hexdigest()


def polygon_area_and_first_R_moment(polygon: np.ndarray) -> tuple[float, float]:
    """Return positive area and integral R dR dZ, independent of orientation."""
    polygon = np.asarray(polygon, dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * np.sum(cross)
    signed_moment = np.sum((polygon[:, 0] + following[:, 0]) * cross) / 6.0
    return float(abs(signed_area)), float(np.sign(signed_area) * signed_moment)


def polygon_contains_strict(
    polygon: np.ndarray, points: np.ndarray, *, boundary_tolerance: float | None = None,
    max_pairs: int = 1_000_000,
) -> np.ndarray:
    """Even/odd containment for a simple polygon, explicitly excluding edges.

    Works for concave polygons of either orientation.  The boundary tolerance
    is a physical distance.  Chunking bounds temporary point-by-edge storage.
    Input polygons must be non-self-intersecting; this is not a topology test.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must have shape (vertices >= 3, 2)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.all(np.isfinite(polygon)) or not np.all(np.isfinite(points)):
        raise ValueError("polygon and points must be finite")
    if boundary_tolerance is None:
        boundary_tolerance = 64 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(polygon))), float(np.max(np.ptp(polygon, axis=0))))
    if boundary_tolerance < 0 or not np.isfinite(boundary_tolerance):
        raise ValueError("boundary_tolerance must be finite and nonnegative")
    start = polygon
    stop = np.roll(polygon, -1, axis=0)
    edge = stop - start
    length = np.linalg.norm(edge, axis=1)
    nonzero = length > 0
    start, stop, edge, length = start[nonzero], stop[nonzero], edge[nonzero], length[nonzero]
    if len(start) < 3:
        raise ValueError("polygon needs at least three nonzero edges")
    chunk_size = max(1, int(max_pairs) // len(start))
    result = np.zeros(len(points), dtype=bool)
    for begin in range(0, len(points), chunk_size):
        current = points[begin:begin + chunk_size]
        dx = current[:, None, 0] - start[None, :, 0]
        dz = current[:, None, 1] - start[None, :, 1]
        vertical_hit = ((start[None, :, 1] > current[:, None, 1]) !=
                        (stop[None, :, 1] > current[:, None, 1]))
        xhit = np.zeros_like(dx)
        np.divide(dz * edge[None, :, 0], edge[None, :, 1], out=xhit,
                  where=edge[None, :, 1] != 0)
        xhit += start[None, :, 0]
        inside = np.sum(vertical_hit & (current[:, None, 0] < xhit), axis=1) % 2 == 1
        cross_distance_numerator = dx * edge[None, :, 1] - dz * edge[None, :, 0]
        along = dx * edge[None, :, 0] + dz * edge[None, :, 1]
        on_edge = ((np.abs(cross_distance_numerator) <= boundary_tolerance * length) &
                   (along >= -boundary_tolerance * length) &
                   (along <= length**2 + boundary_tolerance * length)).any(axis=1)
        result[begin:begin + len(current)] = inside & ~on_edge
    return result


def _freeze_sections(polygons_rz: Sequence[np.ndarray], phis: np.ndarray, nfp: int):
    if isinstance(nfp, bool) or int(nfp) != nfp or nfp < 1:
        raise ValueError("nfp must be a positive integer")
    phis = np.array(phis, dtype=np.float64, copy=True)
    if phis.ndim != 1 or len(phis) < 2 or not np.all(np.isfinite(phis)):
        raise ValueError("phis must be a finite vector with >= 2 sections")
    if len(polygons_rz) != len(phis):
        raise ValueError("one frozen polygon is required per phi")
    period = 2 * np.pi / int(nfp)
    dphi = period / len(phis)
    # A physical-phi lattice over one complete field period.  Do not silently
    # interpret the caller's Boozer angle or repeated endpoint as geometric phi.
    if not np.allclose(np.diff(phis), dphi, rtol=1e-9, atol=1e-12):
        raise ValueError("phis must be uniformly spaced over exactly one field period, endpoint excluded")
    polygons = []
    areas, moments = [], []
    for polygon in polygons_rz:
        polygon = np.array(polygon, dtype=np.float64, copy=True)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("each polygon must have shape (vertices >= 3, 2)")
        if not np.all(np.isfinite(polygon)) or np.min(polygon[:, 0]) <= 0:
            raise ValueError("frozen polygons must be finite and strictly at R > 0")
        if np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        area, moment = polygon_area_and_first_R_moment(polygon)
        if area <= np.finfo(float).eps or moment <= 0:
            raise ValueError("frozen polygon has zero or numerically degenerate area")
        polygons.append(polygon)
        areas.append(area)
        moments.append(moment)
    return polygons, phis, dphi, np.asarray(areas), np.asarray(moments)


def _lattice_in_polygon(polygon: np.ndarray, spacing: float, shift: np.ndarray) -> np.ndarray:
    low = np.min(polygon, axis=0)
    high = np.max(polygon, axis=0)
    first = np.ceil(low / spacing - shift).astype(np.int64)
    last = np.floor(high / spacing - shift).astype(np.int64)
    r = spacing * (np.arange(first[0], last[0] + 1) + shift[0])
    z = spacing * (np.arange(first[1], last[1] + 1) + shift[1])
    rr, zz = np.meshgrid(r, z, indexing="ij")
    points = np.column_stack((rr.ravel(), zz.ravel()))
    return np.ascontiguousarray(points[polygon_contains_strict(polygon, points)])


def _evaluate_lattice(polygons, spacing, shift):
    sections = [_lattice_in_polygon(polygon, spacing, shift) for polygon in polygons]
    return sections, sum(len(section) for section in sections)


def mask_frozen_sections(polygons_rz, rz: np.ndarray, section_index: np.ndarray) -> np.ndarray:
    """Classify a fixed point superset against another frozen section boundary.

    This enables common-point comparisons between a target-V boundary and a
    1.2V fit domain without resampling either fitted s.  Section indices must
    refer to the same geometric-phi planes as the supplied polygon sequence.
    """
    rz = np.asarray(rz, dtype=np.float64)
    section_index = np.asarray(section_index)
    if rz.ndim != 2 or rz.shape[1] != 2 or section_index.shape != (len(rz),):
        raise ValueError("rz and section_index must have aligned shapes (N,2) and (N,)")
    if not np.issubdtype(section_index.dtype, np.integer):
        raise ValueError("section_index must contain integers")
    if np.any(section_index < 0) or np.any(section_index >= len(polygons_rz)):
        raise ValueError("section_index is outside the frozen polygon sequence")
    mask = np.zeros(len(rz), dtype=bool)
    for index, polygon in enumerate(polygons_rz):
        select = section_index == index
        mask[select] = polygon_contains_strict(polygon, rz[select])
    return mask


def sample_frozen_sections(
    polygons_rz: Sequence[np.ndarray], phis: np.ndarray, target_count: int = 82368,
    *, nfp: int = 4, count_tolerance: float = 0.05,
    lattice_shift: tuple[float, float] = (0.5, 0.5), max_refinements: int = 10,
    spacing: float | None = None,
) -> dict:
    """Sample an isotropic physical R-Z lattice masked by frozen section polygons.

    A common spacing starts at sqrt(sum(section areas) / target_count).  If
    necessary, deterministic density adjustments retain a complete lattice at
    the chosen spacing; points are never randomly thinned.  A caller supplying
    ``spacing`` fixes it exactly (for an independent shifted holdout) and opts
    out of count adjustment.  Failure to reach the requested count tolerance in
    automatic mode raises rather than silently returning a sparse sample.

    Returns contiguous arrays ``xyz``, ``rz``, ``phi``, ``section_index``,
    ``weights_equal``, ``weights_volume`` (= R), ``cell_volume_weights``
    (= R h^2 dphi for one field period), ``cell_volume_weights_full``
    (= nfp times one-period weights), plus ``spacing`` and JSON metadata.
    Boundary arrays and point hashes are included for frozen-run provenance.
    No s values, fitted labels, or original fit parameters are consulted here.
    """
    polygons, phis, dphi, areas, moments = _freeze_sections(polygons_rz, phis, nfp)
    if isinstance(target_count, bool) or int(target_count) != target_count or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    target_count = int(target_count)
    if not np.isfinite(count_tolerance) or not 0 < count_tolerance < 1:
        raise ValueError("count_tolerance must lie strictly between zero and one")
    if isinstance(max_refinements, bool) or int(max_refinements) != max_refinements or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    shift = np.asarray(lattice_shift, dtype=np.float64)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)) or np.any((shift < 0) | (shift >= 1)):
        raise ValueError("lattice_shift must contain two finite phases in [0, 1)")
    fixed_spacing = spacing is not None
    h = float(spacing) if fixed_spacing else float(np.sqrt(np.sum(areas) / target_count))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("spacing must be finite and positive")
    history = []
    best = None
    for iteration in range(1 if fixed_spacing else int(max_refinements) + 1):
        section_points, count = _evaluate_lattice(polygons, h, shift)
        error = abs(count / target_count - 1)
        history.append({"spacing_m": h, "point_count": count, "relative_count_error": error})
        if best is None or error < best[0]:
            best = (error, h, section_points)
        if fixed_spacing or (error <= count_tolerance and all(len(p) for p in section_points)):
            break
        h *= np.sqrt(max(count, 1) / target_count)
    error, h, section_points = best
    counts = np.asarray([len(points) for points in section_points], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("selected spacing leaves a frozen section without points")
    if not fixed_spacing and error > count_tolerance:
        raise ValueError(f"could not reach count tolerance: target={target_count}, actual={counts.sum()}, relative_error={error:.6g}")
    section_index = np.repeat(np.arange(len(phis), dtype=np.int64), counts)
    rz = np.ascontiguousarray(np.concatenate(section_points))
    point_phi = np.ascontiguousarray(phis[section_index])
    xyz = np.ascontiguousarray(np.column_stack((
        rz[:, 0] * np.cos(point_phi), rz[:, 0] * np.sin(point_phi), rz[:, 1])))
    weights_equal = np.ones(len(rz), dtype=np.float64)
    weights_volume = np.ascontiguousarray(rz[:, 0].copy())
    cell_volume_weights = np.ascontiguousarray(weights_volume * h * h * dphi)
    estimated_section_areas = counts * h * h
    estimated_section_moments = np.asarray([np.sum(points[:, 0]) * h * h for points in section_points])
    frozen_hash = _hash_arrays(phis, *polygons)
    metadata = {
        "schema": "frozen-irregular-section-lattice-v1",
        "coordinate_convention": "R-Z metres; uniformly spaced geometric phi radians over one field period",
        "nfp": int(nfp), "section_count": len(phis), "dphi_rad": float(dphi),
        "target_count": target_count, "actual_count": len(rz),
        "relative_count_error": float(error), "count_tolerance": float(count_tolerance),
        "spacing_m": h, "fixed_spacing": fixed_spacing, "lattice_shift": shift.tolist(),
        "lattice_origin_RZ_m": [0.0, 0.0],
        "sampling_measure": "uniform R-Z lattice at uniform geometric phi; no thinning",
        "equal_row_measure": "dR dZ dphi", "physical_volume_measure": "R dR dZ dphi",
        "frozen_boundary_sha256": frozen_hash,
        "points_sha256": _hash_arrays(xyz, point_phi, section_index),
        "section_point_counts": counts.tolist(), "section_areas_m2": areas.tolist(),
        "section_first_R_moments_m3": moments.tolist(),
        "section_lattice_area_m2": estimated_section_areas.tolist(),
        "section_lattice_area_relative_error": (estimated_section_areas / areas - 1).tolist(),
        "polygon_field_period_volume_m3": float(dphi * np.sum(moments)),
        "lattice_field_period_volume_m3": float(np.sum(cell_volume_weights)),
        "polygon_full_device_volume_m3": float(nfp * dphi * np.sum(moments)),
        "lattice_full_device_volume_m3": float(nfp * np.sum(cell_volume_weights)),
        "cell_volume_weights_scope": "one field period; cell_volume_weights_full multiplies by nfp",
        "lattice_volume_relative_error": float(np.sum(estimated_section_moments) / np.sum(moments) - 1),
        "strict_inside_fraction": 1.0,
        "within_section_minimum_spacing_lower_bound_m": h,
        "refinement_history": history,
        "topology_scope": "simple frozen section polygons assumed; no global surface topology certificate",
    }
    return {
        "xyz": xyz, "rz": rz, "phi": point_phi, "section_index": section_index,
        "weights_equal": weights_equal, "weights_volume": weights_volume,
        "cell_volume_weights": cell_volume_weights,
        "cell_volume_weights_full": np.ascontiguousarray(nfp * cell_volume_weights),
        "spacing": h, "metadata": metadata,
    }


def shifted_holdout(polygons_rz, phis, training_sample: dict, *, nfp: int = 4) -> dict:
    """Independent interlaced lattice at the training spacing, with no old s use.

    Both R and Z phases are shifted by h/2.  This is an independent collocation
    set inside the *same* frozen polygons, not a new fit or a radial contraction.
    It shares the geometric-phi planes; that limitation is recorded explicitly.
    """
    old_shift = np.asarray(training_sample["metadata"]["lattice_shift"])
    new_shift = tuple((old_shift + 0.5) % 1)
    result = sample_frozen_sections(
        polygons_rz, phis, target_count=training_sample["metadata"]["target_count"],
        nfp=nfp, spacing=training_sample["spacing"], lattice_shift=new_shift)
    if result["metadata"]["frozen_boundary_sha256"] != training_sample["metadata"]["frozen_boundary_sha256"]:
        raise ValueError("holdout boundary differs from the frozen training boundary")
    result["metadata"]["holdout"] = {
        "kind": "half-cell shifted R-Z lattice", "shares_geometric_phi_planes": True,
        "training_points_sha256": training_sample["metadata"]["points_sha256"],
    }
    return result
