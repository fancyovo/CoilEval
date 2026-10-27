"""Independent field-derivative QS diagnostics on solved Simsopt surfaces.

The triple-product reference differentiates ``B dot grad(|B|)`` along the
surface. It uses the analytic Cartesian field Jacobian, independently of the
B-only second-angular-derivative formula used by the tensor metric kernels.
"""
from __future__ import annotations

import time
from operator import index

import numpy as np

from .geometry import surface_from_dofs
from .state import INITIALIZER_KIND, SOURCE_KIND, STANDARD_KIND, require_kind


TWOPI = 2.0 * np.pi


def _integer(value, name, *, positive=False):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if positive and result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _grid_shape(shape):
    if len(shape) != 2 or any(size < 5 or size % 2 == 0 for size in shape):
        raise ValueError("reference grids require two odd dimensions >= 5")
    return shape


def _turn_derivative(values, *, axis, periods):
    """Differentiate an endpoint-free uniform grid with respect to turns."""
    count = values.shape[axis]
    modes = np.fft.fftfreq(count, d=1.0 / count)
    reshape = [1] * values.ndim
    reshape[axis] = count
    multiplier = (1j * TWOPI * periods * modes).reshape(reshape)
    return np.fft.ifft(np.fft.fft(values, axis=axis) * multiplier, axis=axis).real


def _rms(values, weights=None):
    if weights is None:
        return float(np.sqrt(np.mean(np.square(values))))
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def analyze_qs_arrays(*, xyz, gphi, gtheta, B, dBdx, iota, G_turn, nfp,
                      major_radius, helicity=None):
    """Compute f_B and a field-derivative f_T reference on one surface grid.

    ``xyz``, ``gphi`` and ``gtheta`` have shape ``(nphi, ntheta, 3)``. The
    tangents differentiate angles measured in turns: phi covers ``1/nfp`` and
    theta covers one turn. Both grid sizes must be odd and >= 5. Uniform grid
    offsets are allowed. ``dBdx[..., i, j]`` means ``d B_i / d x_j``.

    ``G_turn`` is the Simsopt vacuum-surface convention; ``G_rad=G_turn/(2*pi)``.
    ``major_radius`` is a positive explicit normalization length in metres.
    Helicity is the full-torus integer pair ``(M,N)`` and defaults to ``(1,nfp)``.

    f_B is the square root of asymmetric Fourier energy divided by total
    Fourier energy. f_T is the flux-shell RMS of ``major_radius**2*T/|B|**4``.
    The shell measure is proportional to ``abs(G_rad)/|B|**2``. The returned
    point arrays support independent same-point comparisons; this low-level
    function does not classify a geometric surface as physically acceptable.
    """
    nfp = _integer(nfp, "nfp", positive=True)
    if helicity is None:
        helicity = (1, nfp)
    if not isinstance(helicity, (tuple, list, np.ndarray)) or len(helicity) != 2:
        raise ValueError("helicity must contain two integers")
    M, N = (_integer(value, "helicity") for value in helicity)
    if M == N == 0:
        raise ValueError("helicity cannot be (0,0)")
    radius, iota, G_turn = map(float, (major_radius, iota, G_turn))
    if not np.isfinite([radius, iota, G_turn]).all() or radius <= 0 or G_turn == 0:
        raise ValueError("major_radius must be positive and iota/G finite with nonzero G")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (nphi,ntheta,3)")
    shape = _grid_shape(xyz.shape[:2])
    arrays = [np.asarray(value, dtype=np.float64) for value in (gphi, gtheta, B, dBdx)]
    gp, gt, field, jacobian = arrays
    if gp.shape != xyz.shape or gt.shape != xyz.shape:
        raise ValueError("surface tangents must match xyz shape")
    if field.shape not in (xyz.shape, (np.prod(shape), 3)):
        raise ValueError("field samples must match the surface grid")
    if jacobian.shape not in ((*shape, 3, 3), (np.prod(shape), 3, 3)):
        raise ValueError("field Jacobians must have shape (...,3,3)")
    field = field.reshape(xyz.shape)
    jacobian = jacobian.reshape(*shape, 3, 3)
    if not all(np.isfinite(value).all() for value in (xyz, gp, gt, field, jacobian)):
        raise ValueError("reference arrays must be finite")
    magnitude = np.linalg.norm(field, axis=-1)
    area = np.linalg.norm(np.cross(gt, gp), axis=-1)
    if np.any(magnitude <= 0) or np.any(area <= 0):
        raise ValueError("reference field and surface normal must be nonzero")

    grad_B = np.einsum("...ij,...i->...j", jacobian, field) / magnitude[..., None]
    B_phi = np.sum(grad_B * gp, axis=-1)
    B_theta = np.sum(grad_B * gt, axis=-1)
    parallel_B_derivative = np.sum(field * grad_B, axis=-1)
    q_phi = _turn_derivative(parallel_B_derivative, axis=0, periods=nfp)
    q_theta = _turn_derivative(parallel_B_derivative, axis=1, periods=1)

    # grad(psi)=|B|^2/(2*pi*G_turn) (x_theta cross x_phi).
    # The vector identity (a cross b) cross c=b(a dot c)-a(b dot c)
    # contracts the tangent gradient of q without a radial finite difference.
    triple = magnitude**2 / (TWOPI * G_turn) * (B_theta * q_phi - B_phi * q_theta)
    normalized = radius**2 * triple / magnitude**4
    G_rad = G_turn / TWOPI
    shell_weights = abs(G_rad) / magnitude**2

    spectrum = np.fft.fft2(magnitude) / magnitude.size
    toroidal = np.rint(np.fft.fftfreq(shape[0], d=1.0 / shape[0])).astype(int)
    poloidal = np.rint(np.fft.fftfreq(shape[1], d=1.0 / shape[1])).astype(int)
    # FFT uses exp(+i(k_phi*nfp*phi + m*theta)); conventional n=-k_phi.
    symmetric = poloidal[None, :] * N + toroidal[:, None] * nfp * M == 0
    energy = np.abs(spectrum)**2
    f_B = float(np.sqrt(np.sum(energy[~symmetric]) / np.sum(energy)))

    tangent = gp + iota * gt
    residual = G_turn * field - magnitude[..., None]**2 * tangent
    relative_equation = np.linalg.norm(residual, axis=-1) / (abs(G_turn) * magnitude)
    normal_field = abs(np.sum(field * np.cross(gp, gt), axis=-1)) / (magnitude * area)
    numerical_phi = _turn_derivative(magnitude, axis=0, periods=nfp)
    numerical_theta = _turn_derivative(magnitude, axis=1, periods=1)
    derivative_norm = np.sqrt(np.mean(B_phi**2 + B_theta**2))
    derivative_difference = np.sqrt(np.mean((numerical_phi-B_phi)**2 + (numerical_theta-B_theta)**2))
    return {
        "f_B": f_B,
        "f_T": _rms(normalized, shell_weights),
        "f_T_raw": triple,
        "f_T_normalized": normalized,
        "B_magnitude": magnitude,
        "spectrum": spectrum,
        "shell_weights": shell_weights,
        "relative_boozer_equation_error_rms": _rms(relative_equation),
        "normal_B_sine_p95": float(np.percentile(normal_field, 95)),
        "field_vs_spectral_B_derivative_relative_rms": float(derivative_difference/max(derivative_norm, 1e-300)),
        "field_vs_spectral_B_derivative_absolute_rms": float(derivative_difference),
        "point_count": int(magnitude.size),
        "iota": iota,
        "G_rad": G_rad,
        "G_turn": G_turn,
        "major_radius_m": radius,
        "helicity": [M, N],
        "f_B_definition": "RMS asymmetric Fourier energy fraction; uniform Boozer angles",
        "f_T_definition": "flux-shell RMS of R^2 * (grad(psi) cross grad(B)) dot grad(B dot grad(B)) / B^4",
        "derivative_method": "analytic Cartesian field Jacobian and spectral angular derivative of B dot grad(B)",
    }


def _field_and_jacobian(field, points):
    if all(hasattr(field, name) for name in ("set_points", "B", "dB_by_dX")):
        field.set_points(points)
        return field.B(), field.dB_by_dX()
    if callable(field):
        result = field(points)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result
    raise TypeError("field must provide Simsopt methods or return (B,dBdx)")


def _size_pair(size):
    if np.isscalar(size):
        value = _integer(size, "grid size", positive=True)
        return _grid_shape((value, value))
    if len(size) != 2:
        raise ValueError("each grid size must be an integer or (nphi,ntheta)")
    return _grid_shape(tuple(_integer(value, "grid size", positive=True) for value in size))


def surface_qs_metrics(surface, field, *, major_radius, helicity=None,
                       sizes=(65, 97, 129), convergence_rtol=0.02,
                       convergence_atol=1e-10, return_arrays=False):
    """Evaluate f_B/f_T on an accepted standard Simsopt vacuum surface.

    Rechecks self-intersections on the original Fourier surface, including for
    historical accepted records. Confirmed intersections raise ``ValueError``;
    inconclusive searches retain a quality warning. Mesh convergence describes
    quadrature on this fixed surface and is separate from surface acceptance.
    ``return_arrays=True`` adds the final grid's inputs and point diagnostics.
    """
    require_kind(surface.kind, STANDARD_KIND, "reference QS surface")
    require_kind(surface.seed.kind, INITIALIZER_KIND, "reference QS initializer")
    require_kind(surface.seed.source_kind, SOURCE_KIND, "reference QS source")
    if not surface.accepted:
        raise ValueError("reference QS requires an accepted standard surface")
    if not np.isfinite([convergence_rtol, convergence_atol]).all() or min(convergence_rtol, convergence_atol) < 0:
        raise ValueError("convergence tolerances must be finite and nonnegative")
    sizes = tuple(_size_pair(size) for size in sizes)
    if not sizes:
        raise ValueError("at least one reference grid is required")
    if any(next_size[0] <= size[0] or next_size[1] <= size[1]
           for size, next_size in zip(sizes[:-1], sizes[1:])):
        raise ValueError("reference grid dimensions must increase strictly")
    start = time.perf_counter()
    from .intersections import detect_self_intersection
    check_grid = surface_from_dofs(surface.dofs, nfp=surface.seed.nfp, order=surface.seed.order,
                                   nphi=sizes[-1][0], ntheta=sizes[-1][1], stellsym=surface.seed.stellsym)
    intersection = detect_self_intersection(check_grid)
    if intersection.get("confirmed_self_intersection"):
        raise ValueError("reference QS surface has a confirmed self-intersection")
    rows = []
    arrays = None
    for nphi, ntheta in sizes:
        grid = surface_from_dofs(surface.dofs, nfp=surface.seed.nfp, order=surface.seed.order,
                                 nphi=nphi, ntheta=ntheta, stellsym=surface.seed.stellsym, offset=(.5, .5))
        xyz = np.asarray(grid.gamma())
        B, derivative = _field_and_jacobian(field, np.ascontiguousarray(xyz.reshape(-1, 3)))
        inputs = dict(xyz=xyz, gphi=grid.gammadash1(), gtheta=grid.gammadash2(), B=B,
                      dBdx=derivative, iota=surface.iota, G_turn=surface.G,
                      nfp=surface.seed.nfp, major_radius=major_radius, helicity=helicity)
        result = analyze_qs_arrays(**inputs)
        rows.append(dict(nphi=nphi, ntheta=ntheta,
                         **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)}))
        if return_arrays:
            arrays = {**inputs, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)}}
    convergence = {}
    if len(rows) > 1:
        for key in ("f_B", "f_T"):
            previous, final = rows[-2][key], rows[-1][key]
            difference = abs(final - previous)
            scale = max(abs(previous), abs(final))
            convergence[key] = {
                "absolute_change": difference,
                "relative_change": difference/max(scale, 1e-300),
                "passed": bool(difference <= convergence_atol + convergence_rtol * scale),
            }
    quality = surface.quality
    if intersection.get("status") != "not_detected_in_sampled_sections":
        quality = "accepted_with_quality_warning"
    output = dict(rows[-1], grids=rows, accepted=True, quality=quality,
                  self_intersection=intersection, actual_volume_m3=abs(float(grid.volume())),
                  convergence=convergence, convergence_rtol=float(convergence_rtol),
                  convergence_atol=float(convergence_atol),
                  quadrature_converged=all(item["passed"] for item in convergence.values()) if convergence else None,
                  seconds=time.perf_counter()-start)
    if return_arrays:
        output["arrays"] = arrays
    return output
