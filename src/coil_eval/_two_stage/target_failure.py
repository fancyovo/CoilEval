"""Explain target-volume failures on the sampled fitted scalar geometry."""
import numpy as np

from .boundary import MaxClosedBoundaryError, first_connected_branch
from .branches import branch_limits, resolve_workers
from .polynomial import ray_polynomials


def training_domain_error(error, stage):
    """Retain the exact radial evidence from training-domain construction."""
    details = dict(error.diagnostics)
    ray = details.get('critical_root_diagnostics', details)
    if ray.get('failure_reason') == 'invalid_near_axis_radial_polynomial' and ray.get('quadratic', 0.) < 0.:
        reason = 's_near_axis_nonmonotone'
        message = f'scalar fit {stage} decreases outwards near the magnetic axis'
    else:
        reason = ray.get('failure_reason', 's_training_domain_unavailable')
        message = f'could not construct the training domain from scalar fit {stage}'
    details.update(failure_reason=reason, failure_stage='s_training_domain', psi_fit_stage=stage)
    return MaxClosedBoundaryError(f'{message}. {error}', details)


def _capacity_volume(polynomials, caps, axis_r, theta, level, a):
    low, high = np.zeros_like(caps), caps.copy()
    powers = polynomials.transpose(2, 0, 1)
    for _ in range(48):
        radius = .5*(low+high)
        below = np.polynomial.polynomial.polyval(radius, powers, tensor=False) < level
        low = np.where(below, radius, low)
        high = np.where(below, high, radius)
    radius = .5*a*(low+high)
    return float(4*np.pi**2*np.mean(.5*axis_r[:, None]*radius**2+np.cos(theta)*radius**3/3))


def sampled_first_branch(model, cfg, *, force_guard=False):
    """Use critical roots on the native angular grids, without the inward margin.

    This is a diagnostic for the fitted scalar, with the same physical radius
    limits and volume quadrature as the target solver. It says nothing about
    the existence of a magnetic surface in the underlying field.
    """
    nphi, ntheta = cfg.flux_phi_count, cfg.flux_theta_count
    nguard = max(2*cfg.psi_n_phi, 4*cfg.psi_m_tor+1)
    fractions = np.arange(nphi)/nphi
    if nguard > nphi or force_guard:
        fractions = np.concatenate((fractions, np.arange(nguard)/nguard))
    phi = fractions*2*np.pi/model.nfp
    theta = np.arange(ntheta)*2*np.pi/ntheta
    polynomials = ray_polynomials(model, phi, theta)
    axis_r = model.axis_at(phi)[0]
    radial_cap = np.full(polynomials.shape[:2], cfg.surface_max_radius_scale)
    cosine = np.cos(theta)
    inward = cosine < 0
    radial_cap[:, inward] = np.minimum(radial_cap[:, inward], .999*axis_r[:, None]/(-cosine[None, inward]*model.a))
    evidence = dict(volume_phi_count=nphi, theta_count=ntheta, additional_phi_count=len(phi)-nphi)
    near_axis = polynomials[:, :, 2]
    if np.any(near_axis <= 1e-12):
        i, j = np.unravel_index(near_axis.argmin(), near_axis.shape)
        evidence.update(phi_rad=float(phi[i]), theta_rad=float(theta[j]),
                        near_axis_quadratic=float(near_axis[i, j]))
        evidence['capacity_source'] = ('near_axis_nonmonotone' if near_axis[i, j] < -1e-12
                                       else 'near_axis_uncertified')
        return evidence
    _, workers = resolve_workers()
    caps, folds, _, _ = branch_limits(polynomials, radial_cap, first_connected_branch, workers=workers)
    levels = np.polynomial.polynomial.polyval(caps, polynomials.transpose(2, 0, 1), tensor=False)
    i, j = np.unravel_index(levels.argmin(), levels.shape)
    level = float(levels[i, j])
    if not np.isfinite(level) or level <= 0:
        raise MaxClosedBoundaryError('no positive common first-branch level in diagnostic')
    capacity = _capacity_volume(polynomials[:nphi], caps[:nphi], axis_r[:nphi], theta, level, model.a)
    limiter = 'configured_radius' if radial_cap[i, j] == cfg.surface_max_radius_scale else 'positive_R_limit'
    if folds[i, j]:
        # Independently inspect the winning ray and retain its sign transition.
        branch = first_connected_branch(polynomials[i, j], radial_cap[i, j])
        if branch['reason'] != 'first_genuine_fold':
            raise MaxClosedBoundaryError('winning radial fold did not reproduce')
        limiter = 'radial_fold'
    evidence.update(capacity_source=limiter, capacity_volume_m3=capacity, capacity_level=level,
                    phi_rad=float(phi[i]), theta_rad=float(theta[j]), radius_m=float(caps[i, j]*model.a))
    return evidence


def explain_target_failure(diagnostics, model, cfg, stage, *, force_guard=False):
    """Add a specific cause only when the fitted coefficients support it."""
    original = diagnostics['error_message']
    maximum = diagnostics.get('target_volume_maximum', 0.)
    target = cfg.target_volume
    diagnostics.update(failure_stage='s_target_surface', psi_fit_stage=stage)
    diagnostics.setdefault('native_error_message', original)
    if maximum >= target > 0:
        diagnostics.update(failure_reason='target_volume_inverse_failed_within_capacity',
                           error_message='target-volume inverse solve failed within the fitted scalar branch capacity')
        return
    try:
        detail = sampled_first_branch(model, cfg, force_guard=force_guard)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        detail = dict(capacity_source='unresolved', diagnostic_error=str(error))
    diagnostics['target_branch_diagnostic'] = detail
    source = detail['capacity_source']
    capacity = detail.get('capacity_volume_m3')
    prefix = f'scalar fit {stage}'
    if source == 'near_axis_nonmonotone':
        reason = 's_near_axis_nonmonotone'
        message = f'{prefix} decreases outwards near the magnetic axis on the target geometry grid'
    elif capacity is not None and capacity < target*(1-1e-6):
        if source == 'radial_fold':
            reason = 's_radial_fold_capacity'
            message = f'{prefix} is radially nonmonotone and its first fold limits the target surface'
        else:
            reason = 's_radius_cap_capacity'
            label = 'configured fitting radius' if source == 'configured_radius' else 'positive cylindrical radius limit'
            message = f'{prefix} reaches the {label} before the target surface'
        message += f' ({capacity:.9g} m^3 available, {target:.9g} m^3 requested)'
    else:
        reason = 's_radial_branch_uncertified'
        message = f'{prefix} has no certified radial branch reaching the requested volume'
    previous = diagnostics.get('failure_reason')
    if previous and previous != reason:
        diagnostics['legacy_failure_reason'] = previous
    diagnostics.update(failure_reason=reason, error_message=message)
