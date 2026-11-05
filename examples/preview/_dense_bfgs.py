"""Dense inverse-BFGS and value-only Armijo policy for coil coordinates.

The native worker persists the entire inverse matrix.  After an accepted
point it obtains the next point's gradient once, forms y=g_new-g_old, and then
updates this matrix.  Line-search candidates need formal values only.  No
Hessian finite-difference query, limited-memory approximation, or extra trial
gradient is hidden in this NumPy-only module.
"""
from __future__ import annotations

import numpy as np


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and positive')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or len(value) == 0 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite nonempty vector')
    return value


def _norm(value):
    value = np.asarray(value, dtype=np.float64)
    maximum = float(np.max(abs(value)))
    return 0. if maximum == 0 else float(maximum * np.linalg.norm(value / maximum))


def initial_inverse(n=112, scale=1.):
    """The complete n by n inverse matrix; scale is in normalized coordinates."""
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 1:
        raise ValueError('dimension must be a positive integer')
    return np.eye(int(n), dtype=np.float64) * _positive(scale, 'inverse scale')


def _inverse_or_reset(inverse, n, reset_scale):
    matrix = np.asarray(inverse, dtype=np.float64)
    if matrix.shape != (n, n):
        raise ValueError('complete square inverse matrix must match the coordinate dimension')
    if not np.isfinite(matrix).all():
        return initial_inverse(n, reset_scale), 'nonfinite_inverse'
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-13):
        return initial_inverse(n, reset_scale), 'asymmetric_inverse'
    matrix = (matrix + matrix.T) * .5
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return initial_inverse(n, reset_scale), 'nonpositive_definite_inverse'
    return matrix, None


def descent_direction(inverse, gradient, max_norm=.05, *, reset_scale=1.):
    """Return a descent direction and its validated, possibly reset inverse.

    A global L2 bound avoids a huge first step from H0=I.  Armijo, rather than
    this bound, selects the accepted step length.  Zero gradient yields a zero
    direction and explicit diagnostic; it does not request optimizer stopping.
    """
    gradient = _vector(gradient, 'gradient')
    bound = _positive(max_norm, 'maximum direction norm')
    reset_scale = _positive(reset_scale, 'reset scale')
    matrix, reason = _inverse_or_reset(inverse, len(gradient), reset_scale)
    with np.errstate(over='ignore', invalid='ignore'):
        direction = -(matrix @ gradient)
        slope = float(gradient @ direction)
    gradient_norm, raw_norm = _norm(gradient), _norm(direction)
    if gradient_norm == 0:
        return dict(inverse=matrix, direction=np.zeros_like(gradient),
                    directional_derivative=0., raw_norm=0., norm=0., capped=False,
                    reset=reason is not None, reason=reason or 'zero_gradient')
    # An almost orthogonal direction is unsafe when finite precision has
    # destroyed the useful descent angle even if its dot product is negative.
    if (not np.isfinite(direction).all() or not np.isfinite(slope) or
            not np.isfinite(raw_norm) or raw_norm == 0 or
            slope >= -1e-14 * gradient_norm * raw_norm):
        matrix = initial_inverse(len(gradient), reset_scale)
        direction = -reset_scale * gradient
        raw_norm = _norm(direction)
        reason = 'non_descent_direction'
    capped = raw_norm > bound
    if capped:
        direction *= bound / raw_norm
    slope = float(gradient @ direction)
    if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
        raise ValueError('cannot represent a finite descent direction')
    return dict(inverse=matrix, direction=direction,
                directional_derivative=slope, raw_norm=raw_norm,
                norm=_norm(direction), capped=bool(capped),
                reset=reason is not None, reason=reason or 'inverse_bfgs_descent')


def inverse_bfgs_update(inverse, s, y, damping=.2, *, reset_scale=1.):
    """Powell-damped full inverse-BFGS update, with explicit curvature guards.

    B*s is solved from Hinv*(B*s)=s; no native Hessian is evaluated.  For
    s.T*y < damping*s.T*B*s, replace y with a convex combination of y and B*s.
    The resulting positive curvature preserves SPD.  Unusable pairs reset the
    matrix instead of inventing a tiny denominator or applying an indefinite
    rank update.  Neither caller input is mutated.
    """
    s, y = _vector(s, 'accepted displacement'), _vector(y, 'gradient change')
    if s.shape != y.shape:
        raise ValueError('displacement and gradient change dimensions must match')
    damping = float(damping)
    if not np.isfinite(damping) or not 0 < damping < 1:
        raise ValueError('Powell damping must lie strictly between zero and one')
    reset_scale = _positive(reset_scale, 'reset scale')
    matrix, reset_reason = _inverse_or_reset(inverse, len(s), reset_scale)
    report = dict(updated=False, damped=False, reset=False, reason='not_updated',
                  ys=None, sBs=None, theta=None, used_curvature=None,
                  secant_relative_residual=None, inverse=matrix)

    def reset(reason):
        return dict(report, inverse=initial_inverse(len(s), reset_scale),
                    updated=False, reset=True, reason=reason)

    if reset_reason:
        return reset(reset_reason)
    snorm, ynorm = _norm(s), _norm(y)
    if snorm <= 1e-14:
        return reset('zero_or_subresolution_displacement')
    if ynorm == 0:
        return reset('zero_gradient_change')
    try:
        Bs = np.linalg.solve(matrix, s)
    except np.linalg.LinAlgError:
        return reset('inverse_solve_failed')
    with np.errstate(over='ignore', invalid='ignore'):
        ys, sBs = float(s @ y), float(s @ Bs)
    report.update(ys=ys if np.isfinite(ys) else None,
                  sBs=sBs if np.isfinite(sBs) else None)
    if not np.isfinite(Bs).all() or not np.isfinite(ys) or not np.isfinite(sBs) or sBs <= 0:
        return reset('invalid_reference_curvature')
    theta = 1.
    if ys < damping * sBs:
        theta = (1 - damping) * sBs / (sBs - ys)
    y_used = theta * y + (1 - theta) * Bs
    curvature = float(s @ y_used)
    report.update(theta=float(theta), damped=bool(theta < 1),
                  used_curvature=curvature if np.isfinite(curvature) else None)
    curvature_floor = 32 * np.finfo(float).eps * snorm * _norm(y_used)
    if not np.isfinite(y_used).all() or not np.isfinite(curvature) or curvature <= curvature_floor:
        return reset('unusable_damped_curvature')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rho = 1. / curvature
        transform = np.eye(len(s)) - rho * np.outer(s, y_used)
        updated = transform @ matrix @ transform.T + rho * np.outer(s, s)
        updated = (updated + updated.T) * .5
    if not np.isfinite(updated).all():
        return reset('nonfinite_bfgs_update')
    try:
        diagonal = np.diag(np.linalg.cholesky(updated))
    except np.linalg.LinAlgError:
        return reset('bfgs_update_lost_positive_definiteness')
    residual = _norm(updated @ y_used - s) / snorm
    if not np.isfinite(residual) or residual > 1e-5:
        return reset('bfgs_secant_residual_unreliable')
    return dict(report, inverse=updated, updated=True, reset=False,
                reason='powell_damped_update' if theta < 1 else 'bfgs_update',
                secant_relative_residual=float(residual),
                minimum_cholesky_diagonal=float(np.min(diagonal)))


def armijo_alphas(max_trials=12, contraction=.5):
    """Bounded value-only search, starting at alpha=1 (not a gradient cap)."""
    if isinstance(max_trials, (bool, np.bool_)) or int(max_trials) != max_trials or max_trials < 1:
        raise ValueError('line search trial count must be a positive integer')
    contraction = float(contraction)
    if not np.isfinite(contraction) or not 0 < contraction < 1:
        raise ValueError('backtracking contraction must lie strictly between zero and one')
    return tuple(contraction ** index for index in range(int(max_trials)))


def armijo_decision(f_center, f_trial, slope, alpha, c1=1e-4, *, trial_valid=True):
    """Accept a finite formal scalar with sufficient decrease; no gradients.

    Negative log-QS losses are supported.  Invalid native trials and zero or
    ascending directions are rejected explicitly, never converted to a fake
    scalar.  There is no dead zone or extra objective tolerance.
    """
    f_center, slope = float(f_center), float(slope)
    alpha = _positive(alpha, 'line search alpha')
    c1 = float(c1)
    if not np.isfinite(f_center) or not np.isfinite(slope) or not np.isfinite(c1) or not 0 < c1 < 1:
        raise ValueError('finite center, slope and valid Armijo coefficient required')
    trial = None if f_trial is None else float(f_trial)
    valid = bool(trial_valid and trial is not None and np.isfinite(trial))
    bound = f_center + c1 * alpha * slope
    reduction = f_center - trial if valid else None
    accepted = bool(valid and slope < 0 and trial <= bound and reduction > 0)
    return dict(accepted=accepted, trial_valid=valid, armijo_bound=float(bound),
                actual_reduction=reduction, required_reduction=float(-c1 * alpha * slope),
                alpha=alpha, directional_derivative=slope,
                reason=('invalid_trial' if not valid else 'non_descent_direction' if slope >= 0
                        else 'accepted' if accepted else 'insufficient_armijo_decrease'))
