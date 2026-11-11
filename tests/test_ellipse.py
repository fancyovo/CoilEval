import math

import numpy as np
import pytest

from coil_eval import EvaluationConfig
from coil_eval._two_stage.ellipse import EllipseDomainError, ellipse_chart


def test_ellipse_is_opt_in_and_exclusive_with_refits():
    assert EvaluationConfig(.05).psi_fit == 'two_stage'
    assert EvaluationConfig(.05, psi_fit='near_axis_ellipse').alpha_radial_basis == 'flux_legendre'
    with pytest.raises(ValueError, match='psi_target_refit requires'):
        EvaluationConfig(.05, psi_fit='near_axis_ellipse', psi_target_refit=True)


def test_ellipse_constant_linear_flow_and_unstable_axis():
    linear = np.broadcast_to([[0., -2.], [1., 0.]], (1024, 2, 2)).copy()
    radius = np.full(1024, 1.3)
    chart, slope, diagnostics = ellipse_chart(linear, radius, 1., .1, np)
    axis_ratio = np.linalg.svd(chart[0], compute_uv=False)
    assert axis_ratio[0] / axis_ratio[1] == pytest.approx(math.sqrt(2.))
    assert 2 * math.pi**2 * np.mean(radius * np.linalg.det(chart)) == pytest.approx(.1)
    assert np.max(np.abs(slope)) < 1e-10
    assert diagnostics['metric_periodic_relative_error'] < 1e-10
    with pytest.raises(EllipseDomainError, match='not_positive_definite'):
        ellipse_chart(np.broadcast_to([[0., 1.], [1., 0.]], linear.shape), radius, 1., .1, np)


def test_torch_ellipse_gradient_includes_shape_axis_and_volume():
    torch = pytest.importorskip('torch')
    count = 64
    angles = 2 * np.pi * np.arange(count) / count
    # A smooth periodic perturbation of a trace-free rotating linear field.
    fixed = np.broadcast_to([[0., -2.], [1., 0.]], (count, 2, 2))
    change = np.sin(angles)[:, None, None] * np.array([[1., .2], [.3, -1.]])
    radius = np.full(count, 1.3)
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(count, 2, 2))
    t = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(fixed.copy()) + t * torch.tensor(change)
    chart, slope, _ = ellipse_chart(a, torch.tensor(radius) + t*.05, 1., .1 + t*.01, torch)
    value = ((chart + .1*slope) * torch.tensor(weights)).sum()
    grad, = torch.autograd.grad(value, t)

    def evaluate(u):
        l, lp, _ = ellipse_chart(fixed + u*change, radius + u*.05, 1., .1 + u*.01, np)
        return ((l + .1*lp) * weights).sum()

    h = 1e-5
    expected = (evaluate(.1+h) - evaluate(.1-h))/(2*h)
    assert float(grad) == pytest.approx(expected, abs=5e-9, rel=1e-7)
