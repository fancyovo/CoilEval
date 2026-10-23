"""Analytic nonlinear-loss gradients in physical coil units, with full bases."""

import numpy as np
import pytest

from coil_eval import EvaluationConfig, Evaluator, InvalidEvaluationError
from _oracle import FakeBackend, make_coils


@pytest.mark.parametrize("rotated", [False, True])
def test_complete_centered_gradient_matches_independent_analytic_oracle(rotated):
    coils = make_coils()
    backend = FakeBackend(coils)
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    n = coils.parameter_count
    basis = (np.linalg.qr(np.random.default_rng(812).normal(size=(n, n)))[0]
             if rotated else np.eye(n))
    scales = backend.oracle.scales
    result = evaluator.loss_and_grad(coils, backend.oracle.loss,
                                     scales=scales, h=1e-4, basis=basis)
    np.testing.assert_allclose(result.gradient, backend.oracle.physical_gradient(coils),
                               rtol=2e-7, atol=2e-11)
    assert result.value == pytest.approx(backend.oracle.loss(result.center.metrics))
    assert result.gradient.shape == (n,) and len(result.endpoints) == 2*n
    assert result.timings["endpoint_count"] == 2*n
    assert [call[0] for call in backend.calls] == ["evaluate", "nearby"]
    recorded = np.array(backend.calls[-1][1])
    expected = np.array([coils.parameter_vector()+sign*1e-4*scales*direction
                         for direction in basis for sign in (1, -1)])
    np.testing.assert_array_equal(recorded, expected)
    assert all(endpoint.mode == "nearby" for endpoint in result.endpoints)
    assert all(endpoint.diagnostics["geometry_model"] == "center_linearized"
               for endpoint in result.endpoints)
    for array in (result.gradient, result.scales, result.basis):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_missing_physical_scales_are_rejected_before_native_evaluation():
    coils = make_coils()
    backend = FakeBackend(coils)
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    with pytest.raises(ValueError, match="explicit physical scales"):
        evaluator.loss_and_grad(coils, backend.oracle.loss)
    assert backend.calls == []


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_explicit_scalar_scale_broadcast_matches_vector_scale_and_analytic_gradient(scale):
    coils = make_coils()
    backend = FakeBackend(coils)
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    scalar = evaluator.loss_and_grad(coils, backend.oracle.loss, scales=scale)
    vector = evaluator.loss_and_grad(coils, backend.oracle.loss,
                                     scales=np.full(coils.parameter_count, scale))
    expected = backend.oracle.physical_gradient(coils)
    np.testing.assert_allclose(scalar.gradient, expected, rtol=2e-5, atol=2e-11)
    np.testing.assert_array_equal(scalar.gradient, vector.gradient)
    np.testing.assert_array_equal(scalar.scales, np.full(coils.parameter_count, scale))


@pytest.mark.parametrize("mode", ["invalid_status", "nonfinite_metric"])
def test_invalid_endpoint_rejects_entire_gradient_and_preserves_endpoint_evidence(mode):
    coils = make_coils()
    backend = FakeBackend(coils)
    if mode == "invalid_status":
        backend.reject_nearby_index = 7
    else:
        backend.nonfinite_nearby_index = 7
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    with pytest.raises(InvalidEvaluationError) as captured:
        evaluator.loss_and_grad(coils, backend.oracle.loss, scales=backend.oracle.scales)
    endpoints = captured.value.results
    assert len(endpoints) == 2*coils.parameter_count
    assert not endpoints[7].valid and endpoints[7].metrics is None
    assert sum(endpoint.valid for endpoint in endpoints) == len(endpoints)-1
    assert "7" in str(captured.value)


def test_invalid_center_stops_before_building_any_gradient_endpoints():
    coils = make_coils()
    backend = FakeBackend(coils)
    backend.invalidate_center = True
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    with pytest.raises(InvalidEvaluationError) as captured:
        evaluator.loss_and_grad(coils, backend.oracle.loss, scales=backend.oracle.scales)
    assert [call[0] for call in backend.calls] == ["evaluate"]
    assert len(captured.value.results) == 1
    assert captured.value.results[0].status == "no_surface"


@pytest.mark.parametrize("kwargs", [
    {"h": 0}, {"h": np.nan}, {"scales": 0}, {"scales": -1},
    {"scales": np.full(20, np.inf)}, {"scales": np.ones(19)},
    {"basis": np.eye(20)[:2]}, {"basis": np.ones((20, 20))},
])
def test_bad_gradient_coordinates_are_rejected_before_native_evaluation(kwargs):
    coils = make_coils()
    backend = FakeBackend(coils)
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    with pytest.raises(ValueError):
        evaluator.loss_and_grad(coils, backend.oracle.loss, **({"scales": 1.0} | kwargs))
    assert backend.calls == []


@pytest.mark.parametrize("loss", [lambda m: [m.qs_volume], lambda m: np.nan,
                                  lambda m: np.inf, lambda m: 1.0+0.0j,
                                  lambda m: "1.0", lambda m: True])
def test_custom_loss_must_return_one_finite_scalar(loss):
    coils = make_coils()
    backend = FakeBackend(coils)
    with pytest.raises(ValueError):
        Evaluator(config=EvaluationConfig(0.03), backend=backend).loss_and_grad(
            coils, loss, scales=backend.oracle.scales)
    assert [call[0] for call in backend.calls] == ["evaluate"]
