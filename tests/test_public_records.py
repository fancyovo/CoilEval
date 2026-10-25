"""Public input conventions, immutable records, and metric isolation."""

from dataclasses import FrozenInstanceError, fields
import json

import numpy as np
import pytest

from coil_eval import CoilSet, EvaluationConfig, EvaluationResult, Evaluator, LocalMetrics
from _oracle import FakeBackend, make_coils


def test_coil_parameter_order_units_and_json_roundtrip():
    coils = make_coils()
    vector = np.r_[coils.coeffs_x.ravel(), coils.coeffs_y.ravel(),
                   coils.coeffs_z.ravel(), coils.currents_a]
    np.testing.assert_array_equal(coils.parameter_vector(), vector)
    assert coils.parameter_count == vector.size == 20
    assert coils.n_base_coils == 2 and coils.n_coefficients == 3
    changed = vector + np.arange(vector.size) * 0.01
    np.testing.assert_array_equal(coils.with_parameters(changed).parameter_vector(), changed)
    payload = json.loads(json.dumps(coils.to_dict(), allow_nan=False))
    assert payload["length_unit"] == "m" and payload["current_unit"] == "A"
    assert payload["coefficient_order"] == "constant,sin1,cos1,..."
    np.testing.assert_array_equal(CoilSet.from_dict(payload).parameter_vector(), vector)
    for name, bad in (("length_unit", "cm"), ("current_unit", "kA"),
                      ("stellarator_symmetry", False), ("schema", "external")):
        with pytest.raises(ValueError):
            CoilSet.from_dict({**payload, name: bad})


def test_coil_records_are_copied_and_cannot_be_reenabled_for_writing():
    xyz = np.ones((1, 3))
    current = np.array([1e5])
    coils = CoilSet(xyz, xyz, xyz, current, 2)
    xyz[0, 0], current[0] = -5.0, 0.0
    assert coils.coeffs_x[0, 0] == 1.0 and coils.currents_a[0] == 1e5
    for array in (coils.coeffs_x, coils.coeffs_y, coils.coeffs_z, coils.currents_a):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    copied_vector = coils.parameter_vector()
    copied_vector[:] = 99
    assert coils.coeffs_x[0, 0] == 1.0
    with pytest.raises(FrozenInstanceError):
        coils.nfp = 4


@pytest.mark.parametrize("bad", [np.zeros(19), np.zeros((20, 1)), np.full(20, np.inf)])
def test_parameter_vector_rejects_wrong_shape_and_nonfinite_values(bad):
    with pytest.raises(ValueError):
        make_coils().with_parameters(bad)


@pytest.mark.parametrize("kwargs", [
    {"target_volume": 0}, {"target_volume": np.nan}, {"psi_a": -0.1},
    {"h": np.inf}, {"h": 0}, {"psi_iterations": 0},
    {"alpha_iterations": True}, {"segments_per_coil": 2.5},
    {"helicity": (0, 4)}, {"helicity": (1.5, 4)},
])
def test_config_rejects_unsupported_numerical_settings(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**({"target_volume": 0.03} | kwargs))


def test_only_local_physical_metrics_are_loss_visible_and_diagnostics_are_frozen():
    coils = make_coils()
    backend = FakeBackend(coils)
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=backend)
    result = evaluator.evaluate(coils)
    assert result.valid
    expected = backend.oracle.metrics(coils)
    for name in ("qs_volume", "qs_outer", "iota_boundary", "iota_min", "iota_max",
                 "relative_normal_drift_p95", "coil_length_mean", "coil_curvature_max",
                 "coil_gap_min", "coil_radius_min", "current_abs_max", "high_mode_fraction"):
        assert getattr(result.metrics, name) == pytest.approx(expected[name])
    assert result.metrics.coil_curvature_p95 == 4.3
    assert result.metrics.psi_fit_rms == .014
    metric_names = {field.name for field in fields(LocalMetrics)}
    assert not metric_names.intersection({"score", "alpha_relative_l2", "psi_train_rms",
                                          "surface_volume", "target_quality_penalty"})
    assert result.metrics.relative_normal_drift_p95 == 0.003
    assert result.diagnostics["alpha_relative_l2"] == 987.0
    assert result.diagnostics["full_physical_evaluation"] == "not_run"
    assert result.diagnostics["trace_periods"] == 1
    assert "score" not in result.to_dict() and "score_qs_residual" not in result.diagnostics
    assert "target_volume_cache_id" not in result.diagnostics
    assert not {"target_volume_verified", "surface_strict", "surface_stable"}.intersection(result.diagnostics)
    with pytest.raises(TypeError):
        result.diagnostics["alpha_relative_l2"] = 0.0
    with pytest.raises(TypeError):
        result.timings["native"]["psi"] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.metrics.qs_volume = 0.0
    backend.last_raw["diagnostics"]["alpha_relative_l2"] = -9.0
    assert result.diagnostics["alpha_relative_l2"] == 987.0


def test_result_json_roundtrip_drops_live_context_and_serializes_nonfinite_diagnostics():
    coils = make_coils()
    evaluator = Evaluator(config=EvaluationConfig(0.03), backend=FakeBackend(coils))
    original = evaluator.evaluate(coils)
    record = EvaluationResult(original.status, original.metrics,
                              {"nested": {"array": np.array([1.0, np.nan])}},
                              original.timings, original.mode, original._context)
    payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    assert payload["diagnostics"]["nested"]["array"] == [1.0, None]
    assert "_context" not in payload
    restored = EvaluationResult.from_dict(payload)
    assert restored.valid and restored.metrics == original.metrics
    assert restored._context is None
    assert restored.diagnostics["nested"]["array"] == (1.0, None)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf])
def test_required_metric_cannot_be_missing_or_nonfinite(invalid):
    coils = make_coils()
    backend = FakeBackend(coils)
    original_raw = backend.raw

    def raw(c, config):
        value = original_raw(c, config)
        value["diagnostics"]["qs_target_global_error_per_helicity"] = invalid
        return value

    backend.raw = raw
    result = Evaluator(config=EvaluationConfig(0.03), backend=backend).evaluate(coils)
    assert not result.valid
    assert result.status == "invalid_metrics" and result.metrics is None


def test_optional_metrics_can_be_absent():
    coils = make_coils()
    backend = FakeBackend(coils)
    original_raw = backend.raw

    def raw(c, config):
        value = original_raw(c, config)
        del value["diagnostics"]["coil_curvature_p95"]
        del value["diagnostics"]["coil_high_mode_energy_fraction"]
        del value["diagnostics"]["psi_train_rms"]
        return value

    backend.raw = raw
    result = Evaluator(config=EvaluationConfig(0.03), backend=backend).evaluate(coils)
    assert result.valid
    assert result.metrics.coil_curvature_p95 is None
    assert result.metrics.high_mode_fraction is None
    assert result.metrics.psi_fit_rms is None


def test_legacy_json_and_positional_metrics_accept_missing_psi_fit_rms():
    coils = make_coils()
    evaluator = Evaluator(config=EvaluationConfig(.03), backend=FakeBackend(coils))
    original = evaluator.evaluate(coils)
    legacy = original.to_dict()
    del legacy["metrics"]["psi_fit_rms"]
    restored = EvaluationResult.from_dict(json.loads(json.dumps(legacy)))
    assert restored.valid and restored.metrics.psi_fit_rms is None
    assert restored.metrics.qs_volume == original.metrics.qs_volume
    roundtrip = EvaluationResult.from_dict(json.loads(json.dumps(restored.to_dict())))
    assert roundtrip.metrics == restored.metrics
    old_names = [item.name for item in fields(LocalMetrics) if item.name != "psi_fit_rms"]
    positional = LocalMetrics(*(getattr(original.metrics, name) for name in old_names))
    assert positional.psi_fit_rms is None
    assert positional.high_mode_fraction == original.metrics.high_mode_fraction


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_optional_psi_fit_rms_rejects_present_nonfinite_values(invalid):
    coils = make_coils()
    backend = FakeBackend(coils)
    original_raw = backend.raw
    def raw(value, config):
        result = original_raw(value, config)
        result["diagnostics"]["psi_train_rms"] = invalid
        return result
    backend.raw = raw
    result = Evaluator(config=EvaluationConfig(.03), backend=backend).evaluate(coils)
    assert not result.valid and result.status == "invalid_metrics"
