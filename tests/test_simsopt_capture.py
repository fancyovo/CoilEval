"""Source capture owns data and shares the public evaluator's result decoding."""
from dataclasses import replace

import numpy as np
import pytest

from coil_eval import CoilSet, EvaluationConfig, EvaluationError
from coil_eval._binding import GpuError
from coil_eval.evaluator import _METRIC_FIELDS
from coil_eval.simsopt import capture_sources, GpuCoilField


def _raw(status="ok"):
    return {"status": status, "diagnostics": {key: .1 for key in _METRIC_FIELDS.values()}, "timing": {"total": 1.}}


def _snapshot():
    return {"psi_coefficients": np.array([.2]), "psi_mode_a": np.array([2]),
        "psi_mode_b": np.array([0]), "psi_mode_m": np.array([0]), "psi_mode_kind": np.array([0]),
        "axis_samples": np.tile([1., 0., 0., 0.], (8, 1)), "nfp": np.array(4),
        "config": {"psi_a": .15}, "target_level": np.array(.3),
        "alpha_mode_l": np.array([1]), "alpha_mode_m": np.array([1]),
        "alpha_mode_n": np.array([0]), "alpha_mode_kind": np.array([0]),
        "alpha_coefficients": np.array([.1]), "alpha_iota_coefficients": np.array([1.2])}


@pytest.fixture
def driver(monkeypatch):
    import coil_eval._native_loader
    import coil_eval._two_stage.native
    class Driver:
        instances = []
        rows = [(_raw(), _snapshot())]
        error = None
        def __init__(self, library, device_id):
            self.closed = 0
            self.calls = []
            self.instances.append(self)
        def evaluate_many(self, coils, configs, capture, include_torsion=True):
            self.calls.append((coils, configs, capture))
            self.include_torsion = include_torsion
            if self.error:
                raise self.error
            return self.rows
        def close(self):
            self.closed += 1
    monkeypatch.setattr(coil_eval._native_loader, "resolve_library", lambda library: library)
    monkeypatch.setattr(coil_eval._two_stage.native, "TwoStageNative", Driver)
    return Driver


@pytest.fixture
def coils():
    coordinate = np.zeros((4, 33))
    return CoilSet(coordinate, coordinate, coordinate, np.full(4, 1e5), 4)


def test_single_config_returns_owned_sources_and_closes(driver, coils):
    config = EvaluationConfig(.05, psi_fit="two_stage")
    records = capture_sources(coils, config, library="chosen-library")
    assert len(records) == 1 and records[0].valid
    assert records[0].result.diagnostics["configuration"]["psi_fit"] == "two_stage"
    assert driver.instances[0].closed == 1
    assert driver.instances[0].calls[0][1:] == ([config], "sources")
    driver.rows[0][1]["psi_coefficients"][:] = 99.
    driver.rows[0][1]["alpha_coefficients"][:] = 88.
    assert records[0].source.coefficients[0] == .2
    assert records[0].alpha.lambda_coefficients[0] == .1
    assert not records[0].source.coefficients.flags.writeable


def test_many_preserves_order_and_invalid_result(driver, coils):
    first = EvaluationConfig(.05, psi_fit="two_stage")
    second = replace(first, target_volume=.08)
    driver.rows = [(_raw("no_surface"), None), (_raw(), _snapshot())]
    records = capture_sources(coils, [first, second], library="chosen-library")
    assert len(driver.instances[0].calls) == 1
    assert not records[0].valid and records[0].source is None and records[0].alpha is None
    assert records[0].result.status == "no_surface"
    assert records[1].valid
    assert records[1].result.diagnostics["configuration"]["target_volume"] == .08


def test_source_conversion_failure_is_recorded(driver, coils):
    snapshot = _snapshot()
    del snapshot["alpha_coefficients"]
    driver.rows = [(_raw(), snapshot)]
    record, = capture_sources(coils, EvaluationConfig(.05, psi_fit="two_stage"), library="chosen-library")
    assert not record.valid and record.source is None and record.alpha is None
    assert record.result.status == "invalid_source_capture"
    assert "source_capture_error" in record.result.diagnostics
    assert driver.instances[0].closed == 1


def test_missing_owned_snapshot_has_an_explicit_failure_status(driver, coils):
    driver.rows = [(_raw(), None)]
    record, = capture_sources(coils, EvaluationConfig(.05, psi_fit="two_stage"), library="chosen-library")
    assert not record.valid and not record.result.valid
    assert record.result.status == "invalid_source_capture"
    assert "no owned source snapshot" in record.result.diagnostics["source_capture_error"]
    assert driver.instances[0].closed == 1


def test_native_failure_releases_driver(driver, coils):
    driver.error = GpuError("synthetic native failure")
    with pytest.raises(EvaluationError, match="native source capture failed"):
        capture_sources(coils, EvaluationConfig(.05, psi_fit="two_stage"), library="chosen-library")
    assert driver.instances[0].closed == 1


def test_source_capture_can_skip_optional_torsion_without_changing_records(driver, coils):
    record, = capture_sources(coils, EvaluationConfig(.05, psi_fit="two_stage"),
                             library="chosen-library", include_torsion=False)
    assert record.valid and driver.instances[0].include_torsion is False


def test_validation_precedes_native_construction(driver, coils):
    config = EvaluationConfig(.05, psi_fit="two_stage")
    with pytest.raises(TypeError):
        capture_sources(object(), config)
    with pytest.raises(TypeError):
        capture_sources(coils, [object()])
    with pytest.raises(ValueError, match="two_stage"):
        capture_sources(coils, EvaluationConfig(.05))
    with pytest.raises(ValueError, match="target_volume"):
        capture_sources(coils, [config, replace(config, psi_a=.2)])
    assert capture_sources(coils, []) == []
    assert not driver.instances


def test_gpu_field_context_callback_and_idempotent_close(monkeypatch, coils):
    import coil_eval._binding
    import coil_eval._native_loader
    created = []
    class Field:
        def __init__(self, library, *arrays, **options):
            self.library, self.arrays, self.options, self.closed = library, arrays, options, 0
            created.append(self)
        def eval_B(self, xyz, precision):
            self.precision = precision
            return np.ones_like(xyz)
        def close(self):
            self.closed += 1
    monkeypatch.setattr(coil_eval._binding, "CoilFieldGpu", Field)
    monkeypatch.setattr(coil_eval._native_loader, "resolve_library", lambda path: "resolved-library")
    with GpuCoilField(coils, precision="fp64") as field:
        np.testing.assert_array_equal(field(np.zeros((3, 3))), np.ones((3, 3)))
        assert created[0].library == "resolved-library" and created[0].precision == "fp64"
        assert created[0].options == {"segments_per_coil": 256, "device_id": 0}
    field.close()
    assert created[0].closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        field(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="precision"):
        GpuCoilField(coils, precision="half")
