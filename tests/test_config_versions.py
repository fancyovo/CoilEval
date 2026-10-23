import pytest
from coil_eval import EvaluationConfig

def test_fast_is_explicit_and_standard_is_unchanged():
    default = EvaluationConfig(target_volume=0.05)
    assert default.fast is False
    assert default.native_options()['surface_trace_steps'] == 400
    assert 'axis_trace_steps' not in default.native_options()
    fast = EvaluationConfig(target_volume=0.05, fast=True)
    assert fast.native_options()['axis_trace_steps'] == 240
    assert fast.native_options()['axis_sample_count'] == 240
    assert fast.native_options()['surface_trace_steps'] == 100
    assert fast.to_dict()['fast'] is True
    with pytest.raises(TypeError):
        EvaluationConfig(target_volume=0.05, fast='yes')

def test_flux_selection_is_explicit():
    assert EvaluationConfig(target_volume=0.05).flux_method == 'area'
    assert EvaluationConfig(target_volume=0.05, fast=True, flux_method='stokes').flux_method == 'stokes'
    with pytest.raises(ValueError):
        EvaluationConfig(target_volume=0.05, flux_method='stokes')
    with pytest.raises(ValueError):
        EvaluationConfig(target_volume=0.05, flux_method='auto')
