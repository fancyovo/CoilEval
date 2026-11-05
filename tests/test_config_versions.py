import pytest
from coil_eval import EvaluationConfig

def test_explicit_standard_trace_remains_available():
    default = EvaluationConfig(target_volume=0.05, fast=False, flux_method='area')
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

def test_flux_selection_and_fast_requirement():
    assert EvaluationConfig(target_volume=0.05).flux_method == 'stokes'
    assert EvaluationConfig(target_volume=0.05, fast=True, flux_method='stokes').flux_method == 'stokes'
    with pytest.raises(ValueError):
        EvaluationConfig(target_volume=0.05, fast=False, flux_method='stokes')
    with pytest.raises(ValueError):
        EvaluationConfig(target_volume=0.05, flux_method='auto')


def test_refit_ridge_is_explicit_and_does_not_change_stage1():
    default=EvaluationConfig(target_volume=.05,psi_fit='two_stage')
    assert default.psi_refit_ridge is None
    config=EvaluationConfig(target_volume=.05,psi_fit='two_stage',psi_refit_ridge=1e-10)
    assert config.to_dict()['psi_refit_ridge']==1e-10
    assert 'psi_ridge' not in config.native_options()
    assert 'psi_refit_ridge' not in config.native_options()
    assert EvaluationConfig(**config.to_dict())==config
    for value in (-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError,match='psi_refit_ridge'):
            EvaluationConfig(target_volume=.05,psi_fit='two_stage',psi_refit_ridge=value)
    with pytest.raises(ValueError,match='two_stage'):
        EvaluationConfig(target_volume=.05,psi_fit='single_stage',psi_refit_ridge=1e-10)


def test_public_defaults_match_validated_s2_recipe():
    default=EvaluationConfig(.05)
    frozen=EvaluationConfig(.05,fast=True,flux_method='stokes',psi_fit='two_stage',
        psi_refit_points=82368,psi_refit_sampling='polar_quadrature',
        alpha_radial_weighting='smooth',alpha_radial_basis='flux_legendre',
        alpha_weight_sigma=.05,alpha_count_floor=32.,stokes_validation_rtol=.001,
        psi_refit_ridge=None,psi_target_refit=False)
    assert default == frozen
    assert default.native_options()['surface_trace_steps'] == 100
    assert not default.psi_target_refit
    assert EvaluationConfig(**default.to_dict()) == default
