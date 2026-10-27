"""Optional QS API contracts, with native capture replaced by a CPU fixture."""
from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval import EvaluationConfig
from coil_eval.qs import SurfaceQSConfig, SurfaceQSEvaluator


def test_config_numpy_scalars_are_serializable_and_immutable():
    config = SurfaceQSConfig(nphi=np.int64(65), ntheta=np.int32(67),
        nu_order=np.uint16(12), max_iterations=np.int64(30),
        tolerance=np.float64(1e-11), min_jacobian=np.float32(1e-6),
        resonance_tolerance=np.float64(1e-10), alpha_order=np.int64(8),
        alpha_ridge=np.float32(1e-12),alpha_fit_nphi=np.int32(65),alpha_fit_ntheta=np.int64(67))
    assert json.loads(json.dumps(asdict(config))) == asdict(config)
    for name in ('nphi', 'ntheta', 'nu_order', 'max_iterations','alpha_order','alpha_fit_nphi','alpha_fit_ntheta'):
        assert type(getattr(config, name)) is int
    for name in ('tolerance', 'min_jacobian', 'resonance_tolerance','alpha_ridge'):
        assert type(getattr(config, name)) is float
    with pytest.raises(FrozenInstanceError):
        config.nphi = 129


@pytest.mark.parametrize('name', ['nphi', 'ntheta', 'nu_order', 'max_iterations'])
@pytest.mark.parametrize('value', [True, np.bool_(True), 1.0, np.float64(1),
    np.array(1), np.array([1]), '1', None, 0, -1])
def test_config_rejects_invalid_integer_fields(name, value):
    with pytest.raises(ValueError, match=name):
        SurfaceQSConfig(**{name: value})


@pytest.mark.parametrize('name', ['tolerance', 'min_jacobian', 'resonance_tolerance'])
@pytest.mark.parametrize('value', [True, np.bool_(True), np.bool_(False),
    np.array(0.01), np.array([0.01]), '0.01', None, complex(1, 0),
    float('nan'), float('inf'), -float('inf'), 0, -0.01])
def test_config_rejects_invalid_real_fields(name, value):
    with pytest.raises(ValueError, match=name):
        SurfaceQSConfig(**{name: value})


@pytest.mark.parametrize('kwargs, message', [
    ({'nphi': 4}, 'odd sizes'), ({'ntheta': 3}, 'odd sizes'),
    ({'nphi': 25, 'nu_order': 13}, 'resolve'),
    ({'min_jacobian': 1.0}, 'below one'),
])
def test_config_resolution_and_jacobian_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SurfaceQSConfig(**kwargs)


def test_smallest_resolved_configuration():
    config = SurfaceQSConfig(nphi=5, ntheta=5, nu_order=2,
        max_iterations=1, tolerance=1, resonance_tolerance=1)
    assert config.tolerance == 1.0


@pytest.mark.parametrize('options,field',[
    ({'compute_f_T':1},'compute_f_T'),({'compute_f_T':np.bool_(True)},'compute_f_T'),
    ({'alpha_order':-1},'alpha_order'),({'alpha_order':np.bool_(True)},'alpha_order'),
    ({'alpha_order':2.0},'alpha_order'),({'alpha_order':np.array(2)},'alpha_order'),
    ({'alpha_ridge':-1},'alpha_ridge'),({'alpha_ridge':np.nan},'alpha_ridge'),
    ({'alpha_ridge':True},'alpha_ridge'),({'alpha_ridge':np.array(.01)},'alpha_ridge'),
    ({'alpha_fit_nphi':48},'alpha_fit_nphi'),({'alpha_fit_ntheta':47},'alpha_fit_ntheta'),
    ({'alpha_fit_nphi':True},'alpha_fit_nphi'),({'alpha_fit_ntheta':np.bool_(True)},'alpha_fit_ntheta'),
    ({'alpha_fit_nphi':np.array(65)},'alpha_fit_nphi'),
    ({'alpha_coordinates':'unknown'},'alpha_coordinates'),({'nu_method':'unknown'},'nu_method'),
])
def test_new_surface_options_reject_invalid_inputs(options,field):
    with pytest.raises(ValueError,match=field):SurfaceQSConfig(**options)


def test_presets_and_sampler_options_keep_metric_switch_separate():
    assert SurfaceQSConfig()==SurfaceQSConfig.preset('high')
    assert not SurfaceQSConfig().compute_f_T
    for name,expected in [('high',(129,131,32,24)),('balanced',(97,99,24,16)),
                          ('fast',(65,67,8,12))]:
        config=SurfaceQSConfig.preset(name,compute_f_T=True)
        assert (config.nphi,config.ntheta,config.alpha_order,config.nu_order)==expected
        assert config.alpha_coordinates==('straight' if name=='fast' else 'geometric')
        assert config.nu_method=='potential'
        options=config.sampler_options()
        assert 'compute_f_T' not in options
        assert options=={k:v for k,v in asdict(config).items() if k!='compute_f_T'}
    assert SurfaceQSConfig(alpha_order=0,alpha_ridge=0).alpha_ridge==0
    with pytest.raises(ValueError,match='accuracy'):SurfaceQSConfig.preset('unknown')


@pytest.mark.parametrize('value',[None,True,3,[],{},np.array('high'),np.array(['high'])])
def test_preset_name_requires_a_string(value):
    with pytest.raises(ValueError,match='accuracy'):
        SurfaceQSConfig.preset(value)


@pytest.fixture
def tensor_api(monkeypatch):
    torch = pytest.importorskip('torch')
    import coil_eval.torch as api
    import coil_eval._native_loader as loader
    import coil_eval._autograd.cuda_memory as memory
    import coil_eval._autograd.boozer as boozer
    calls = []
    monkeypatch.setattr(loader, 'resolve_library', lambda path: Path('fake.so'))
    monkeypatch.setattr(api, 'NativeFieldLibrary', lambda path: object())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    monkeypatch.setattr(memory, 'prepare_native_memory', lambda device: {})

    class Field:
        def __init__(self, positions, elements, library):
            self.positions, self.elements, self.library = positions, elements, library
            self.trace = self.potential = None

    monkeypatch.setattr(api, 'SegmentField', Field)
    monkeypatch.setattr(api, 'BatchSegmentField', Field)
    monkeypatch.setattr(api, 'coil_segments', lambda x, **kw: (x, x))
    monkeypatch.setattr(api, 'coil_segments_batch', lambda x, **kw: (x, x))
    monkeypatch.setattr(api, 'DownstreamConfig', SimpleNamespace(
        from_mapping=lambda mapping: SimpleNamespace(**mapping)))
    def geometry(parameters):
        return SimpleNamespace(parameters=parameters.double(),
            axis_samples=parameters[:, :2].double())
    monkeypatch.setattr(api, 'build_front', lambda field, *a, **kw:
        SimpleNamespace(geometry=geometry(field.positions), level=torch.tensor(0.5),
            psi_train_rms=field.positions.sum()*0+0.01))
    monkeypatch.setattr(api, 'build_batch_front', lambda field, *a, **kw:
        SimpleNamespace(geometry=lambda i: geometry(field.positions[i]),
            levels=field.positions.new_full((len(field.positions),), 0.5),
            psi_train_rms=field.positions.sum((-2, -1))*0+0.01))

    def physics(geometry):
        if hasattr(geometry, 'parameters'):
            geometry = geometry.parameters
        value = geometry.sum(dim=(-2, -1))
        result = {name: value for name in ('raw_qs', 'iota_boundary',
            'target_quality_error', 'surface_volume', 'raw_qs_outer',
            'iota_min', 'iota_max')}
        result['flux_validation'] = {'method': 'stokes'}
        return result

    monkeypatch.setattr(api, 'downstream', lambda geometry, *a: physics(geometry))
    monkeypatch.setattr(api, 'batch_downstream', lambda front, cache, field, cfg:
        physics(field.positions))
    monkeypatch.setattr(api, '_EngineeringMetrics', SimpleNamespace(
        apply=lambda x, *args: x.sum()*torch.tensor([1., 2., 3.])))

    def sample(geometry, level, phys, field, cfg, **settings):
        calls.append(settings)
        geometry = geometry.parameters
        phi = torch.arange(settings['nphi'], dtype=geometry.dtype)[:, None]
        phi = phi*(2*np.pi/settings['nphi'])
        theta = torch.arange(settings['ntheta'], dtype=geometry.dtype)[None, :]
        theta = theta*(2*np.pi/settings['ntheta'])
        amplitude = geometry.sum()*0.01
        B = 2 + amplitude*torch.cos(theta) + .1*torch.sin(phi+2*theta)
        result = {'B_magnitude': B, 'iota': geometry.sum()*0+1.3,
            'G': geometry.sum()*0+2.5, 'major_radius': geometry.sum()*0+1.,
            'diagnostics': {'jacobian_min': torch.tensor(.9)}}
        if settings['field_derivatives']:
            zero=torch.zeros_like(B);one=torch.ones_like(B)
            bt=-amplitude*torch.sin(theta)+.2*torch.cos(phi+2*theta)
            bp=.4*torch.cos(phi+2*theta)
            first=torch.stack((B/2.5*(bp+1.3*bt),bt,zero),-1)
            result.update(B=torch.stack((B,zero,zero),-1),
                dBdx=torch.stack((first,torch.zeros_like(first),torch.zeros_like(first)),-2),
                x_theta=torch.stack((zero,one,zero),-1),
                x_phi=torch.stack((2.5/B,-1.3*one,zero),-1))
        return result

    monkeypatch.setattr(boozer, 'sample_boozer_surface', sample)

    def evaluator(surface_qs=None):
        config = EvaluationConfig(target_volume=.05, fast=True, flux_method='stokes',
            stokes_validation_rtol=None)
        engine = api.TensorEvaluator(config=config, surface_qs=surface_qs,
            batch_axis=False)
        engine.device = torch.device('cpu')
        engine._capture = lambda *args: {'config': config.native_options(),
            'score_result': {'status': 'ok'}, 'alpha_upper': [1.],
            'alpha_scale': [1.], 'alpha_center_scaled_solution': [1.]}
        engine._center = lambda snapshot: {}
        return engine

    return torch, api, evaluator, calls


def test_legacy_result_positional_construction(tensor_api):
    torch, api, _, _ = tensor_api
    result = api.TensorEvaluation(torch.ones(1), None, {}, {})
    assert result.surface_qs is None


@pytest.mark.parametrize('compute_ft',[False,True])
def test_optional_metrics_preserve_existing_single_results(tensor_api,compute_ft):
    torch, _, evaluator, calls = tensor_api
    x = torch.ones((4, 10), dtype=torch.float64, requires_grad=True)
    with evaluator() as engine:
        baseline = engine.evaluate(x)
    assert baseline.surface_qs is None
    assert calls == []
    config = SurfaceQSConfig(nphi=9, ntheta=11, nu_order=3,compute_f_T=compute_ft)
    with evaluator(config) as engine:
        result = engine.evaluate(x)
    for name in vars(baseline.metrics):
        torch.testing.assert_close(getattr(result.metrics, name),
                                   getattr(baseline.metrics, name))
    assert calls == [{**config.sampler_options(),'field_derivatives':compute_ft}]
    assert result.surface_qs.f_B.ndim == 0
    assert (result.surface_qs.f_T.ndim==0) if compute_ft else result.surface_qs.f_T is None
    assert result.surface_qs.diagnostics['region'] == 'target_surface'
    assert result.surface_qs.diagnostics['surfaces'][0]['jacobian_min'] > 0
    assert 'boozer_surface_qs' in result.timings
    loss=result.surface_qs.f_B.square()
    if compute_ft:loss=loss+result.surface_qs.f_T.square()
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) == x.numel()


@pytest.mark.parametrize('compute_ft',[False,True])
def test_batch_optional_metrics_and_gradients_are_independent(tensor_api,compute_ft):
    torch, _, evaluator, calls = tensor_api
    config = SurfaceQSConfig(nphi=9, ntheta=11, nu_order=3,compute_f_T=compute_ft)
    x = torch.stack([torch.ones((4, 10)), torch.full((4, 10), 2.)])
    x = x.double().requires_grad_()
    with evaluator(config) as engine:
        result = engine.evaluate_batch(x)
        single = engine.evaluate(x[0])
    assert len(calls) == 3
    for name in (('f_B', 'f_T') if compute_ft else ('f_B',)):
        values = getattr(result.surface_qs, name)
        assert values.shape == (2,)
        torch.testing.assert_close(values[0], getattr(single.surface_qs, name))
        grad, = torch.autograd.grad(values[0], x, retain_graph=True)
        assert torch.count_nonzero(grad[0]) > 0
        torch.testing.assert_close(grad[1], torch.zeros_like(grad[1]))
    assert len(result.surface_qs.diagnostics['surfaces']) == 2


@pytest.mark.parametrize('value', [False, True, {}, 1])
def test_tensor_constructor_rejects_invalid_optional_config(tensor_api, value):
    _, _, evaluator, _ = tensor_api
    with pytest.raises(TypeError, match='SurfaceQSConfig'):
        evaluator(value)


def test_value_wrapper_enables_explicit_or_default_config(monkeypatch):
    torch = pytest.importorskip('torch')
    import coil_eval.torch as api
    captured = []
    closed = []

    class FakeTensorEvaluator:
        def __init__(self, *args, **kwargs):
            captured.append((args, kwargs))
        def close(self):
            closed.append(True)

    monkeypatch.setattr(api, 'TensorEvaluator', FakeTensorEvaluator)
    config = EvaluationConfig(target_volume=.05, fast=True, flux_method='stokes')
    explicit = SurfaceQSConfig(nphi=129, ntheta=131)
    with SurfaceQSEvaluator('native.so', 'field.so', config=config,
                            surface_qs=explicit, device_id=2):
        pass
    with SurfaceQSEvaluator(config=config):
        pass
    assert captured[0][0] == ('native.so', 'field.so')
    assert captured[0][1]['surface_qs'] is explicit
    assert captured[0][1]['field_dtype'] is torch.float64
    assert captured[0][1]['device_id'] == 2
    assert captured[1][1]['surface_qs'] == SurfaceQSConfig()
    assert len(closed) == 2
