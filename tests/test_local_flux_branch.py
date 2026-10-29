"""Local finite differences retain the center's quadrature branch."""
from pathlib import Path
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')

from coil_eval import CoilSet, EvaluationConfig
from coil_eval._two_stage.local import nearby
from coil_eval.torch import TensorEvaluator


@pytest.mark.parametrize('method',['area','stokes'])
@pytest.mark.parametrize('nfp,coil_count',[(4,4),(2,3)])
def test_one_center_choice_is_shared_by_all_endpoint_chunks(monkeypatch,method,nfp,coil_count):
    import coil_eval.torch as tensor_api
    import coil_eval._two_stage.native as native_api
    coils=CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    coils=CoilSet(coils.coeffs_x[:coil_count],coils.coeffs_y[:coil_count],
                  coils.coeffs_z[:coil_count],coils.currents_a[:coil_count],nfp)
    config=EvaluationConfig(target_volume=.05,fast=True,flux_method='stokes',
                            psi_fit='two_stage',psi_fd_batch_size=2)
    center_raw=dict(status='ok',diagnostics=dict(axis_R=1.,axis_Z=0.),components={})
    snapshot=object()
    events=[]
    choice=dict(method=method,validation='checked',fallback=method=='area',tolerance=1e-3,
                edge_relative_error=.03 if method=='area' else 1e-5,
                derivative_relative_error=.1 if method=='area' else 2e-5)
    class Driver:
        def __init__(self,*a):pass
        def evaluate(self,*a,**k):
            events.append(('capture',))
            return center_raw,snapshot
    class Evaluator:
        device=torch.device('cpu')
        def __init__(self,*a,**k):assert k['nfp']==nfp
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def _choose_local_flux(self,parameters,captured):
            assert captured is snapshot
            events.append(('select',))
            return choice,{'flux_choice':.01}
        def _evaluate(self,parameters,**kw):
            assert kw['_flux_choice'] is choice
            assert kw['_local_value'] and not kw['_parameter_grad']
            assert all(s is snapshot for s in kw['_snapshots'])
            n=len(parameters);events.append(('endpoints',n))
            names=('qs_volume','qs_outer','iota_boundary','iota_min','iota_max',
                'relative_normal_drift_p95','surface_volume','psi_fit_rms')
            metrics=SimpleNamespace(**{name:torch.ones(n)*(.05 if name=='surface_volume' else .01)
                                       for name in names})
            metadata={**choice,'validation':'center_selected','center_validation':'checked'}
            return SimpleNamespace(metrics=metrics,timings={},
                diagnostics={'flux_integration':[dict(metadata) for _ in range(n)]})
    monkeypatch.setattr(native_api,'TwoStageNative',Driver)
    monkeypatch.setattr(tensor_api,'TensorEvaluator',Evaluator)
    backend=SimpleNamespace(library='test',device_id=0,_add_geometry=lambda *a:None)
    results=nearby(backend,coils,[coils]*5,center_raw,config)
    assert events==[('capture',),('select',),('endpoints',2),('endpoints',2),('endpoints',1)]
    assert len(results)==5
    assert all(r['diagnostics']['flux_integration']['method']==method for r in results)
    assert all(r['diagnostics']['flux_integration']['validation']=='center_selected' for r in results)
    assert all(r['local_timing']['center_flux_choice']==choice for r in results)


def test_disabled_validation_selects_stokes_without_a_center_replay():
    obj=SimpleNamespace(config=EvaluationConfig(.05,stokes_validation_rtol=None))
    obj._evaluate=lambda *a,**k:pytest.fail('disabled check should skip center replay')
    choice,timings=TensorEvaluator._choose_local_flux(obj,None,None)
    assert choice['method']=='stokes' and choice['validation']=='disabled' and timings=={}


def test_enabled_selection_runs_only_the_center_front_and_flux():
    events=[]
    obj=SimpleNamespace(config=EvaluationConfig(.05))
    def replay(*args,**kwargs):
        events.append(kwargs)
        assert not torch.is_grad_enabled()
        return {'method':'area'}, {'flux_choice':.01}
    obj._evaluate=replay
    TensorEvaluator._choose_local_flux(obj,object(),object())
    assert len(events)==1 and events[0]['_flux_choice_only']
    assert not events[0]['_parameter_grad'] and not events[0]['_engineering']
    assert not events[0].get('_local_value',False)
