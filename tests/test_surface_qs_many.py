"""Independent source batching without losing query order or configuration."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.evaluator import EvaluationError, InvalidEvaluationError
from coil_eval.qs import SurfaceQSEvaluator, SurfaceQSResult, SurfaceQSConfig


def coil(marker, width=3, nfp=4, nc=4):
    coefficients=np.zeros((nc,width));coefficients[:,0]=marker
    return CoilSet(coefficients,coefficients*2,coefficients*3,np.full(nc,1e5),nfp)


@pytest.fixture
def environment(monkeypatch):
    import coil_eval._autograd.axis_batch as axis
    import coil_eval.simsopt.capture as capture
    config=EvaluationConfig(.05,fast=True,flux_method='stokes',psi_fit='two_stage')
    engine=SimpleNamespace(config=config,device=torch.device('cpu',0),library_path='native.so',
        surface_qs=SurfaceQSConfig(),share_geometry=True,batch_axis=True,_check=lambda:None)
    evaluator=SurfaceQSEvaluator.__new__(SurfaceQSEvaluator);evaluator._evaluator=engine
    events=[];preparations=[];captures=[];sampled=[]
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)

    class Prepared:
        enabled=True
        def __init__(self,physical,configs):
            self.physical=physical.copy();self.configs=list(configs)
            self.metadata={'enabled':True,'count':len(configs)}
            self.used=[[] for _ in configs]
        @contextmanager
        def capture(self,index):
            events.append(('arm',float(self.physical[index,0,0])))
            try:yield
            finally:
                self.used[index].append({'consumed':True,'cleared':True})
                events.append(('clear',float(self.physical[index,0,0])))
        def observations(self):return self.used

    def prepare(physical,configs,library,**kwargs):
        assert library=='native.so'
        result=Prepared(physical,configs);result.nfp=kwargs["nfp"];preparations.append(result)
        return result
    monkeypatch.setattr(axis,'prepare_axis_batch',prepare)

    def sources(coils,configs,**kwargs):
        assert kwargs==dict(library='native.so',device_id=0,include_torsion=False)
        captures.append((coils,list(configs)))
        marker=float(coils.coeffs_x[0,0]);events.append(('capture',marker))
        return [SimpleNamespace(valid=True,result=SimpleNamespace(status='ok'),
            source={'coil':marker,'volume':cfg.target_volume,'radius':cfg.psi_a},
            alpha={'volume':cfg.target_volume,'basis':cfg.alpha_radial_basis}) for cfg in configs]
    monkeypatch.setattr(capture,'capture_sources',sources)

    def sample(coils,cfg,source,alpha,timings,**kwargs):
        assert source['coil']==float(coils.coeffs_x[0,0])
        assert source['volume']==alpha['volume']==cfg.target_volume
        assert source['radius']==cfg.psi_a and alpha['basis']==cfg.alpha_radial_basis
        assert not torch.is_grad_enabled()
        sampled.append((coils,cfg,source,alpha))
        return SurfaceQSResult(source['coil'],cfg.target_volume,
            {'timings':timings,'source':source,'alpha':alpha,'configuration':cfg.to_dict()})
    monkeypatch.setattr(evaluator,'_evaluate_source',sample)
    return SimpleNamespace(evaluator=evaluator,engine=engine,config=config,
        preparations=preparations,captures=captures,sampled=sampled,events=events)


def test_grouped_sources_restore_original_order_and_configuration(environment):
    e=environment
    inputs=[coil(1),coil(2),coil(1),coil(3),coil(2)]
    configs=[replace(e.config,target_volume=v) for v in (.05,.06,.08,.09,.11)]
    results=e.evaluator.evaluate_many(iter(inputs),configs=iter(configs))
    assert isinstance(results,tuple)
    assert [r.f_B for r in results]==[1,2,1,3,2]
    assert [r.f_T for r in results]==[.05,.06,.08,.09,.11]
    assert len(e.preparations)==1
    np.testing.assert_array_equal(e.preparations[0].physical[:,0,0],[1,2,3])
    assert [[cfg.target_volume for cfg in cs] for _,cs in e.captures]==[[.05,.08],[.06,.11],[.09]]
    assert e.events==[(step,marker) for marker in (1.,2.,3.) for step in ('arm','capture','clear')]
    for index,result in enumerate(results):
        batch=result.diagnostics['batch']
        assert batch['query_index']==index and batch['query_count']==5
        assert batch['source_group_count']==3
        assert batch['axis_consumption']==[{'consumed':True,'cleared':True}]
        assert batch['surface_execution']=='sequential_per_query'


def test_only_target_volume_differences_share_a_source(environment):
    e=environment
    configs=[e.config,replace(e.config,psi_a=.2),replace(e.config,alpha_radial_basis='flux_legendre')]
    results=e.evaluator.evaluate_many([coil(1)]*3,configs=configs)
    assert len(e.captures)==3
    assert [r.diagnostics['source']['radius'] for r in results]==[.15,.2,.15]
    assert results[2].diagnostics['alpha']['basis']=='flux_legendre'


def test_exact_duplicates_share_and_requested_disable_is_honored(environment):
    e=environment
    results=e.evaluator.evaluate_many([coil(1),coil(1)])
    assert len(e.captures)==1 and len(e.captures[0][1])==2
    assert results[0].f_B==results[1].f_B
    e.engine.share_geometry=False;e.engine.batch_axis=False
    e.captures.clear();e.preparations.clear()
    results=e.evaluator.evaluate_many([coil(1),coil(1)])
    assert len(e.captures)==2 and not e.preparations
    assert results[0].diagnostics['batch']['axis_batch']['enabled'] is False


def test_mixed_fourier_orders_use_compatible_axis_batches(environment):
    e=environment
    results=e.evaluator.evaluate_many([coil(1,3),coil(2,5),coil(3,3)])
    assert [p.physical.shape for p in e.preparations]==[(2,4,10),(1,4,16)]
    assert [r.f_B for r in results]==[1,2,3]


@pytest.mark.parametrize('inputs,configs,error', [
    ([coil(1)],[],ValueError), ([coil(1)],[object()],ValueError),
    ([object()],None,TypeError),
    ([coil(1)],[EvaluationConfig(.05)],ValueError),
])
def test_malformed_requests_fail_before_preparation(environment,inputs,configs,error):
    e=environment
    with pytest.raises(error):e.evaluator.evaluate_many(inputs,configs=configs)
    assert not e.preparations and not e.captures


def test_empty_input_has_no_native_side_effect(environment):
    assert environment.evaluator.evaluate_many([])==()
    assert not environment.preparations


def test_invalid_source_preserves_failure_and_clears_axis_slot(environment,monkeypatch):
    import coil_eval.simsopt.capture as capture
    e=environment
    record=SimpleNamespace(valid=False,result=SimpleNamespace(status='invalid_source_capture'),
                           source=None,alpha=None)
    monkeypatch.setattr(capture,'capture_sources',lambda *a,**k:[record])
    with pytest.raises(InvalidEvaluationError,match='invalid_source_capture') as raised:
        e.evaluator.evaluate_many([coil(1)])
    assert raised.value.results==(record.result,)
    assert e.events==[('arm',1.),('clear',1.)]
    assert not e.sampled


def test_incomplete_group_is_not_silently_truncated(environment,monkeypatch):
    import coil_eval.simsopt.capture as capture
    monkeypatch.setattr(capture,'capture_sources',lambda *a,**k:[])
    with pytest.raises(EvaluationError,match='incomplete'):
        environment.evaluator.evaluate_many([coil(1)])
    assert not environment.sampled


def test_scalar_retains_ordinary_capture_without_batch_preparation(environment):
    e=environment
    result=e.evaluator.evaluate(coil(1))
    assert result.f_B==1 and result.f_T==.05
    assert len(e.captures)==1 and not e.preparations
    assert 'batch' not in result.diagnostics


def test_single_stage_batch_uses_each_query_config_without_source_sharing(environment,monkeypatch):
    from coil_eval.simsopt.state import SourcePsi
    from coil_eval.simsopt.clebsch import AlphaFit
    e=environment
    e.engine.config=replace(e.config,psi_fit='single_stage')
    calls=[]
    def capture(physical,config,*,nfp):
        assert nfp == 4
        calls.append((physical.copy(),config))
        return {'score_result':{'status':'ok'},
            'source':{'coil':float(physical[0,0]),'volume':config.target_volume,'radius':config.psi_a},
            'alpha':{'volume':config.target_volume,'basis':config.alpha_radial_basis}}
    e.engine._capture=capture
    monkeypatch.setattr(SourcePsi,'from_snapshot',lambda snapshot:snapshot['source'])
    monkeypatch.setattr(AlphaFit,'from_snapshot',lambda snapshot:snapshot['alpha'])
    second=replace(e.engine.config,target_volume=.08)
    results=e.evaluator.evaluate_many([coil(1),coil(1)],configs=[e.engine.config,second])
    assert [cfg.target_volume for _,cfg in calls]==[.05,.08]
    assert [r.f_T for r in results]==[.05,.08]
    assert not e.captures
    np.testing.assert_array_equal(calls[0][0][:,-1],np.full(4,1e5))


@pytest.mark.parametrize('corruption',['psi_modes','alpha_modes','alpha_basis'])
def test_source_basis_provenance_guards_survive_shared_helper(environment,corruption):
    from coil_eval._autograd.front import native_modes
    from coil_eval._autograd.downstream import DownstreamConfig
    from coil_eval.simsopt.state import SourcePsi
    from coil_eval.simsopt.clebsch import AlphaFit, build_modes
    e=environment
    modes=np.asarray(native_modes(2,0))
    if corruption=='psi_modes':modes=modes[::-1]
    source=SourcePsi(np.zeros(len(modes)),*modes.T,
        np.tile([1.,0.,0.,0.],(4,1)),nfp=4,a=.15,level=.3)
    cfg=DownstreamConfig.from_mapping({**e.config.to_dict(),**e.config.native_options(),'nfp':4})
    modes=np.asarray(build_modes(cfg.alpha_radial_order,cfg.alpha_poloidal_order,cfg.alpha_toroidal_order))
    if corruption=='alpha_modes':modes=modes[::-1]
    alpha=AlphaFit(modes,np.zeros(len(modes)),np.array([1.2]),4,
        radial_basis='flux_legendre' if corruption=='alpha_basis' else cfg.alpha_radial_basis)
    with pytest.raises(ValueError,match='source (psi|alpha) mode order'):
        SurfaceQSEvaluator._evaluate_source(e.evaluator,coil(1),e.config,source,alpha,{})


def test_mixed_field_periods_and_coil_counts_are_bucketed_without_sharing(environment):
    e = environment
    inputs = [coil(1, nfp=2, nc=3), coil(1, nfp=3, nc=3),
              coil(2, nfp=2, nc=1), coil(3, nfp=2, nc=3)]
    results = e.evaluator.evaluate_many(inputs)
    assert [r.f_B for r in results] == [1, 1, 2, 3]
    assert [(p.nfp, p.physical.shape) for p in e.preparations] == [
        (2, (2, 3, 10)), (3, (1, 3, 10)), (2, (1, 1, 10))]
    assert len(e.captures) == 4
    assert [(c.nfp, c.n_base_coils) for c, _ in e.captures] == [(2,3),(3,3),(2,1),(2,3)]
