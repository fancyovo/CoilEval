"""Public value batches use exact axis caches without importing PyTorch."""
from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval import (CenterContextError, CoilSet, EvaluationConfig,
                       EvaluationError, Evaluator)
from coil_eval._backend import NativeBackend
from coil_eval._autograd import axis_batch
from _oracle import FakeBackend, make_coils


def four_coils(order=1,nfp=4,shift=0.):
    n=2*order+1
    x=np.zeros((4,n));x[:,0]=np.arange(4)+shift;x[:,2]=.2
    return CoilSet(x,x+.1,x+.2,np.arange(4)*1e4+1e5,nfp)


class Prepared:
    def __init__(self,physical,log,enabled=True):
        self.physical=physical;self.log=log
        self.metadata=dict(enabled=enabled,batch=len(physical),
                           reason=None if enabled else 'native_api_unavailable')
        self.entered=[]

    @contextmanager
    def capture(self,index):
        self.entered.append(index)
        self.log.append(('arm',float(self.physical[index,0,0])))
        try:yield
        finally:self.log.append(('clear',float(self.physical[index,0,0])))

    def observations(self):
        return [dict(consumed=True,index=i) for i in self.entered]


@pytest.mark.parametrize('enabled',[True,False])
def test_native_value_groups_preserve_order_and_clear_each_query(monkeypatch,enabled):
    log=[];preparations=[]
    def prepare(physical,configs,path,**kwargs):
        result=Prepared(physical,log,enabled)
        preparations.append((result,kwargs))
        return result
    monkeypatch.setattr(axis_batch,'prepare_axis_batch',prepare)
    backend=NativeBackend('missing.so')
    def evaluate(coils,config):
        log.append(('score',float(coils.coeffs_x[0,0])))
        return dict(status='ok',diagnostics={})
    backend.evaluate=evaluate
    # Compatible groups may interleave. Scoring itself keeps the input order,
    # while the singleton two-coil input retains the scalar path.
    coils=[four_coils(1,4,0),four_coils(2,3,1),four_coils(1,4,2),
           make_coils(),four_coils(2,3,4)]
    records=backend.evaluate_many(coils,EvaluationConfig(.03))
    assert [v for k,v in log if k=='score']==[0,1,2,1,4]
    assert len(preparations)==2
    assert [v['nfp'] for _,v in preparations]==[4,3]
    assert [p.physical.shape for p,_ in preparations]==[(2,4,10),(2,4,16)]
    for p,_ in preparations:assert p.entered==[0,1]
    assert records[3]['diagnostics']['independent_batch_axis']['reason']=='single_query'
    assert all(r['diagnostics']['independent_batch_axis']['enabled']==enabled
               for i,r in enumerate(records) if i!=3)
    shared=records[0]['independent_batch_timing']
    assert shared['query_count']==5 and shared['input_order_preserved']
    assert all(r['independent_batch_timing'] is shared for r in records)
    assert len(shared['query_s'])==5


def test_batch_query_error_exits_armed_context_and_preserves_error(monkeypatch):
    log=[];prepared=[]
    def prepare(physical,*args,**kwargs):
        p=Prepared(physical,log);prepared.append(p);return p
    monkeypatch.setattr(axis_batch,'prepare_axis_batch',prepare)
    backend=NativeBackend('missing.so')
    error=RuntimeError('formal query failure')
    def evaluate(coils,config):
        if coils.coeffs_x[0,0]>0:raise error
        return dict(status='ok',diagnostics={})
    backend.evaluate=evaluate
    with pytest.raises(RuntimeError) as captured:
        backend.evaluate_many([four_coils(shift=0),four_coils(shift=1)],EvaluationConfig(.03))
    assert captured.value is error
    assert log==[('arm',0.),('clear',0.),('arm',1.),('clear',1.)]


def test_empty_and_singleton_batches_do_not_prepare(monkeypatch):
    monkeypatch.setattr(axis_batch,'prepare_axis_batch',lambda *a,**k:pytest.fail('unexpected batch'))
    backend=NativeBackend('missing.so');backend.evaluate=lambda *a:dict(status='ok',diagnostics={})
    assert backend.evaluate_many([],EvaluationConfig(.03))==[]
    one=backend.evaluate_many([four_coils()],EvaluationConfig(.03))
    assert one[0]['diagnostics']['independent_batch_axis']['reason']=='single_query'


def test_mixed_nfp_and_coil_counts_prepare_separate_compatible_groups(monkeypatch):
    preparations=[];seen=[]
    def prepare(physical,configs,path,**kwargs):
        preparations.append((physical.shape,kwargs['nfp']))
        return Prepared(physical,[])
    monkeypatch.setattr(axis_batch,'prepare_axis_batch',prepare)
    backend=NativeBackend('missing.so')
    backend.evaluate=lambda coils,cfg:seen.append((coils.nfp,coils.n_base_coils,cfg.helicity)) or dict(status='ok')
    two=make_coils()
    four=four_coils(nfp=3)
    qa=EvaluationConfig(.03,helicity=(1,0))
    qh=EvaluationConfig(.03,helicity=(1,3))
    result=backend.evaluate_many([two,four,two,four],qa,[qa,qh,qa,qh])
    assert preparations==[((2,2,10),2),((2,4,10),3)]
    assert seen==[(2,2,(1,0)),(3,4,(1,3))]*2
    assert [r['diagnostics']['independent_batch_axis']['group'] for r in result]==[0,1,0,1]


def test_two_stage_nearby_routes_generic_coils_to_local_tensor_backend(monkeypatch):
    from coil_eval._two_stage import local
    observed=[]
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(version=SimpleNamespace(cuda='test')))
    def nearby(backend,coils,candidates,center_raw,config):
        observed.append((coils.nfp,coils.n_base_coils,config.helicity))
        return [dict(status='ok') for _ in candidates]
    monkeypatch.setattr(local,'nearby',nearby)
    coils=make_coils()
    cfg=EvaluationConfig(.03,psi_fit='two_stage',fast=True,flux_method='stokes',helicity=(1,0))
    result=NativeBackend('missing.so').nearby(coils,[coils],{},cfg)
    assert observed==[(2,2,(1,0))]
    assert result[0]['diagnostics']['nearby_method']=='local'
    assert result[0]['diagnostics']['nearby_backend']=='torch'


@pytest.mark.parametrize('psi_fit',['single_stage','two_stage'])
@pytest.mark.parametrize('mismatch',['nfp','coil_count'])
def test_nearby_rejects_incompatible_candidates_before_backend_loading(psi_fit,mismatch):
    coils=make_coils()
    candidate=(CoilSet(coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,coils.currents_a,3)
               if mismatch=='nfp' else four_coils(nfp=2))
    with pytest.raises(ValueError,match='match center shape and nfp'):
        NativeBackend('missing.so').nearby(coils,[candidate],{},EvaluationConfig(.03,psi_fit=psi_fit))


class BulkBackend(FakeBackend):
    def evaluate_many(self,coils,config,configs=None):
        self.calls.append(('evaluate_many',len(coils)))
        configs=[config]*len(coils) if configs is None else configs
        return [self.evaluate(c,cfg) for c,cfg in zip(coils,configs)]


def test_public_bulk_context_generations_and_empty_batch():
    coils=make_coils();backend=BulkBackend(coils)
    evaluator=Evaluator(config=EvaluationConfig(.03),backend=backend)
    old=evaluator.evaluate(coils)
    assert evaluator.evaluate_many([])==()
    assert evaluator.evaluate_nearby(old,[coils])[0].valid
    parameters=coils.parameter_vector()
    variants=[coils.with_parameters(parameters+x) for x in [0,.01,-.02]]
    results=evaluator.evaluate_many(variants)
    assert sum(c[0]=='evaluate_many' for c in backend.calls)==1
    for result,candidate in zip(results,variants):
        assert result.metrics.qs_volume==pytest.approx(backend.oracle.metrics(candidate)['qs_volume'])
        assert result.timings['python_wall_s']>=0
    for result in [old,*results[:-1]]:
        with pytest.raises(CenterContextError):evaluator.evaluate_nearby(result,[coils])
    assert evaluator.evaluate_nearby(results[-1],[coils])[0].valid


@pytest.mark.parametrize('failure',['raises','wrong_count'])
def test_failed_bulk_invalidates_previous_center(failure):
    coils=make_coils();backend=BulkBackend(coils)
    evaluator=Evaluator(config=EvaluationConfig(.03),backend=backend)
    old=evaluator.evaluate(coils)
    def broken(*args):
        if failure=='raises':raise RuntimeError('broken bulk')
        return []
    backend.evaluate_many=broken
    with pytest.raises((RuntimeError,EvaluationError)):
        evaluator.evaluate_many([coils])
    with pytest.raises(CenterContextError):evaluator.evaluate_nearby(old,[coils])


def test_plain_value_batch_has_no_torch_import_dependency():
    source=Path(__file__).resolve().parents[1]/'src'
    code=r'''
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('PyTorch imported')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import numpy as np
from coil_eval import CoilSet,EvaluationConfig
from coil_eval._backend import NativeBackend
from coil_eval._autograd import axis_batch
from contextlib import nullcontext
class Prepared:
    metadata={'enabled':False,'reason':'native_api_unavailable'}
    def capture(self,index):return nullcontext()
    def observations(self):return []
axis_batch.prepare_axis_batch=lambda *a,**k:Prepared()
backend=NativeBackend('missing.so');backend.evaluate=lambda *a:{'status':'ok','diagnostics':{}}
x=np.ones((4,3));coils=CoilSet(x,x,x,np.ones(4),4)
assert len(backend.evaluate_many([coils,coils],EvaluationConfig(.03)))==2
'''
    env={**os.environ,'PYTHONPATH':str(source)}
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_trace_layout_groups_allow_different_target_and_radius(monkeypatch):
    prepared=[];seen=[]
    def prepare(physical,configs,*args,**kwargs):
        prepared.append(configs)
        return Prepared(physical,[])
    monkeypatch.setattr(axis_batch,'prepare_axis_batch',prepare)
    backend=NativeBackend('missing.so')
    backend.evaluate=lambda c,cfg:seen.append(cfg) or dict(status='ok',diagnostics={})
    configs=[EvaluationConfig(.03,psi_a=.15,fast=True),EvaluationConfig(.03,fast=False),
             EvaluationConfig(.08,psi_a=.18,fast=True),EvaluationConfig(.08,fast=False)]
    backend.evaluate_many([four_coils(shift=i) for i in range(4)],configs[0],configs)
    assert prepared==[[configs[0],configs[2]],[configs[1],configs[3]]]
    assert seen==configs


@pytest.mark.parametrize('bulk',[True,False])
def test_per_query_configs_retained_by_last_center_and_nearby(bulk):
    coils=make_coils();backend=(BulkBackend if bulk else FakeBackend)(coils)
    default=EvaluationConfig(.01)
    configs=[EvaluationConfig(.03,psi_a=.15),EvaluationConfig(.07,psi_a=.22)]
    evaluator=Evaluator(config=default,backend=backend)
    records=evaluator.evaluate_many([coils,coils],configs=configs)
    assert [r.diagnostics['configuration']['target_volume'] for r in records]==[.03,.07]
    assert evaluator.config is default
    observed=[];original=backend.nearby
    def nearby(c,candidates,raw,cfg):
        observed.append(cfg);return original(c,candidates,raw,cfg)
    backend.nearby=nearby
    result=evaluator.evaluate_nearby(records[-1],[coils])[0]
    assert observed==[configs[-1]]
    assert result.diagnostics['configuration']['target_volume']==.07
    assert evaluator.evaluate(coils).diagnostics['configuration']['target_volume']==.01


@pytest.mark.parametrize('configs',[[],[None],[EvaluationConfig(.03)]*2])
def test_invalid_config_list_leaves_existing_center_live(configs):
    coils=make_coils();backend=FakeBackend(coils)
    evaluator=Evaluator(config=EvaluationConfig(.03),backend=backend)
    old=evaluator.evaluate(coils)
    with pytest.raises(ValueError,match='one EvaluationConfig'):
        evaluator.evaluate_many([coils],configs=configs)
    assert evaluator.evaluate_nearby(old,[coils])[0].valid
