"""Generic Fourier layouts and one-shot native cache ownership without CUDA."""
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval import EvaluationConfig
from coil_eval._autograd import axis_batch as m


class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)


class FakeLibrary:
    instances=[]
    fail=None
    def __init__(self,path,*,include_refinement=True):
        self.path=path;self.state=0;self.refinement_state=0;self.clears=0;self.armed=[]
        self.instances.append(self)
        self.lib=SimpleNamespace(sgpu_default_score_config=Function(self.default))
    def default(self,p):
        c=p._obj
        for k,v in dict(device_id=0,segments_per_coil=256,axis_grid=8,axis_trace_steps=960,
            axis_max_candidates=16,axis_newton_iters=6,axis_tolerance=1e-7,
            axis_fd_absolute=2e-6,axis_fd_relative=2e-4).items():setattr(c,k,v)
        return 0
    def check(self,code):
        if code:raise RuntimeError('native error')
    config=m.AxisBatchLibrary.config
    def grid(self,physical,config):
        n=int(config.axis_grid)**2
        return np.arange(n,dtype=float)+physical[0,0],np.zeros(n)
    def domain(self,physical,config):return np.array([.1,2.1,-.8,.8])
    def host_segments(self,batch,physical,*,nfp,segments):
        self.host_shape=physical.shape
        if self.fail=='segments':raise RuntimeError('segments failure')
    def refine_batch(self,batch,configs,nfp,traces,domains):
        self.domains=domains.copy();self.traces=tuple(v.copy() for v in traces)
        if self.fail=='refine':raise RuntimeError('refinement failure')
        return [m.AxisRefinementPayload(np.zeros((1,6)),np.ones(1,dtype=int)) for _ in configs]
    def set(self,entry):
        if self.state==1:raise RuntimeError('already armed')
        self.state=1;self.armed.append(entry)
    def set_refinement(self,entry):
        if self.fail=='arm_refine':raise RuntimeError('refinement arm failure')
        self.refinement_state=1
    def status(self):return self.state
    def refinement_status(self):return self.refinement_state
    def clear(self):self.state=0;self.refinement_state=0;self.clears+=1


class FakeBatch:
    instances=[]
    fail=None
    def __init__(self,path,x,y,z,current,nfp,**kwargs):
        self.blocks=tuple(a.copy() for a in (x,y,z,current));self.handle=37
        self.nfp=nfp
        self.options=kwargs;self.closed=False;self.instances.append(self)
    def trace_period(self,R,Z,*,steps):
        self.steps=steps
        if self.fail=='trace':raise RuntimeError('trace failure')
        return R+.01,Z+.02
    def close(self):self.closed=True


@pytest.fixture
def fake_native(monkeypatch):
    FakeLibrary.instances=[];FakeLibrary.fail=None;FakeBatch.instances=[];FakeBatch.fail=None
    monkeypatch.setattr(m,'AxisBatchLibrary',FakeLibrary)
    monkeypatch.setattr(m.native,'BatchCoilFieldGpu',FakeBatch)
    monkeypatch.setattr(m,'resolve_library',lambda path:Path('mock.so'))


def physical(batch=2,order=16,coils=4):
    width=3*(2*order+1)+1
    a=np.arange(batch*coils*width,dtype=float).reshape(batch,coils,width)/1000
    a[:,:,-1]=1e5
    return a


def configs(batch=2,**options):
    return [EvaluationConfig(.03+i*.01,fast=True,**options) for i in range(batch)]


@pytest.mark.parametrize('order',[4,16])
def test_generic_blocks_native_options_and_owned_entries(fake_native,order):
    a=physical(order=order);before=a.copy();cfg=configs()
    prepared=m.prepare_axis_batch(a,cfg,'unused',device=2)
    assert prepared.enabled and prepared.query_count==2
    lib=FakeLibrary.instances[0];batch=FakeBatch.instances[0];nc=2*order+1
    assert batch.closed and batch.steps==240 and lib.host_shape==a.shape
    assert [v.shape for v in batch.blocks]==[(2,4,nc)]*3+[(2,4)]
    assert prepared.metadata['pending_native_cache'] is False and lib.state==0
    for i,e in enumerate(prepared.entries):
        assert e.config.device_id==2 and (e.config.target_M,e.config.target_N)==(1,4)
        assert e.config.target_volume==cfg[i].target_volume
        assert e.config.psi_a==cfg[i].psi_a and e.config.axis_trace_steps==240
        np.testing.assert_array_equal(e.coefficients[0],before[i,:,:nc])
    a[:]=99
    np.testing.assert_array_equal(prepared.entries[0].coefficients[0],before[0,:,:nc])


def test_same_geometry_representatives_with_different_volume_configs(fake_native):
    a=physical();a[1]=a[0]
    p=m.prepare_axis_batch(a,configs(),'unused')
    lib=FakeLibrary.instances[0]
    for i in range(2):
        with p.capture(i):lib.state=lib.refinement_state=2
    assert lib.clears==2 and all(v[0]['consumed'] and v[0]['refinement_consumed'] for v in p.observations())


def test_helicity_is_explicit_and_independent_of_field_periods(fake_native):
    cfg=[EvaluationConfig(.03,fast=True,helicity=(1,0)),
         EvaluationConfig(.04,fast=True,helicity=(2,7))]
    p=m.prepare_axis_batch(physical(),cfg,'unused',nfp=4)
    assert [(e.config.target_M,e.config.target_N) for e in p.entries]==[(1,0),(2,7)]
    assert all(e.nfp==4 for e in p.entries)


@pytest.mark.parametrize('coils,nfp,helicity',[(1,1,(1,0)),(2,3,(1,0)),(5,2,(2,3))])
def test_arbitrary_basic_coils_and_field_periods_reach_native_batch(fake_native,coils,nfp,helicity):
    values=physical(coils=coils,order=2)
    prepared=m.prepare_axis_batch(values,configs(helicity=helicity),'unused',nfp=nfp)
    field=FakeBatch.instances[0]
    assert field.nfp==nfp
    assert [v.shape for v in field.blocks]==[(2,coils,5)]*3+[(2,coils)]
    assert prepared.metadata['grouping_signature'][2]==nfp
    for entry in prepared.entries:
        assert entry.nfp==nfp and entry.coefficients[0].shape==(coils,5)
        assert (entry.config.target_M,entry.config.target_N)==helicity


@pytest.mark.parametrize('nfp',[0,-1,True,np.bool_(True),1.5,'2'])
def test_invalid_nfp_rejected_before_native_allocation(fake_native,nfp):
    with pytest.raises(ValueError,match='nfp'):
        m.prepare_axis_batch(physical(coils=2),configs(),'unused',nfp=nfp)
    assert not FakeLibrary.instances and not FakeBatch.instances


@pytest.mark.parametrize('state',[1,3,4])
def test_unconsumed_cache_rejects_successful_body_and_always_clears(fake_native,state):
    p=m.prepare_axis_batch(physical(),configs(),'unused');lib=FakeLibrary.instances[0]
    with pytest.raises(RuntimeError,match='not consumed'):
        with p.capture(0):lib.state=state
    assert (lib.state,lib.refinement_state)==(0,0) and lib.clears==1


def test_failed_body_and_failed_refinement_arm_clear_only_the_owned_slot(fake_native):
    p=m.prepare_axis_batch(physical(),configs(),'unused');lib=FakeLibrary.instances[0]
    with pytest.raises(ValueError,match='body failure'):
        with p.capture(0):raise ValueError('body failure')
    assert lib.clears==1 and p.observations()[0][0]['body_failed']
    lib.fail='arm_refine'
    with pytest.raises(RuntimeError,match='arm failure'):
        with p.capture(1):pytest.fail('body must not execute')
    assert lib.clears==2 and lib.state==0
    lib.state=1
    with pytest.raises(RuntimeError,match='already armed'):
        with p.capture(0):pytest.fail('must not overwrite another slot')
    assert lib.state==1 and lib.clears==2


def test_wrong_thread_fails_before_arming(fake_native):
    p=m.prepare_axis_batch(physical(),configs(),'unused');lib=FakeLibrary.instances[0]
    p.entries[0].owner_thread=-1
    with pytest.raises(RuntimeError,match='preparing host thread'):
        with p.capture(0):pytest.fail('wrong thread')
    assert lib.state==0 and lib.clears==0


@pytest.mark.parametrize('failure',['segments','refine','trace'])
def test_preparation_failures_close_batch_without_arming(fake_native,failure):
    if failure=='trace':FakeBatch.fail=failure
    else:FakeLibrary.fail=failure
    with pytest.raises(RuntimeError):m.prepare_axis_batch(physical(),configs(),'unused')
    assert FakeBatch.instances[0].closed
    assert FakeLibrary.instances[0].state==0 and not FakeLibrary.instances[0].armed


def test_single_query_is_a_noop_before_library_resolution(fake_native,monkeypatch):
    monkeypatch.setattr(m,'resolve_library',lambda path:pytest.fail('no library resolution for B1'))
    p=m.prepare_axis_batch(physical(1),configs(1),'unused')
    assert not p.enabled and p.metadata['reason']=='single_query'
    with p.capture(0):pass
    assert not FakeBatch.instances and not FakeLibrary.instances
    with pytest.raises(IndexError):p.capture(1)


def test_missing_symbols_are_explicit_and_never_arm_or_clear(fake_native,monkeypatch):
    def unavailable(*args,**kwargs):raise m.AxisBatchUnavailable(['sgpu_axis_refinement_batch'])
    monkeypatch.setattr(m,'AxisBatchLibrary',unavailable)
    p=m.prepare_axis_batch(physical(),configs(),'unused')
    assert not p.enabled and p.metadata['reason']=='native_api_unavailable'
    assert p.metadata['missing_symbols']==['sgpu_axis_refinement_batch']
    with p.capture(1):pass
    assert not FakeBatch.instances and not FakeLibrary.instances
    with pytest.raises(m.AxisBatchUnavailable):
        m.prepare_axis_batch(physical(),configs(),'unused',allow_missing=False)


def test_symbol_discovery_precedes_binding_any_native_function(monkeypatch):
    monkeypatch.setattr(m.ctypes,'CDLL',lambda path:SimpleNamespace())
    monkeypatch.setattr(m.native,'_bind_native_score',lambda lib:pytest.fail('discovery must be first'))
    with pytest.raises(m.AxisBatchUnavailable):m.AxisBatchLibrary('old-library')


def test_coarse_only_mode_and_mixed_layout_rejection(fake_native):
    p=m.prepare_axis_batch(physical(),configs(),'unused',include_refinement=False)
    assert p.enabled and all(e.refinement is None for e in p.entries)
    mixed=[configs(1)[0],EvaluationConfig(.04,fast=False,flux_method='area')]
    with pytest.raises(ValueError,match='explicit groups'):
        m.prepare_axis_batch(physical(),mixed,'unused')


@pytest.mark.parametrize('shape',[(0,4,28),(2,0,28),(2,4,29),(2,4,4),(4,28)])
def test_bad_layouts_reject_before_native_allocation(fake_native,shape):
    with pytest.raises(ValueError,match='physical input'):
        m.prepare_axis_batch(np.zeros(shape),configs(),'unused')
    assert not FakeLibrary.instances
