"""CPU-domain/native overlap preserves caller-thread ownership and failures."""
from dataclasses import replace
import threading

import numpy as np
import pytest

from coil_eval._two_stage import native as module
from coil_eval._two_stage.capture_batch import capture_two_stage_batch, CapturePipelineError
from test_two_stage_native import factory, configurations, Function
from _oracle import make_coils


def inputs():
    coils=make_coils()
    tokens=np.concatenate([coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,
                           coils.currents_a[:,None]],axis=-1)
    # The scorer's batch entry is restricted to four physical base coils.
    tokens=np.tile(tokens,(4//len(tokens),1)) if len(tokens)!=4 else tokens
    physical=np.stack([tokens,tokens,tokens.copy()])
    physical[2,0,0]+=.001
    base=configurations()[0]
    return physical,[base,replace(base,target_volume=.01),base],[[0,1],[2]]


def track_native(library):
    threads=[];live=set();high_water=[0];counter=[0]
    for name in ('sgpu_default_score_config','sgpu_set_reverse_capture',
                 'sgpu_two_stage_begin_v1','sgpu_two_stage_fit_v1',
                 'sgpu_two_stage_resume_v1','sgpu_two_stage_close_v1',
                 'sgpu_copy_psi_warm_preconditioner'):
        original=getattr(library,name)
        def call(*args,_name=name,_original=original):
            threads.append(threading.get_ident())
            if _name=='sgpu_two_stage_close_v1':
                live.remove(args[0].value)
            result=_original(*args)
            if _name=='sgpu_two_stage_begin_v1' and args[9]._obj.value:
                counter[0]+=1;args[9]._obj.value=counter[0]
                live.add(counter[0]);high_water[0]=max(high_water[0],len(live))
            return result
        setattr(library,name,Function(call))
    return threads,live,high_water


def test_domain_runs_during_next_native_begin_and_caches_stay_on_caller(factory,monkeypatch):
    _,library,_=factory()
    threads,live,high_water=track_native(library)
    main_thread=threading.get_ident();domain_threads=[]
    second_begin=threading.Event()
    original_begin=library.sgpu_two_stage_begin_v1
    begins=[0]
    def begin(*args):
        begins[0]+=1
        if begins[0]==2:second_begin.set()
        return original_begin(*args)
    library.sgpu_two_stage_begin_v1=Function(begin)
    boundary=module.max_closed_boundary
    def domain(*args,**kwargs):
        domain_threads.append(threading.get_ident())
        assert second_begin.wait(5),'CPU domain blocked the next native S1'
        return boundary(*args,**kwargs)
    monkeypatch.setattr(module,'max_closed_boundary',domain)
    physical,configs,groups=inputs()
    results,metadata=capture_two_stage_batch(physical,configs,groups,'unused-native.so')
    assert set(threads)=={main_thread}
    assert len(set(domain_threads))==1 and main_thread not in domain_threads
    assert not live and high_water==[2] and library.flags[-1]==0
    assert [r['score'] for r,s in results]==[.03,.01,.03]
    assert [float(s['target_level']) for r,s in results]==[.03,.01,.03]
    # Both S1 factors were copied before the first S2 overwrote native scratch.
    assert results[0][1]['two_stage']['stage1_scale'][0]==1
    assert results[2][1]['two_stage']['stage1_scale'][0]==2
    assert results[0][1]['psi_scale'][0]==3
    assert results[2][1]['psi_scale'][0]==4
    assert results[0][1]['psi_scale'] is results[1][1]['psi_scale']
    assert not results[0][1]['psi_scale'].flags.writeable
    results[0][1]['target_level'][...]=-99
    assert results[1][1]['target_level']==.01
    assert metadata['query_count']==3 and metadata['group_count']==2
    assert metadata['maximum_pending_contexts']==2
    assert metadata['total_s']>0 and metadata['domain_wait_s']>=0


@pytest.mark.parametrize('failure',['begin','fit','resume','domain','snapshot'])
def test_pipeline_failure_closes_every_live_context_and_restores_flags(factory,monkeypatch,failure):
    _,library,_=factory(fail=failure if failure in {'begin','fit','resume'} else None)
    _,live,high_water=track_native(library)
    def fail(*args,**kwargs):raise RuntimeError('host failure')
    if failure=='domain':monkeypatch.setattr(module,'max_closed_boundary',fail)
    if failure=='snapshot':monkeypatch.setattr(module,'copy_reverse_snapshot',fail)
    physical,configs,groups=inputs()
    with pytest.raises(RuntimeError):
        capture_two_stage_batch(physical,configs,groups,'unused-native.so')
    assert not live and high_water[0]<=2 and library.flags[-1]==0


def test_pipeline_retains_physical_failure_status_and_input_order(factory):
    _,library,_=factory(statuses={.01:3})
    physical,configs,groups=inputs()
    results,_=capture_two_stage_batch(physical,configs,groups,'unused-native.so')
    assert results[0][0]['status']=='ok' and results[2][0]['status']=='ok'
    assert results[1][0]['status']!='ok' and results[1][1] is None
    assert library.snapshot_count==2


def test_large_group_queue_keeps_at_most_two_native_contexts(factory):
    _,library,_=factory()
    _,live,high_water=track_native(library)
    physical,configs,_=inputs()
    physical=np.repeat(physical[:1],5,axis=0)
    physical[:,0,0]+=np.arange(5)*.001
    results,metadata=capture_two_stage_batch(physical,[configs[0]]*5,
        [[i] for i in range(5)],'unused-native.so')
    assert len(results)==5 and not live and high_water==[2]
    assert [record['group_index'] for record in metadata['groups']]==list(range(5))
    assert len([name for name,value in library.calls if name=='close'])==5


@pytest.mark.parametrize('kind',['partition','geometry','config'])
def test_invalid_grouping_fails_before_native_calls(factory,kind):
    _,library,_=factory()
    physical,configs,groups=inputs()
    if kind=='partition':groups=[[0,1],[1,2]]
    if kind=='geometry':physical[1,0,0]+=.01
    if kind=='config':configs[1]=replace(configs[1],psi_a=.2)
    with pytest.raises(ValueError):
        capture_two_stage_batch(physical,configs,groups,'unused-native.so')
    assert not library.calls and not library.flags


@pytest.mark.parametrize('failure,stage,indices', [
    ('begin', 'stage1', (0, 1)), ('fit', 'stage2_fit', (0, 1)),
    ('resume', 'resume', (1,)), ('domain', 'domain', (0, 1)),
    ('snapshot', 'snapshot', (0, 1)),
])
def test_pipeline_exception_identifies_group_or_exact_volume_and_stage(
        factory, monkeypatch, failure, stage, indices):
    _, library, _ = factory(fail=failure if failure in {'begin', 'fit'} else None)
    _, live, _ = track_native(library)
    def fail(*args, **kwargs):
        raise ValueError('injected host failure')
    if failure == 'domain':
        monkeypatch.setattr(module, 'max_closed_boundary', fail)
    if failure == 'snapshot':
        monkeypatch.setattr(module.TwoStageNative, '_prepare_snapshot_arrays', fail)
    if failure == 'resume':
        original = library.sgpu_two_stage_resume_v1
        def resume(*args):
            if abs(args[4]._obj.target_volume - .01) < 1e-10:
                args[5]._obj.error_message = b'injected target-volume failure'
                return 19
            return original(*args)
        library.sgpu_two_stage_resume_v1 = Function(resume)
    physical, configs, groups = inputs()
    with pytest.raises(CapturePipelineError) as caught:
        capture_two_stage_batch(physical, configs, groups, 'unused-native.so')
    assert caught.value.query_indices == indices
    assert caught.value.stage == stage
    assert caught.value.__cause__ is not None
    assert not live and library.flags[-1] == 0
