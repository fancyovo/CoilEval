"""Private native capture/axis ownership for experimental host worker overlap."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from coil_eval import _binding
from coil_eval._autograd import axis_batch
from coil_eval._two_stage import capture_batch as batch
from coil_eval._two_stage import native as module
from test_capture_pipeline import inputs,track_native
from test_two_stage_native import factory,FakeLibrary,Function


@pytest.fixture(autouse=True)
def solver_stream_for_worker_unit_tests(monkeypatch):
    # Fake-library ownership tests exercise the worker scheduler. The public
    # native path stays disabled until its separate cuSOLVER handle isolation
    # is implemented and validated.
    monkeypatch.setenv('COIL_EVAL_CAPTURE_WORKERS_TEST', '1')


def private_libraries(monkeypatch,*,barrier=None,fail=None):
    libraries=[];records=[]
    def create(path):
        owner=threading.get_ident()
        lib=FakeLibrary(fail=fail if not libraries else None)
        threads,live,water=track_native(lib)
        if barrier is not None:
            original=lib.sgpu_two_stage_begin_v1
            def begin(*args):
                barrier.wait(5)
                return original(*args)
            lib.sgpu_two_stage_begin_v1=Function(begin)
        libraries.append(lib);records.append((owner,threads,live,water))
        return lib
    monkeypatch.setattr(module.ct,'CDLL',create)
    return libraries,records


def test_complete_native_captures_overlap_on_private_threads_and_copy_correct_factors(factory,monkeypatch):
    factory()
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    libraries,records=private_libraries(monkeypatch,barrier=threading.Barrier(2))
    physical,configs,groups=inputs()
    results,metadata=batch.capture_two_stage_workers(physical,configs,groups,
        'unused-native.so',workers=2)
    assert len(records)==2 and len({r[0] for r in records})==2
    assert threading.get_ident() not in {r[0] for r in records}
    for owner,threads,live,water in records:
        assert set(threads)=={owner} and not live and water==[1]
    assert all(lib.flags==[1,0] for lib in libraries)
    assert [r['score'] for r,s in results]==[.03,.01,.03]
    assert [float(s['target_level']) for r,s in results]==[.03,.01,.03]
    for raw,snapshot in results:
        assert snapshot['two_stage']['stage1_scale'][0]==1
        assert snapshot['psi_scale'][0]==2
        assert raw['timing']['two_stage_group']['overlapping_group_spans']
    assert results[0][1]['psi_scale'] is results[1][1]['psi_scale']
    assert results[0][1]['psi_scale'] is not results[2][1]['psi_scale']
    assert metadata['workers']==2 and metadata['maximum_pending_contexts']==2


@pytest.mark.parametrize('failure',['begin','fit','resume','domain','snapshot'])
def test_workers_wait_for_cleanup_after_native_or_host_failure(factory,monkeypatch,failure):
    factory()
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    libraries,records=private_libraries(monkeypatch,fail=failure)
    def fail(*args,**kwargs):raise RuntimeError('host failure')
    if failure=='domain':monkeypatch.setattr(module,'max_closed_boundary',fail)
    if failure=='snapshot':monkeypatch.setattr(module,'copy_reverse_snapshot',fail)
    physical,configs,groups=inputs()
    with pytest.raises(RuntimeError):
        batch.capture_two_stage_workers(physical,configs,groups,'unused-native.so',workers=2)
    assert libraries
    for lib,(owner,threads,live,water) in zip(libraries,records):
        assert set(threads)=={owner} and not live and lib.flags[-1]==0


def test_worker_axis_copy_keeps_original_owner_and_readonly_arrays(monkeypatch):
    class Library:
        def __init__(self,path,**kwargs):self.path=path
    monkeypatch.setattr(axis_batch,'AxisBatchLibrary',Library)
    values=tuple(np.arange(4,dtype=float) for _ in range(4))
    payload=axis_batch.AxisRefinementPayload(np.ones((2,6)),np.ones(2,dtype=np.int32))
    original=axis_batch.AxisCoarseEntry(Library('unused'),values,
        _binding._SgpuScoreConfig(),values,refinement=payload,domain=np.arange(4.))
    with ThreadPoolExecutor(max_workers=1) as executor:
        cloned=executor.submit(batch._fork_axis_entry,original).result()
    assert original.owner_thread==threading.get_ident()!=cloned.owner_thread
    assert cloned.library is not original.library and cloned.config is not original.config
    for a,b in zip(original.coefficients,cloned.coefficients):
        assert np.shares_memory(a,b) and not b.flags.writeable
    assert not cloned.refinement.values.flags.writeable
    cloned.observations.append({'consumed':True})
    assert not original.observations


def test_worker_count_respects_cpu_allowance(factory,monkeypatch):
    factory();monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    libraries,records=private_libraries(monkeypatch)
    physical,configs,groups=inputs()
    results,metadata=batch.capture_two_stage_workers(physical,configs,groups,'unused-native.so',workers=4)
    assert len(results)==3 and metadata['workers']==1
    assert metadata['worker_limit_reason']
    assert len({r[0] for r in records})==1


@pytest.mark.parametrize('workers',[True,0,5,1.5,'2'])
def test_invalid_worker_count_fails_before_native_calls(factory,workers):
    _,lib,_=factory();physical,configs,groups=inputs()
    with pytest.raises(ValueError,match='capture workers'):
        batch.capture_two_stage_workers(physical,configs,groups,'unused-native.so',workers=workers)
    assert not lib.calls


def test_shared_native_snapshot_writer_is_rejected(factory,monkeypatch):
    _,lib,_=factory();monkeypatch.setenv('SGPU_PSI_QR_SNAPSHOT','diagnostic.bin')
    physical,configs,groups=inputs()
    with pytest.raises(ValueError,match='SGPU_PSI_QR_SNAPSHOT'):
        batch.capture_two_stage_workers(physical,configs,groups,'unused-native.so')
    assert not lib.calls
