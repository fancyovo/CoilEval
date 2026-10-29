"""Bounded CPU-domain/native capture pipeline for independent geometries.

Native calls, one-shot axis slots and snapshot copies stay on the creating
thread. Only the closed-domain construction and sampling run in one host worker.
Two native contexts permit that host work to overlap the next S1 or previous
S2/formal continuation without concurrent access to native thread-local state.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import os
import threading
from time import perf_counter

import numpy as np

from .._backend import _coil_arrays
from ..types import CoilSet
from .native import TwoStageNative
from .._binding import GpuError


class CapturePipelineError(GpuError):
    """A capture failure with the original query indices and pipeline stage."""
    def __init__(self, query_indices, stage, cause):
        self.query_indices = tuple(query_indices)
        self.stage = stage
        identity = (f'geometry {self.query_indices[0]}' if len(self.query_indices) == 1
                    else f'geometries {list(self.query_indices)}')
        super().__init__(f'{identity}: {stage}: {cause}')


@contextmanager
def _capture_stage(query_indices, stage):
    try:
        yield
    except CapturePipelineError:
        raise
    except Exception as exc:
        raise CapturePipelineError(query_indices, stage, exc) from exc


def _validated_groups(physical,configs,groups,prepared_axis,nfp):
    from operator import index
    if isinstance(nfp,(bool,np.bool_)):
        raise ValueError('nfp must be a positive integer')
    try:nfp=index(nfp)
    except TypeError as exc:raise ValueError('nfp must be a positive integer') from exc
    if nfp<1:raise ValueError('nfp must be a positive integer')
    physical=np.asarray(physical,dtype=np.float64)
    width=physical.shape[-1] if physical.ndim else 0
    ncoeff=(width-1)//3
    if (physical.ndim!=3 or physical.shape[1]<1 or ncoeff<3 or ncoeff%2!=1 or
            width!=3*ncoeff+1 or not len(physical) or not np.isfinite(physical).all()):
        raise ValueError('physical input must be finite [query,coil,3*ncoeff+1] with nonempty coils and odd ncoeff>=3')
    configs=list(configs)
    groups=[list(group) for group in groups]
    if len(configs)!=len(physical) or any(c.psi_fit!='two_stage' for c in configs):
        raise ValueError('provide one two-stage configuration per query')
    if not groups or any(not group for group in groups) or sorted(i for g in groups for i in g)!=list(range(len(physical))):
        raise ValueError('groups must partition every query exactly once')
    for group in groups:
        reference=configs[group[0]].to_dict();reference.pop('target_volume')
        for i in group[1:]:
            options=configs[i].to_dict();options.pop('target_volume')
            if options!=reference or not np.array_equal(physical[i],physical[group[0]]):
                raise ValueError('each group must share physical geometry and differ only in target_volume')
    if prepared_axis is not None and prepared_axis.query_count!=len(groups):
        raise ValueError('prepared axis count must match geometry groups')
    if prepared_axis is not None and any(entry.nfp!=nfp for entry in prepared_axis.entries):
        raise ValueError('prepared axis entries must share the capture nfp')
    return physical,configs,groups,ncoeff,nfp


def capture_two_stage_batch(physical, configs, groups, library, *, device_id=0,
                            prepared_axis=None,nfp=4):
    """Return ordered ``(raw_result, snapshot)`` pairs and pipeline timings.

``groups`` is the exact-geometry/volume partition used by TensorEvaluator.
All inputs share ``nfp`` and a rectangular basic-coil layout. Each group's
configurations may differ only in target volume. Physical inputs
and configuration are validated before native calls. Native work is sequential
because its capture/QR caches are thread-local; CPU domain work is independent
and overlaps it. At most two fields and one running CPU domain are retained.
The ordinary ``TwoStageNative.evaluate_many`` remains the serial reference.
    """
    physical,configs,groups,ncoeff,nfp=_validated_groups(physical,configs,groups,prepared_axis,nfp)

    started=perf_counter()
    results=[None]*len(physical)
    records=[]
    pending=deque()
    owned=[]
    capture_library=None
    metadata=dict(enabled=True,mode='cpu_domain_native_pipeline',workers=1,
        maximum_pending_contexts=2,query_count=len(physical),group_count=len(groups),
        native_execution='caller_thread',host_execution='one_cpu_domain_worker',
        serial_native_reason='thread-local capture, QR and axis-cache ownership',
        scheduling='S1[n+1] and S2/resume[n] overlap CPU domain construction',groups=records)

    def begin(group_index,executor):
        with _capture_stage(groups[group_index], 'stage1'):
            begin_group(group_index, executor)

    def begin_group(group_index,executor):
        nonlocal capture_library
        group=groups[group_index];i=group[0];row=physical[i]
        coils=CoilSet(row[:,:ncoeff],row[:,ncoeff:2*ncoeff],row[:,2*ncoeff:3*ncoeff],row[:,-1],nfp=nfp)
        arrays=_coil_arrays(coils)
        driver=TwoStageNative(library,device_id)
        owned.append(driver)
        capture_library=driver.lib
        native_configs=[driver._config(configs[k],None) for k in group]
        work=dict(group=group,driver=driver,arrays=arrays,configs=native_configs,
                  timings={},started=perf_counter(),group_index=group_index)
        driver._check(driver.lib.sgpu_set_reverse_capture(1))
        scope=prepared_axis.capture(group_index) if prepared_axis is not None else nullcontext()
        with scope:
            initial=driver._begin_prepare(coils,configs[i],native_configs[0],arrays,True,work['timings'])
        work['initial']=initial
        if 'partial_result' not in initial:
            work['future']=executor.submit(driver._prepare_domain,
                coils,configs[i],native_configs[0],initial,work['timings'])
        pending.append(work)

    def finish(work):
        driver,initial=work['driver'],work['initial']
        group,timings=work['group'],work['timings']
        formal=[];postprocess=[];pairs=[];wait_seconds=0.
        if 'partial_result' in initial:
            pairs=[(deepcopy(initial['partial_result']),None) for _ in group]
            formal=[0.]*len(group);postprocess=[0.]*len(group)
            shared_seconds=perf_counter()-work['started']
        else:
            tick=perf_counter()
            with _capture_stage(group, 'domain'):
                domain=work['future'].result()
            wait_seconds=perf_counter()-tick
            with _capture_stage(group, 'stage2_fit'):
                driver._check(driver.lib.sgpu_set_reverse_capture(1))
                prepared=driver._finish_prepare(work['configs'][0],initial,domain,True,timings)
            with _capture_stage(group, 'snapshot'):
                driver._prepare_snapshot_arrays(prepared,'reverse',False)
            shared_seconds=perf_counter()-work['started']
            for query_index,cfg in zip(group,work['configs']):
                with _capture_stage([query_index], 'resume'):
                    pair,formal_seconds,postprocess_seconds=driver._resume(
                        prepared,cfg,work['arrays'],'reverse',False)
                pairs.append(pair);formal.append(formal_seconds);postprocess.append(postprocess_seconds)
        driver.close()
        total=perf_counter()-work['started']
        group_timing=dict(query_count=len(group),completed_volumes=len(pairs),
            shared_preparation_s=shared_seconds,each_formal_s=formal,
            each_postprocess_s=postprocess,total_s=total,
            overhead_s=max(0.,total-shared_seconds-sum(formal)-sum(postprocess)),
            overlapping_group_spans=True,
            accounting='group spans overlap; use capture pipeline total_s for batch wall time')
        for index,(raw,snapshot) in enumerate(pairs):
            if 'partial_result' not in initial:
                raw['timing']['two_stage']=dict(timings,formal_downstream=formal[index],
                    total=total if len(group)==1 else shared_seconds+formal[index]+postprocess[index])
            raw['timing']['two_stage_group']=deepcopy(group_timing)
            results[group[index]]=(raw,snapshot)
        records.append(dict(group_index=work['group_index'],query_indices=group,
            cpu_domain_s=timings.get('training_boundary',0.)+timings.get('training_points',0.),
            domain_wait_s=wait_seconds,stage1_s=timings.get('stage1',0.),
            stage2_fit_s=timings.get('stage2_fit',0.),formal_s=sum(formal),elapsed_s=total))

    # Native calls never run on the executor. Future shutdown waits for host-only
    # work before input arrays can be released, including exceptional exits.
    try:
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='coil-eval-domain') as executor:
            for group_index in range(len(groups)):
                begin(group_index,executor)
                if len(pending)==2:
                    finish(pending.popleft())
            while pending:
                finish(pending.popleft())
    finally:
        for driver in owned:
            driver.close()
        if capture_library is not None:
            capture_library.sgpu_set_reverse_capture(0)
    metadata.update(total_s=perf_counter()-started,
        cpu_domain_s=sum(record['cpu_domain_s'] for record in records),
        domain_wait_s=sum(record['domain_wait_s'] for record in records))
    return results,metadata


def _fork_axis_entry(entry):
    """Own the native slot/bindings on this worker; share only readonly inputs."""
    from .._autograd.axis_batch import AxisBatchLibrary,AxisRefinementPayload
    def readonly(array):
        if array is None:return None
        value=array.view();value.setflags(write=False)
        return value
    library=AxisBatchLibrary(entry.library.path,include_refinement=entry.refinement is not None)
    refinement=(AxisRefinementPayload(readonly(entry.refinement.values),readonly(entry.refinement.winding))
                if entry.refinement is not None else None)
    return replace(entry,library=library,coefficients=tuple(readonly(a) for a in entry.coefficients),
        config=type(entry.config).from_buffer_copy(bytes(entry.config)),
        traces=tuple(readonly(a) for a in entry.traces),domain=readonly(entry.domain),
        refinement=refinement,observations=[],owner_thread=threading.get_ident())


def capture_two_stage_workers(physical,configs,groups,library,*,device_id=0,
                              prepared_axis=None,workers=2,nfp=4):
    """Experimental complete capture groups on two to four isolated host workers.

Each native context, cuBLAS/cuSOLVER handle, one-shot axis slot, QR factor and
snapshot is created/consumed/copied on one worker. Its fully owned result is
returned to the caller. Native CUDA uses its existing default stream and global
synchronizations; overlap comes from independent host preparation and queued
GPU work. No torch/autograd state is accessed by these workers.
    """
    from operator import index
    from .branches import resolve_workers
    if isinstance(workers,bool):raise ValueError('capture workers must be an integer between 1 and 4')
    try:requested=index(workers)
    except TypeError as exc:raise ValueError('capture workers must be an integer between 1 and 4') from exc
    if not 1<=requested<=4:raise ValueError('capture workers must be an integer between 1 and 4')
    if os.environ.get('SGPU_PSI_QR_SNAPSHOT'):
        raise ValueError('concurrent capture does not support the shared SGPU_PSI_QR_SNAPSHOT output file')
    if requested>1 and os.environ.get('COIL_EVAL_CAPTURE_WORKERS_TEST') != '1':
        raise ValueError('capture_workers>1 is not enabled: concurrent two-stage cuSOLVER handles require a separate native isolation implementation')
    physical,configs,groups,ncoeff,nfp=_validated_groups(physical,configs,groups,prepared_axis,nfp)
    _,allowance=resolve_workers(requested)
    effective=min(allowance,len(groups))
    started=perf_counter()
    records=[None]*len(groups)
    observations=[None]*len(groups)
    results=[None]*len(physical)

    def run(group_index):
        tick=perf_counter()
        group=groups[group_index];row=physical[group[0]]
        coils=CoilSet(row[:,:ncoeff],row[:,ncoeff:2*ncoeff],row[:,2*ncoeff:3*ncoeff],row[:,-1],nfp=nfp)
        # CDLL binding instances are private even though their native library
        # code is shared. No utility-library global cache is used on this path.
        driver=TwoStageNative(library,device_id)
        entry=None
        try:
            if prepared_axis is not None and prepared_axis.enabled:
                entry=_fork_axis_entry(prepared_axis.entries[group_index])
            with entry.capture() if entry is not None else nullcontext():
                pairs=driver.evaluate_many(coils,[configs[k] for k in group],
                    capture=True,include_torsion=False)
            for raw,snapshot in pairs:
                raw['timing']['two_stage_group'].update(overlapping_group_spans=effective>1,
                    accounting='group spans overlap; use capture workers total_s for batch wall time')
            records[group_index]=dict(group_index=group_index,query_indices=group,
                host_thread=threading.get_ident(),elapsed_s=perf_counter()-tick,
                completed_volumes=len(pairs))
            return group,pairs
        finally:
            driver.close()
            if entry is not None:observations[group_index]=entry.observations.copy()

    executor=ThreadPoolExecutor(max_workers=effective,thread_name_prefix='coil-eval-capture')
    futures=[]
    try:
        futures=[executor.submit(run,i) for i in range(len(groups))]
        for future in futures:
            group,pairs=future.result()
            if len(pairs)!=len(group):raise RuntimeError('capture worker returned an incomplete geometry group')
            for i,pair in zip(group,pairs):results[i]=pair
    finally:
        for future in futures:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
        # Mutate the caller's observations only after all worker slots are clear.
        if prepared_axis is not None and prepared_axis.enabled:
            for entry,record in zip(prepared_axis.entries,observations):
                if record is not None:entry.observations.extend(record)
    metadata=dict(enabled=True,mode='isolated_native_capture_workers',
        workers_requested=requested,workers=effective,maximum_pending_contexts=effective,
        query_count=len(physical),group_count=len(groups),
        native_execution='one complete group per private host thread and native context',
        gpu_execution='existing native default stream and device-wide synchronization',
        scheduling='independent host preparation overlaps other groups CUDA work',
        groups=records,total_s=perf_counter()-started)
    if effective<requested:
        metadata['worker_limit_reason']='available CPU allocation or independent group count'
    return results,metadata
