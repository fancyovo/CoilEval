"""Opt-in batching of native coarse traces with unchanged axis selection.

Native CPU code generates the exact raster and the same segment arrays used by
the ordinary scorer. Only coarse trace endpoints are cached; native winding,
candidate refinement, FP64/topology review and final selection remain in place.
The cache is consumed once, keyed to its physical geometry and trace settings.
Optional native candidate refinement batching retains the original finite-state
Newton algorithm; it requires exact seed/winding/configuration cache matching.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import time
import threading
from pathlib import Path
from operator import index as integer_index

import numpy as np
from .. import _binding as native
from ..config import EvaluationConfig
from .._native_loader import resolve_library


STATUS_NAMES = {0:'idle', 1:'armed', 2:'consumed', 3:'key_mismatch', 4:'unused'}


def _pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _coefficients(physical):
    width=physical.shape[-1]
    count=(width-1)//3
    if count<3 or width!=3*count+1 or count%2!=1:
        raise ValueError('physical tokens require three odd Fourier blocks and a current')
    return tuple(np.ascontiguousarray(physical[..., part], dtype=np.float64)
        for part in (slice(0,count),slice(count,2*count),slice(2*count,3*count),-1))


def _trace_signature(config, nfp):
    return (int(config.device_id), int(config.segments_per_coil), int(nfp),
            int(config.axis_grid), int(config.axis_trace_steps))


class AxisBatchUnavailable(RuntimeError):
    def __init__(self, symbols):
        self.missing_symbols=tuple(symbols)
        super().__init__('native axis batch API unavailable: '+', '.join(symbols))


class AxisBatchLibrary:
    def __init__(self, path, *, include_refinement=True):
        self.path = str(path)
        self.lib = ctypes.CDLL(self.path)
        required=['sgpu_axis_coarse_grid','sgpu_axis_coarse_cache_set',
            'sgpu_axis_coarse_cache_clear','sgpu_axis_coarse_cache_status',
            'sgpu_batch_field_use_host_segments']
        if include_refinement:
            required+=['sgpu_axis_coarse_domain','sgpu_axis_refinement_batch',
                       'sgpu_axis_refinement_cache_set','sgpu_axis_refinement_cache_status']
        missing=[name for name in required if not hasattr(self.lib,name)]
        if missing:raise AxisBatchUnavailable(missing)
        native._bind_native_score(self.lib)
        p, i, cp = ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(native._SgpuScoreConfig)
        specs = {
            'sgpu_axis_coarse_grid': [p,p,p,i,i,cp,p,p,i],
            'sgpu_axis_coarse_cache_set': [p,p,p,p,i,i,i,cp,p,p,p,p,i],
            'sgpu_axis_coarse_cache_clear': [],
            'sgpu_axis_coarse_cache_status': [],
            'sgpu_batch_field_use_host_segments': [ctypes.c_void_p,p,p,p,p,i,i,i,i],
        }
        for name, arguments in specs.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = i

    def check(self, code):
        native._check_lib_code(self.lib, code)

    def config(self, config, *, device=0, nfp=4):
        if not isinstance(config,EvaluationConfig):
            raise TypeError('one EvaluationConfig is required per geometry')
        cfg=native._SgpuScoreConfig()
        self.check(self.lib.sgpu_default_score_config(ctypes.byref(cfg)))
        cfg.device_id=int(device)
        cfg.target_M,cfg.target_N=config.helicity
        native._apply_score_config_overrides(cfg,config.native_options())
        if cfg.axis_hint_enabled or cfg.axis_hint_require_continuation:
            raise ValueError('axis batching requires ordinary unhinted selection')
        return cfg

    def grid(self, physical, config):
        x,y,z,_ = _coefficients(physical)
        count = int(config.axis_grid)**2
        R,Z = np.empty(count, dtype=np.float64), np.empty(count, dtype=np.float64)
        self.check(self.lib.sgpu_axis_coarse_grid(_pointer(x), _pointer(y), _pointer(z),
            len(x), x.shape[-1], ctypes.byref(config), _pointer(R), _pointer(Z), count))
        return R,Z

    def host_segments(self, batch, physical, *, nfp, segments):
        x,y,z,current = _coefficients(physical)
        self.check(self.lib.sgpu_batch_field_use_host_segments(batch.handle,
            _pointer(x), _pointer(y), _pointer(z), _pointer(current),
            physical.shape[1], x.shape[-1], int(nfp), int(segments)))

    def set(self, entry):
        x,y,z,current = entry.coefficients
        R0,Z0,R1,Z1 = entry.traces
        self.check(self.lib.sgpu_axis_coarse_cache_set(_pointer(x), _pointer(y), _pointer(z),
            _pointer(current), len(x), x.shape[-1], entry.nfp, ctypes.byref(entry.config),
            _pointer(R0), _pointer(Z0), _pointer(R1), _pointer(Z1), len(R0)))

    def status(self):
        return int(self.lib.sgpu_axis_coarse_cache_status())

    def clear(self):
        self.check(self.lib.sgpu_axis_coarse_cache_clear())

    def _bind_refinement(self):
        if getattr(self,'_refinement_bound',False):
            return
        p,i = ctypes.POINTER(ctypes.c_double),ctypes.c_int
        ip,cp = ctypes.POINTER(ctypes.c_int),ctypes.POINTER(native._SgpuScoreConfig)
        specs = {
            'sgpu_axis_coarse_domain':[p,p,p,i,i,cp,p],
            'sgpu_axis_refinement_batch':[ctypes.c_void_p,cp,i,i,p,p,p,p,p,i,i,p,ip,ip],
            'sgpu_axis_refinement_cache_set':[cp,p,ip,i],
            'sgpu_axis_refinement_cache_status':[],
            'sgpu_axis_refinement_reference':[ctypes.c_void_p,cp,i,p,p,i,p],
        }
        for name,arguments in specs.items():
            function = getattr(self.lib,name,None)
            if function is None:
                if name=='sgpu_axis_refinement_reference':continue
                raise AxisBatchUnavailable([name])
            function.argtypes,function.restype = arguments,i
        self._refinement_bound = True

    def domain(self,physical,config):
        """Return original build_axis_domain boundaries, not raster estimates."""
        self._bind_refinement()
        x,y,z,_ = _coefficients(physical)
        domain = np.empty(4,dtype=np.float64)
        self.check(self.lib.sgpu_axis_coarse_domain(_pointer(x),_pointer(y),_pointer(z),
            len(x),x.shape[-1],ctypes.byref(config),_pointer(domain)))
        return domain

    def refine_batch(self,batch,configs,nfp,traces,domains):
        self._bind_refinement()
        count = len(configs)
        capacity = max(int(c.axis_max_candidates) for c in configs)
        contiguous_configs = (native._SgpuScoreConfig*count)(*configs)
        payload = np.full((count,capacity,6),np.nan,dtype=np.float64)
        winding = np.full((count,capacity),-1,dtype=np.int32)
        counts = np.zeros(count,dtype=np.int32)
        ip = ctypes.POINTER(ctypes.c_int)
        self.check(self.lib.sgpu_axis_refinement_batch(batch.handle,contiguous_configs,count,int(nfp),
            *[_pointer(t) for t in traces],_pointer(domains),int(configs[0].axis_grid),capacity,
            _pointer(payload),winding.ctypes.data_as(ip),counts.ctypes.data_as(ip)))
        if np.any(counts < 0) or np.any(counts > capacity):
            raise RuntimeError('native refinement returned invalid candidate counts')
        return [AxisRefinementPayload(np.ascontiguousarray(payload[i,:n]).copy(),
                                     np.ascontiguousarray(winding[i,:n]).copy())
                for i,n in enumerate(counts)]

    def set_refinement(self,entry):
        self._bind_refinement()
        payload = entry.refinement
        self.check(self.lib.sgpu_axis_refinement_cache_set(ctypes.byref(entry.config),
            _pointer(payload.values),payload.winding.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(payload.values)))

    def refinement_status(self):
        self._bind_refinement()
        return int(self.lib.sgpu_axis_refinement_cache_status())

    def refinement_reference(self,single_field,entry):
        """Validation only: invoke original single-query candidate refinement."""
        self._bind_refinement()
        if entry.domain is None:
            raise ValueError('original native domain is required for refinement reference')
        if not hasattr(self.lib,'sgpu_axis_refinement_reference'):
            raise AxisBatchUnavailable(['sgpu_axis_refinement_reference'])
        domain = entry.domain
        payload = entry.refinement.values
        result = np.empty((len(payload),3),dtype=np.float64)
        self.check(self.lib.sgpu_axis_refinement_reference(single_field.handle,ctypes.byref(entry.config),
            entry.nfp,_pointer(domain),_pointer(payload),len(payload),_pointer(result)))
        return result


@dataclass
class AxisRefinementPayload:
    values: np.ndarray
    winding: np.ndarray

    def __post_init__(self):
        self.values = np.ascontiguousarray(self.values,dtype=np.float64)
        self.winding = np.ascontiguousarray(self.winding,dtype=np.int32)
        if self.values.ndim != 2 or self.values.shape[1] != 6 or self.winding.shape != (len(self.values),):
            raise ValueError('refinement payload must be [candidate,6] with one winding flag per candidate')


@dataclass
class AxisCoarseEntry:
    library: object
    coefficients: tuple
    config: object
    traces: tuple
    nfp: int = 4
    observations: list = field(default_factory=list)
    refinement: AxisRefinementPayload | None = None
    domain: np.ndarray | None = None
    owner_thread: int = field(default_factory=threading.get_ident)

    @contextmanager
    def capture(self, *, require_consumed=True):
        """Arm immediately around one native capture, always clear on exit.

        A different geometry/configuration fails the key and rejects that score;
        a later unarmed call uses ordinary native tracing. This wrapper records
        and rejects a nonconsumed outcome by default.
        It never clears someone else's already-armed cache to make room.
        """
        if threading.get_ident()!=self.owner_thread:
            raise RuntimeError('axis capture must use its preparing host thread')
        self.library.set(self)
        observation = dict(body_failed=False,refinement_requested=self.refinement is not None)
        try:
            try:
                observation['status_before_capture'] = self.library.status()
                if self.refinement is not None:
                    self.library.set_refinement(self)
                    observation['refinement_status_before_capture'] = self.library.refinement_status()
                yield self
            except BaseException:
                observation['body_failed'] = True
                raise
        finally:
            try:
                status = self.library.status()
                observation.update(status=status, status_name=STATUS_NAMES.get(status, 'unknown'),
                                   consumed=status == 2)
                if self.refinement is not None:
                    refinement_status = self.library.refinement_status()
                    observation.update(refinement_status=refinement_status,
                        refinement_status_name=STATUS_NAMES.get(refinement_status,'unknown'),
                        refinement_consumed=refinement_status == 2)
            finally:
                try:
                    self.library.clear()
                    observation['cleared'] = True
                finally:
                    self.observations.append(observation)
            if require_consumed and not observation['body_failed'] and not observation.get('consumed'):
                raise RuntimeError('native coarse cache was not consumed: '+observation.get('status_name', 'unknown'))
            if require_consumed and not observation['body_failed'] and self.refinement is not None and not observation.get('refinement_consumed'):
                raise RuntimeError('native refinement cache was not consumed: '+observation.get('refinement_status_name','unknown'))


@dataclass
class PreparedAxisBatch:
    entries: list
    metadata: dict
    query_count: int

    @property
    def enabled(self):
        return bool(self.metadata['enabled'])

    def capture(self,index,*,require_consumed=True):
        index=integer_index(index)
        if not 0<=index<self.query_count:raise IndexError('axis batch query index out of range')
        if self.enabled:return self.entries[index].capture(require_consumed=require_consumed)
        return self._disabled_capture(index)

    @contextmanager
    def _disabled_capture(self,index):
        yield None

    def observations(self):
        return [entry.observations.copy() for entry in self.entries]


def prepare_axis_batch(physical,configs,native_library=None,*,device=0,nfp=4,
                       include_refinement=True,allow_missing=True):
    """Prepare ordinary axis search for independent geometry representatives.

    ``physical`` has shape ``(B,4,3*ncoeff+1)`` in metres/amperes. Supply one
    EvaluationConfig per representative. Wrap exactly one ordinary native score
    or two-stage ``evaluate_many`` group in ``with prepared.capture(i)``.

    Preparation never arms a native slot. One-query batches and libraries without
    the additive batch API return ``enabled=False`` and an explicit reason in
    metadata. ``allow_missing=False`` makes unavailable symbols an error.
    """
    started=time.perf_counter()
    if hasattr(physical,'detach'):
        physical=physical.detach().double().cpu().numpy()
    physical=np.asarray(physical,dtype=np.float64)
    width=physical.shape[-1] if physical.ndim else 0
    coefficients=(width-1)//3
    if (physical.ndim!=3 or physical.shape[1]!=4 or coefficients<3 or
        width!=3*coefficients+1 or coefficients%2!=1 or not len(physical) or
        not np.isfinite(physical).all()):
        raise ValueError('physical input must be finite [query,4,3*ncoeff+1] with odd ncoeff>=3')
    configs=list(configs)
    if len(configs)!=len(physical) or any(not isinstance(c,EvaluationConfig) for c in configs):
        raise ValueError('provide one EvaluationConfig per geometry')
    if isinstance(device,bool) or int(device)!=device or device<0:
        raise ValueError('device must be a nonnegative integer')
    if isinstance(nfp,bool) or int(nfp)!=nfp or nfp<=0:
        raise ValueError('nfp must be a positive integer')
    metadata=dict(enabled=False,batch=len(physical),include_refinement=bool(include_refinement),
        geometry_inputs='native CPU exact raster and original CPU coil segments',
        selector='ordinary winding, fallback, FP64, topology and priority selection',
        pending_native_cache=False,timings={})
    def disabled(reason,**extra):
        metadata.update(status='disabled',reason=reason,**extra)
        metadata['timings']['total']=time.perf_counter()-started
        return PreparedAxisBatch([],metadata,len(physical))
    if len(physical)==1:return disabled('single_query')
    path=resolve_library(native_library)
    try:
        library=AxisBatchLibrary(path,include_refinement=include_refinement)
    except AxisBatchUnavailable as exc:
        if not allow_missing:raise
        return disabled('native_api_unavailable',missing_symbols=list(exc.missing_symbols))
    native_configs=[library.config(c,device=device,nfp=nfp) for c in configs]
    signatures=[_trace_signature(c,nfp) for c in native_configs]
    if len(set(signatures))!=1:
        raise ValueError('mixed device/segments/nfp/grid/trace layouts require explicit groups')
    cfg=native_configs[0]
    metadata.update(enabled=True,mode='strict_native_coarse_and_refinement_cache' if include_refinement
        else 'strict_native_coarse_cache',grouping_signature=list(signatures[0]))
    def measure(name,fn):
        tick=time.perf_counter();value=fn();metadata['timings'][name]=time.perf_counter()-tick
        return value
    grids=measure('native_exact_grids',lambda:[library.grid(row,c) for row,c in zip(physical,native_configs)])
    domains=(measure('native_exact_domains',lambda:np.ascontiguousarray(
        [library.domain(row,c) for row,c in zip(physical,native_configs)],dtype=np.float64))
        if include_refinement else None)
    R0=np.ascontiguousarray([v[0] for v in grids]);Z0=np.ascontiguousarray([v[1] for v in grids])
    blocks=_coefficients(physical)
    batch=measure('native_batch_construct',lambda:native.BatchCoilFieldGpu(
        str(path),*blocks,nfp,segments_per_coil=int(cfg.segments_per_coil),device_id=int(device)))
    try:
        measure('native_host_segment_replace',lambda:library.host_segments(batch,physical,
            nfp=nfp,segments=cfg.segments_per_coil))
        R1,Z1=measure('batch_coarse_trace',lambda:batch.trace_period(R0,Z0,steps=int(cfg.axis_trace_steps)))
        refinements=(measure('batch_candidate_refinement',lambda:library.refine_batch(
            batch,native_configs,nfp,(R0,Z0,R1,Z1),domains)) if include_refinement else [None]*len(physical))
    finally:
        batch.close()
    entries=[AxisCoarseEntry(library,_coefficients(row),cfg,
        tuple(np.ascontiguousarray(v[i],dtype=np.float64).copy() for v in (R0,Z0,R1,Z1)),int(nfp),
        refinement=refinements[i],domain=domains[i].copy() if include_refinement else None)
        for i,(row,cfg) in enumerate(zip(physical,native_configs))]
    metadata.update(status='prepared',grid=int(cfg.axis_grid),trace_steps=int(cfg.axis_trace_steps),
        segments_per_coil=int(cfg.segments_per_coil),nfp=int(nfp),coarse_line_count=int(cfg.axis_grid)**2,
        finite_endpoint_count=(np.isfinite(R1)&np.isfinite(Z1)).sum(-1).tolist(),
        complete_finite_endpoints=(np.isfinite(R1)&np.isfinite(Z1)).all(-1).tolist())
    metadata['timings']['total']=time.perf_counter()-started
    metadata['timings']['amortized_per_query']=metadata['timings']['total']/len(physical)
    return PreparedAxisBatch(entries,metadata,len(physical))
