"""Two scalar fits and complete native continuation on the replacement model."""
from __future__ import annotations
import ctypes as ct
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from .. import _binding as native
from .._autograd.capture import _bind as bind_capture, copy_reverse_snapshot, _config_dict
from .model import PsiModel
from .boundary import max_closed_boundary
from .sampling import sample_frozen_sections

_D=ct.POINTER(ct.c_double)
_I=ct.POINTER(ct.c_int)

_COMMON_CAPTURE_ARRAYS = (
    'capture_version', 'nfp', 'axis_R', 'axis_Z', 'axis_R_phi', 'axis_Z_phi', 'axis_samples',
    'psi_coefficients', 'psi_mode_a', 'psi_mode_b', 'psi_mode_m', 'psi_mode_kind',
    'alpha_mode_l', 'alpha_mode_m', 'alpha_mode_n', 'alpha_mode_kind', 'alpha_radial_coefficients',
    'alpha_radial_basis_mode',
)
_SOURCE_CAPTURE_ARRAYS = frozenset(_COMMON_CAPTURE_ARRAYS) | {
    'target_level', 'alpha_coefficients', 'alpha_iota_coefficients',
    # Old libraries expose the physical alpha solution through these vectors.
    'alpha_scale', 'alpha_center_scaled_solution',
}


def _shared_array_tree(value):
    """Copy containers while retaining readonly owned group-common arrays."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return value
    if isinstance(value, dict):
        return {key: _shared_array_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shared_array_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_shared_array_tree(item) for item in value)
    return value
def pointer(a):
    return a.ctypes.data_as(_D)


class TwoStageNative:
    """Own the common physical field until both fits and downstream complete."""
    def __init__(self, library, device_id=0):
        self.lib=ct.CDLL(str(Path(library).resolve()))
        self.device_id=int(device_id)
        native._bind_native_score(self.lib)
        bind_capture(self.lib)
        self._context=None
        self.last_many_timing=None
        self.lib.sgpu_two_stage_begin_v1.argtypes=[_D,_D,_D,_D,ct.c_int,ct.c_int,ct.c_int,
            ct.POINTER(native._SgpuScoreConfig),ct.POINTER(native._SgpuScoreResult),
            ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p),_D,ct.c_int,_I,_D,ct.c_int,_I]
        self.lib.sgpu_two_stage_begin_v1.restype=ct.c_int
        self.lib.sgpu_two_stage_fit_v1.argtypes=[ct.c_void_p,_D,_D,_D,ct.c_int,_D,_D,_D,ct.c_int]
        self.lib.sgpu_two_stage_fit_v1.restype=ct.c_int
        alpha_weighting=getattr(self.lib,'sgpu_two_stage_set_alpha_radial_weighting_v1',None)
        if alpha_weighting is not None:
            alpha_weighting.argtypes=[ct.c_void_p,ct.c_int,ct.c_double,ct.c_double]
            alpha_weighting.restype=ct.c_int
        alpha_basis=getattr(self.lib,'sgpu_two_stage_set_alpha_radial_basis_v1',None)
        if alpha_basis is not None:
            alpha_basis.argtypes=[ct.c_void_p,ct.c_int]
            alpha_basis.restype=ct.c_int
        weighted_fit=getattr(self.lib,'sgpu_two_stage_fit_weighted_v1',None)
        if weighted_fit is not None:
            weighted_fit.argtypes=[ct.c_void_p,_D,_D,_D,ct.c_int,_D,_D,_D,_D,ct.c_int]
            weighted_fit.restype=ct.c_int
        refit_ridge=getattr(self.lib,'sgpu_two_stage_set_refit_ridge_v1',None)
        if refit_ridge is not None:
            refit_ridge.argtypes=[ct.c_void_p,ct.c_double]
            refit_ridge.restype=ct.c_int
        self.lib.sgpu_two_stage_resume_v1.argtypes=[ct.c_void_p,_D,ct.c_int,ct.c_double,
            ct.POINTER(native._SgpuScoreConfig),ct.POINTER(native._SgpuScoreResult),
            _D,ct.c_int,_I,_D,ct.c_char_p]
        self.lib.sgpu_two_stage_resume_v1.restype=ct.c_int
        self.lib.sgpu_two_stage_close_v1.argtypes=[ct.c_void_p]
        self.lib.sgpu_two_stage_close_v1.restype=None
        self.lib.sgpu_copy_psi_warm_preconditioner.argtypes=[_D,_D,ct.c_int]
        self.lib.sgpu_copy_psi_warm_preconditioner.restype=ct.c_int

    def _check(self, code):
        native._check_lib_code(self.lib,code)

    def _check_result(self,code,raw):
        # A native RAII capture guard can clear the thread-local error during
        # unwinding. The result owns the stage-qualified failure message.
        message=bytes(raw.error_message).split(b'\0',1)[0] if code else b''
        if message:
            raise native.GpuError(message.decode('utf-8','replace'))
        self._check(code)

    def _factor(self, count):
        upper=np.empty((count,count),dtype=np.float64)
        scale=np.empty(count,dtype=np.float64)
        self._check(self.lib.sgpu_copy_psi_warm_preconditioner(pointer(upper),pointer(scale),count))
        return scale,upper

    def _config(self, config, axis_hint):
        cfg=native._SgpuScoreConfig()
        self._check(self.lib.sgpu_default_score_config(ct.byref(cfg)))
        cfg.device_id=self.device_id
        cfg.target_M,cfg.target_N=config.helicity
        options=config.native_options()
        if axis_hint is not None:
            options.update(axis_hint_enabled=1,axis_hint_require_continuation=2,
                           axis_hint_R=float(axis_hint[0]),axis_hint_Z=float(axis_hint[1]))
        native._apply_score_config_overrides(cfg,options)
        return cfg

    def _prepare(self,coils,config,cfg,arrays,capture,timings):
        """Create one field/S1/domain/S2 model, independently of target volume."""
        initial=self._begin_prepare(coils,config,cfg,arrays,capture,timings)
        if 'partial_result' in initial:
            return initial
        domain=self._prepare_domain(coils,config,cfg,initial,timings)
        prepared=self._finish_prepare(cfg,initial,domain,capture,timings)
        prepared['axis']=initial['axis']
        return prepared

    def _begin_prepare(self,coils,config,cfg,arrays,capture,timings):
        """Run native S1 on the caller thread and own all exported host data."""
        modes=((cfg.psi_poly_degree+1)*(cfg.psi_poly_degree+2)//2-3)*(2*cfg.psi_m_tor+1)-1
        axes=np.empty(4*cfg.axis_sample_count,dtype=np.float64)
        first=np.empty(modes,dtype=np.float64)
        count,npsi=ct.c_int(),ct.c_int()
        context,field=ct.c_void_p(),ct.c_void_p()
        raw=native._SgpuScoreResult()
        # The native begin routine also clears this parameter internally. An
        # explicit copy makes volume independence visible at the Python C ABI.
        begin_cfg=native._SgpuScoreConfig.from_buffer_copy(bytes(cfg))
        begin_cfg.target_volume=0.
        stage=perf_counter()
        code=self.lib.sgpu_two_stage_begin_v1(*[pointer(a) for a in arrays],
            *arrays[0].shape,int(coils.nfp),ct.byref(begin_cfg),ct.byref(raw),ct.byref(context),
            ct.byref(field),pointer(axes),cfg.axis_sample_count,ct.byref(count),
            pointer(first),modes,ct.byref(npsi))
        # Own a returned context before checking the return code, so even a
        # partially allocated failing C call is released by the outer finally.
        self._context=context
        self._check_result(code,raw)
        timings['stage1']=perf_counter()-stage
        if raw.status!=0 or not context.value:
            return dict(partial_result=native._score_result_dict(raw))
        alpha_options=dict(alpha_radial_weighting=getattr(config,'alpha_radial_weighting','histogram'),
            alpha_weight_sigma=float(getattr(config,'alpha_weight_sigma',.05)),
            alpha_count_floor=float(getattr(config,'alpha_count_floor',32.)),
            alpha_radial_basis=getattr(config,'alpha_radial_basis','zernike'))
        if alpha_options['alpha_radial_basis']!='zernike':
            setter=getattr(self.lib,'sgpu_two_stage_set_alpha_radial_basis_v1',None)
            if setter is None:
                raise RuntimeError('native library does not provide the selected alpha radial basis; rebuild the library')
            if alpha_options['alpha_radial_basis']!='flux_legendre':
                raise ValueError('alpha_radial_basis must be zernike or flux_legendre')
            self._check(setter(context,1))
        if alpha_options['alpha_radial_weighting']=='smooth':
            setter=getattr(self.lib,'sgpu_two_stage_set_alpha_radial_weighting_v1',None)
            if setter is None:
                raise RuntimeError('native library does not provide smooth alpha weighting; rebuild the library')
            self._check(setter(context,1,alpha_options['alpha_weight_sigma'],alpha_options['alpha_count_floor']))
        refit_ridge=getattr(config,'psi_refit_ridge',None)
        if refit_ridge is not None:
            setter=getattr(self.lib,'sgpu_two_stage_set_refit_ridge_v1',None)
            if setter is None:
                raise RuntimeError('native library does not support psi_refit_ridge; rebuild the library')
            self._check(setter(context,refit_ridge))
            alpha_options['psi_refit_ridge']=float(refit_ridge)
        axis=axes[:4*count.value].reshape(4,count.value).T.copy()
        first=first[:npsi.value].copy()
        s1_scale,s1_upper=self._factor(len(first)) if capture else (None,None)
        return dict(axis=axis,first=first,alpha_options=alpha_options,
                    s1_scale=s1_scale,s1_upper=s1_upper)

    @staticmethod
    def _prepare_domain(coils,config,cfg,initial,timings):
        """Pure host geometry work; it never reads native thread-local state."""
        axis,first=initial['axis'],initial['first']
        model=PsiModel(first,axis,nfp=coils.nfp,a=cfg.psi_a,
                       degree=cfg.psi_poly_degree,mtor=cfg.psi_m_tor)
        phis=np.arange(cfg.psi_n_phi)*2*np.pi/(coils.nfp*cfg.psi_n_phi)
        stage=perf_counter()
        boundary=max_closed_boundary(model,phis,ntheta=256,max_radius=cfg.psi_a)
        timings['training_boundary']=perf_counter()-stage
        stage=perf_counter()
        if getattr(config,'psi_refit_sampling','cartesian')=='polar_quadrature':
            from .quadrature import sample_polar_quadrature
            sample=sample_polar_quadrature(boundary,target_count=config.psi_refit_points,nfp=coils.nfp)
        else:
            sample=sample_frozen_sections(boundary['polygons_rz'],phis,
                target_count=config.psi_refit_points,nfp=coils.nfp)
        timings['training_points']=perf_counter()-stage
        rows=[np.ascontiguousarray(sample['rz'][:,i]) for i in range(2)]
        rows.append(np.ascontiguousarray(sample['phi']))
        return dict(boundary=boundary,sample=sample,rows=rows)

    def _finish_prepare(self,cfg,initial,domain,capture,timings):
        """Fit S2 on the caller thread, copying its factor before another fit."""
        first=initial['first']
        s1_scale,s1_upper=initial['s1_scale'],initial['s1_upper']
        boundary,sample,rows=domain['boundary'],domain['sample'],domain['rows']
        coefficients=np.empty_like(first)
        rms=ct.c_double();stats=np.zeros(16,dtype=np.float64)
        stage=perf_counter()
        weights=sample.get('row_weights')
        if weights is None:
            self._check(self.lib.sgpu_two_stage_fit_v1(self._context,*[pointer(a) for a in rows],
                len(rows[0]),pointer(coefficients),ct.byref(rms),pointer(stats),len(stats)))
        else:
            weights=np.ascontiguousarray(weights,dtype=np.float64)
            if weights.shape!=(len(rows[0]),) or not np.isfinite(weights).all() or np.any(weights<=0):
                raise ValueError('two-stage row_weights must be a finite positive vector matching the training rows')
            if not np.isclose(weights.mean(),1.,rtol=0.,atol=1e-6):
                raise ValueError('two-stage row_weights must have mean one')
            weighted_fit=getattr(self.lib,'sgpu_two_stage_fit_weighted_v1',None)
            if weighted_fit is None:
                raise RuntimeError('native library does not provide weighted two-stage fitting; rebuild the library')
            self._check(weighted_fit(self._context,*[pointer(a) for a in rows],len(rows[0]),pointer(weights),
                pointer(coefficients),ct.byref(rms),pointer(stats),len(stats)))
        timings['stage2_fit']=perf_counter()-stage
        s2_scale,s2_upper=self._factor(len(first)) if capture else (None,None)
        for factor in (s1_scale,s1_upper,s2_scale,s2_upper):
            if factor is not None:
                factor.setflags(write=False)
        members=hashlib.sha256()
        if sample.get('kind')=='polar_quadrature':
            for key,dtype in (('theta_indices','<i8'),('radial_u','<f8'),('section_index','<i8')):
                members.update(np.ascontiguousarray(sample[key],dtype=dtype).tobytes())
        else:
            for values in (sample['section_index'],sample['lattice_indices']):
                members.update(np.ascontiguousarray(values,dtype='<i8').tobytes())
        branch=dict(limiting_ray=boundary['diagnostics']['limiting_ray'],
            capacity_source=boundary['diagnostics']['active_capacity_source'],
            capacity_gap=boundary['diagnostics']['active_capacity_gap'],
            sampling_members_sha256=members.hexdigest(),spacing=sample['spacing'],training_count=len(rows[0]),
            sampling_kind=sample.get('kind','cartesian'),weighted_fit=weights is not None)
        return dict(coefficients=coefficients,train_rms=rms.value,first=first,
            axis=initial['axis'],
            alpha_options=initial['alpha_options'],
            s1_scale=s1_scale,s1_upper=s1_upper,s2_scale=s2_scale,s2_upper=s2_upper,
            boundary=boundary,sample=sample,branch=branch,training_count=len(rows[0]))

    def _prepare_target(self,prepared,config,cfg,capture,timings):
        """Keep S1/S2 shared; own one S3 and its factors per target volume."""
        if not getattr(config,'psi_target_refit',False):
            return prepared
        from .target_domain import target_training_domain
        stage=perf_counter()
        model=PsiModel(prepared['coefficients'],prepared['axis'],nfp=prepared['nfp'],
            a=cfg.psi_a,degree=cfg.psi_poly_degree,mtor=cfg.psi_m_tor)
        phis=np.arange(cfg.psi_n_phi)*2*np.pi/(model.nfp*cfg.psi_n_phi)
        metadata=target_training_domain(model,phis,config.target_volume,
            factor=config.psi_target_refit_factor,target_count=config.psi_refit_points)
        timings['target_training_domain']=perf_counter()-stage
        metadata.update(stage2_coefficients=prepared['coefficients'],
            stage2_scale=prepared['s2_scale'],stage2_upper=prepared['s2_upper'])
        sample=metadata['sampling']
        rows=[np.ascontiguousarray(sample['rz'][:,i]) for i in range(2)]
        rows.append(np.ascontiguousarray(sample['phi']))
        weights=np.ascontiguousarray(sample['row_weights'])
        setter=getattr(self.lib,'sgpu_two_stage_set_refit_ridge_v1',None)
        if setter is None:
            raise RuntimeError('native library lacks target-refit regularization support; rebuild the library')
        coefficients=np.empty_like(prepared['coefficients'])
        rms=ct.c_double();stats=np.zeros(16,dtype=np.float64)
        stage=perf_counter()
        self._check(setter(self._context,config.psi_target_refit_ridge))
        try:
            self._check(self.lib.sgpu_two_stage_fit_weighted_v1(self._context,
                *[pointer(a) for a in rows],len(rows[0]),pointer(weights),pointer(coefficients),
                ct.byref(rms),pointer(stats),len(stats)))
            scale,upper=self._factor(len(coefficients)) if capture else (None,None)
        finally:
            self._check(setter(self._context,config.psi_refit_ridge if config.psi_refit_ridge is not None else cfg.psi_ridge))
        timings['stage3_fit']=perf_counter()-stage
        for factor in (scale,upper):
            if factor is not None:
                factor.setflags(write=False)
        result=dict(prepared,coefficients=coefficients,train_rms=rms.value,
            s2_scale=scale,s2_upper=upper,target_refit=metadata,
            alpha_options={**prepared['alpha_options'],
                'psi_target_refit':True,'psi_target_refit_factor':config.psi_target_refit_factor,
                'psi_target_refit_ridge':config.psi_target_refit_ridge})
        for key in ('shared_snapshot_arrays','shared_two_stage','native_source_capture'):
            result.pop(key,None)
        return result

    @staticmethod
    def _prepare_snapshot_arrays(prepared,capture_mode,source_native):
        prepared['native_source_capture']=source_native
        prepared['shared_snapshot_arrays']=({
            'psi_scale':prepared['s2_scale'],'psi_upper':prepared['s2_upper']}
            if capture_mode=='reverse' else {})
        prepared['shared_two_stage']=_shared_array_tree(dict(
            stage1_coefficients=prepared['first'],stage1_scale=prepared['s1_scale'],
            stage1_upper=prepared['s1_upper'],boundary=prepared['boundary'],sampling=prepared['sample']))
        if 'target_refit' in prepared:
            prepared['shared_two_stage']['target_refit']=_shared_array_tree(prepared['target_refit'])

    def _resume(self,prepared,cfg,arrays,capture_mode,include_torsion):
        coefficients=prepared['coefficients']
        raw=native._SgpuScoreResult()
        iota=np.empty(cfg.iota_degree+1,dtype=np.float64)
        niota,edge_iota=ct.c_int(),ct.c_double()
        stage=perf_counter()
        code=self.lib.sgpu_two_stage_resume_v1(self._context,pointer(coefficients),len(coefficients),
            prepared['train_rms'],ct.byref(cfg),ct.byref(raw),pointer(iota),len(iota),ct.byref(niota),
            ct.byref(edge_iota),None)
        self._check_result(code,raw)
        formal_seconds=perf_counter()-stage
        stage=perf_counter()
        result=native._score_result_dict(raw)
        if include_torsion:
            result=native._with_torsion(self.lib,result,*arrays[:3])
        result['diagnostics'].update(iota_target_surface=edge_iota.value,psi_fit_method='two_stage',
            **prepared['alpha_options'],
            psi_refit_domain_volume=prepared['boundary']['volume_m3'],
            psi_refit_training_count=prepared['training_count'],psi_refit_branch=deepcopy(prepared['branch']))
        result['diagnostics']['psi_fit_stages']=3 if 'target_refit' in prepared else 2
        if 'target_refit' in prepared:
            result['diagnostics']['psi_target_refit_domain']=dict(
                prepared['target_refit']['boundary']['diagnostics'],
                training_count=len(prepared['target_refit']['sampling']['rz']))
        snapshot=None
        if capture_mode is not None and raw.status==0:
            # All per-volume native arrays are copied before the next resume
            # replaces thread-local native capture storage. Only the explicitly
            # read-only common preconditioners are shared across snapshots.
            shared=prepared['shared_snapshot_arrays']
            snapshot=copy_reverse_snapshot(self,shared=shared,
                names=_SOURCE_CAPTURE_ARRAYS if capture_mode=='sources' else None)
            for name in _COMMON_CAPTURE_ARRAYS:
                if name in snapshot and name not in shared:
                    snapshot[name].setflags(write=False)
                    shared[name]=snapshot[name]
            snapshot.update(config=_config_dict(cfg),score_result=result)
            snapshot['config'].update(prepared['alpha_options'])
            if capture_mode=='reverse':
                snapshot['psi_scale'],snapshot['psi_upper']=prepared['s2_scale'],prepared['s2_upper']
            snapshot['two_stage']=_shared_array_tree(prepared['shared_two_stage'])
            snapshot['metadata']=dict(capture_version=1,psi_fit_method='two_stage',
                capture_kind=capture_mode,
                source_capture_backend=('native' if prepared['native_source_capture'] else 'legacy_host_filter')
                    if capture_mode=='sources' else None,
                discrete_choices=('fixed active limiting ray and quadrature nodes for local derivatives'
                    if prepared['sample'].get('kind')=='polar_quadrature' else
                    'fixed active limiting ray and lattice membership for local derivatives'))
            if 'alpha_center_scaled_solution' in snapshot:
                nmode=len(snapshot['alpha_mode_l'])
                snapshot['alpha_coefficients']=(snapshot['alpha_center_scaled_solution']/snapshot['alpha_scale'])[:nmode]
        return (result,snapshot),formal_seconds,perf_counter()-stage

    def evaluate(self,coils,config,*,capture=False,diagnostics=False,axis_hint=None,include_torsion=True):
        """Return a raw result plus an optional owned reverse/initializer snapshot."""
        return self.evaluate_many(coils,[config],capture=capture,diagnostics=diagnostics,
                                  axis_hint=axis_hint,include_torsion=include_torsion)[0]

    def evaluate_many(self,coils,configs,*,capture=False,diagnostics=False,axis_hint=None,include_torsion=True):
        """Evaluate volumes on one axis/S1/domain/S2 fit, preserving input order.

        Only ``target_volume`` may differ between configurations. Every native
        continuation recomputes the complete formal downstream chain. Contexts
        are closed before return; captured arrays are owned host storage.
        Group-common arrays are shared readonly; per-volume state is independent.
        ``include_torsion=False`` skips the additional host torsion diagnostic.
        ``capture='sources'`` returns only initialization source fields. New
        libraries omit reverse-state preparation; older libraries retain their
        full native preparation and filter the host copy for compatibility.

        ``last_many_timing`` and each result's ``timing['two_stage_group']``
        contain one measured shared preparation, the per-volume formal and
        postprocessing times, and actual whole-group wall time. The existing
        ``two_stage`` timing keys remain available per returned volume; their
        shared preparation must be counted only once for group throughput.
        """
        from .._backend import _coil_arrays
        if isinstance(capture,(bool,np.bool_)):
            capture_mode='reverse' if capture else None
        elif capture in ('reverse','sources'):
            capture_mode=capture
        else:
            raise ValueError("capture must be a boolean, 'reverse' or 'sources'")
        if capture_mode=='sources' and diagnostics:
            raise ValueError('source-only capture does not include reverse diagnostics')
        configs=list(configs)
        if not configs:
            self.last_many_timing=dict(query_count=0,completed_volumes=0,shared_preparation_s=0.,
                each_formal_s=[],each_postprocess_s=[],total_s=0.,overhead_s=0.)
            return []
        if len(configs)>1:
            reference=configs[0].to_dict()
            for config in configs[1:]:
                current=config.to_dict()
                different={key for key in reference.keys()|current.keys()
                           if key!='target_volume' and reference.get(key)!=current.get(key)}
                if different:
                    raise ValueError('two-stage evaluate_many permits only target_volume differences: '+', '.join(sorted(different)))
        self.close()
        self.last_many_timing=None
        started=perf_counter()
        arrays=_coil_arrays(coils)
        native_configs=[self._config(config,axis_hint) for config in configs]
        baseline=_config_dict(native_configs[0]);baseline.pop('target_volume')
        for cfg in native_configs[1:]:
            options=_config_dict(cfg);options.pop('target_volume')
            if options!=baseline:
                raise ValueError('native two-stage configuration differs beyond target_volume')
        timings={};results=[];formal=[];postprocess=[];target_times=[];query_timings=[]
        shared_seconds=None
        try:
            capability=getattr(self.lib,'sgpu_source_capture_version',None)
            source_native=False
            if capture_mode=='sources' and capability is not None:
                capability.argtypes=[];capability.restype=ct.c_int
                source_native=int(capability())>=1
            flag=4 if source_native else 3 if diagnostics else 1 if capture_mode is not None else 0
            self._check(self.lib.sgpu_set_reverse_capture(flag))
            prepared=self._prepare(coils,configs[0],native_configs[0],arrays,capture_mode=='reverse',timings)
            prepared['nfp']=coils.nfp
            if capture_mode is not None and 'partial_result' not in prepared:
                self._prepare_snapshot_arrays(prepared,capture_mode,source_native)
            shared_seconds=perf_counter()-started
            if 'partial_result' in prepared:
                results=[(deepcopy(prepared['partial_result']),None) for _ in configs]
                formal=[0.]*len(configs);postprocess=[0.]*len(configs)
                target_times=[0.]*len(configs)
            else:
                for config,cfg in zip(configs,native_configs):
                    tick=perf_counter();local_timing={}
                    selected=self._prepare_target(prepared,config,cfg,capture_mode=='reverse',local_timing)
                    if selected is not prepared and capture_mode is not None:
                        self._prepare_snapshot_arrays(selected,capture_mode,source_native)
                    target_times.append(perf_counter()-tick if selected is not prepared else 0.)
                    query_timings.append(local_timing)
                    result,formal_seconds,postprocess_seconds=self._resume(selected,cfg,arrays,capture_mode,include_torsion)
                    results.append(result);formal.append(formal_seconds);postprocess.append(postprocess_seconds)
        finally:
            self.lib.sgpu_set_reverse_capture(0)
            self.close()
            total=perf_counter()-started
            if shared_seconds is None:
                shared_seconds=total
            self.last_many_timing=dict(query_count=len(configs),completed_volumes=len(results),
                shared_preparation_s=shared_seconds,each_formal_s=list(formal),
                each_target_preparation_s=list(target_times),
                each_postprocess_s=list(postprocess),total_s=total,
                overhead_s=max(0.,total-shared_seconds-sum(formal)-sum(postprocess)-sum(target_times)))
        for index,(result,snapshot) in enumerate(results):
            if 'partial_result' not in prepared:
                result['timing']['two_stage']=dict(timings,**query_timings[index],formal_downstream=formal[index],
                    total=(self.last_many_timing['total_s'] if len(configs)==1 else
                           shared_seconds+target_times[index]+formal[index]+postprocess[index]))
            result['timing']['two_stage_group']=deepcopy(self.last_many_timing)
        return results

    def close(self):
        if self._context is not None and self._context.value:
            self.lib.sgpu_two_stage_close_v1(self._context)
        self._context=None
