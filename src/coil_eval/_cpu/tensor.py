"""CPU tensor frontend using the shared local-response mathematics."""
from __future__ import annotations
from dataclasses import asdict, fields
from operator import index
import threading
from time import perf_counter
import numpy as np
import torch
from .backend import CpuBackend
from .autograd import CpuAutogradLibrary,CpuSegmentField
from .threads import thread_scope
from .._autograd.engineering import split_tokens
from .._autograd.field import coil_segments
from .._autograd.front import build_front
from .._autograd.two_stage_front import build_two_stage_front
from .._autograd.downstream import DownstreamConfig,downstream
from .._tensor_config import validate_tensor_config,validate_tensor_batch_configs
from ..evaluator import CenterContextError,InvalidEvaluationError
from ..types import CoilSet


class CpuTensorEvaluator:
    def __init__(self,library=None,field_library=None,*,config,device_id=0,
                 field_dtype=torch.float32,batch_axis=True,share_geometry=True,surface_qs=None,
                 device='cpu',threads=None,axis_grid=None,nfp=4):
        validate_tensor_config(config)
        if isinstance(nfp,(bool,np.bool_)):
            raise ValueError('nfp must be a positive integer')
        try:
            self.nfp=index(nfp)
        except TypeError as exc:
            raise ValueError('nfp must be a positive integer') from exc
        if self.nfp<1:
            raise ValueError('nfp must be a positive integer')
        if field_dtype not in (torch.float32,torch.float64):
            raise ValueError('field_dtype must be torch.float32 or torch.float64')
        from ..qs import SurfaceQSConfig
        if surface_qs is not None and not isinstance(surface_qs,SurfaceQSConfig):
            raise TypeError('surface_qs must be SurfaceQSConfig or None')
        self.config,self.dtype,self.device=config,field_dtype,torch.device('cpu')
        self.backend=CpuBackend(library,threads=threads,axis_grid=axis_grid)
        self.threads=self.backend.threads;self.library_path=str(self.backend.library.path)
        self.field_library=CpuAutogradLibrary(field_library or self.library_path,self.threads)
        self.surface_qs=surface_qs
        self._thread=threading.get_ident();self._closed=False

    def _check(self):
        if self._closed or threading.get_ident()!=self._thread:
            raise CenterContextError('use an open evaluator on its creating thread')

    def _center(self,snapshot):
        names=('axis_samples','psi_coefficients','psi_scale','psi_upper','target_adjoint_weights',
               'target_axis_weights','target_phi','target_axis_R','target_level',
               'target_adjoint_residual','target_dVds')
        center={'config':snapshot['config'],**{k:torch.as_tensor(snapshot[k],dtype=torch.float64).detach() for k in names}}
        if 'two_stage' in snapshot:center['two_stage']=snapshot['two_stage']
        return center

    def evaluate(self,parameters):
        return self._evaluate(parameters,batch=False)

    def evaluate_batch(self,parameters,*,configs=None):
        return self._evaluate(parameters,batch=True,configs=configs)

    def _evaluate(self,parameters,*,batch,configs=None,_snapshots=None,_parameter_grad=True,**kwargs):
        from ..torch import TensorEvaluation,TensorMetrics
        self._check()
        x=parameters if isinstance(parameters,torch.Tensor) else torch.as_tensor(parameters,dtype=torch.float64)
        if x.device.type != 'cpu':
            raise ValueError(f'parameters are on {x.device}, but this evaluator uses cpu. '
                             'Use pack_coils(device="cpu") for CPU evaluation.')
        if (x.ndim!=(3 if batch else 2) or x.shape[-2]<1 or
            x.shape[-1]<10 or (x.shape[-1]-1)%3 or ((x.shape[-1]-1)//3)%2!=1 or
            x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all())):
            raise ValueError('parameters require finite CPU [coils,3*ncoeff+1] physical values with at least one coil (or a batch)')
        if not _parameter_grad:x=x.detach()
        elif not x.requires_grad:x=x.detach().requires_grad_(True)
        xs=x if batch else x[None]
        cfgs=validate_tensor_batch_configs([self.config]*len(xs) if configs is None else configs,count=len(xs))
        unknown=set(kwargs)-{'_local_value','_engineering'}
        if unknown:raise TypeError('unsupported CPU local options: '+', '.join(sorted(unknown)))
        start=perf_counter()
        with thread_scope(self.threads):
            results=[self._one(v,cfg,None if _snapshots is None else _snapshots[i],**kwargs)
                     for i,(v,cfg) in enumerate(zip(xs,cfgs))]
        if not batch:
            result=results[0]
            return TensorEvaluation(x,result.metrics,result.diagnostics,
                {**result.timings,'total':perf_counter()-start},result.surface_qs)
        metrics=TensorMetrics(**{f.name:torch.stack([getattr(r.metrics,f.name) for r in results])
                                for f in fields(TensorMetrics)})
        surface=None
        if self.surface_qs is not None:
            from ..qs import SurfaceQSResult
            surface=SurfaceQSResult(torch.stack([r.surface_qs.f_B for r in results]),
                torch.stack([r.surface_qs.f_T for r in results]) if self.surface_qs.compute_f_T else None,
                {**results[0].surface_qs.diagnostics,
                 'surfaces':[r.surface_qs.diagnostics['surfaces'][0] for r in results]})
        return TensorEvaluation(x,metrics,dict(device='cpu',cpu_threads=self.threads,nfp=self.nfp,
            queries=[r.diagnostics for r in results],gradient_method='local_reverse'),
            dict(total=perf_counter()-start,queries=[r.timings for r in results]),surface)

    def _one(self,x,config,snapshot=None,*,_local_value=False,_engineering=True,**kwargs):
        from ..torch import TensorMetrics,TensorEvaluation
        from .engineering import engineering_metrics
        timings={}
        def stage(name,fn):
            t=perf_counter();result=fn();timings[name]=perf_counter()-t;return result
        if snapshot is None:
            coils=CoilSet(*split_tokens(x.detach().double().numpy()),nfp=self.nfp)
            raw,snapshot=stage('center_capture',lambda:self.backend._capture(coils,config))
            if snapshot is None:raise InvalidEvaluationError(f'CPU center: {raw["status"]}: {raw["diagnostics"].get("error_message","")}')
        center=self._center(snapshot)
        if center['config']['nfp']!=self.nfp:
            raise ValueError('CPU snapshot field periods do not match evaluator nfp')
        p,w=coil_segments(x.double(),nfp=self.nfp,count=config.segments_per_coil)
        field=CpuSegmentField(p.to(self.dtype),w.to(self.dtype),self.field_library)
        front=stage('axis_psi_surface',lambda:build_two_stage_front(field,field.trace,center,dtype=torch.float64,
            local_value=_local_value,value_only=not x.requires_grad) if config.psi_fit=='two_stage'
            else build_front(field,field.trace,center,dtype=torch.float64))
        cache={name:torch.as_tensor(snapshot[key],dtype=self.dtype) for name,key in
            (('upper','alpha_upper'),('scale','alpha_scale'),('scaled_solution','alpha_center_scaled_solution'))}
        cache.update(reference_flux_coefficients=torch.as_tensor(snapshot['flux_coefficients'],dtype=torch.float64),
            reference_flux_level=torch.as_tensor(snapshot['target_level'],dtype=torch.float64).reshape(()))
        for name in ('volume_source_indices','volume_sampling_shape'):
            cache[name]=torch.as_tensor(snapshot[name],dtype=torch.int64)
        downstream_config=DownstreamConfig.from_mapping({**snapshot['config'],'nfp':self.nfp,
            'stokes_validation_rtol':config.stokes_validation_rtol})
        physics=stage('surface_metrics',lambda:downstream(front.geometry,front.level,cache,field,
            field.trace,field.potential,downstream_config))
        engineering=stage('coil_metrics',lambda:engineering_metrics(x,self.backend.library,self.nfp)) if _engineering else x.new_zeros(3)
        metrics=TensorMetrics(physics['raw_qs'],physics['iota_boundary'],physics['target_quality_error'],
            *engineering.unbind(-1),physics['surface_volume'],front.psi_train_rms,
            physics['raw_qs_outer'],physics['iota_min'],physics['iota_max'])
        surface=None
        if self.surface_qs is not None:
            from .._autograd.boozer import sample_boozer_surface
            from ..qs_metrics import boozer_spectrum_metrics,boozer_qs_metrics_field
            from ..qs import SurfaceQSResult
            def sample():
                fp64=CpuSegmentField(p,w,self.field_library)
                phys={**physics,'major_radius':front.geometry.axis_samples[:,0].double().mean()}
                sampled=sample_boozer_surface(front.geometry,front.level,phys,fp64,downstream_config,
                    **self.surface_qs.sampler_options(),field_derivatives=self.surface_qs.compute_f_T)
                values=boozer_qs_metrics_field(sampled['B'],sampled['dBdx'],sampled['x_theta'],sampled['x_phi'],
                    nfp=self.nfp,helicity=config.helicity,iota=sampled['iota'],G=sampled['G'],
                    major_radius=sampled['major_radius']) if self.surface_qs.compute_f_T else boozer_spectrum_metrics(
                    sampled['B_magnitude'],nfp=self.nfp,helicity=config.helicity)
                diagnostics={k:(v.detach().item() if v.numel()==1 else v.detach().tolist())
                    if isinstance(v,torch.Tensor) else v for k,v in sampled['diagnostics'].items()}
                diagnostics.update(major_radius_m=float(sampled['major_radius'].detach()),
                    iota_surface=float(sampled['iota'].detach()),G_rad=float(sampled['G'].detach()),
                    spectral_tail_fraction=float(values['spectral_tail_fraction'].detach()))
                return SurfaceQSResult(values['f_B'],values['f_T'] if self.surface_qs.compute_f_T else None,
                    dict(configuration=asdict(self.surface_qs),nfp=self.nfp,surfaces=[diagnostics],region='target_surface',
                        f_B_average='uniform_boozer_angle_energy',f_T_average='flux_shell_rms',
                        f_T_normalization='R_axis_mean^2 * T / B^4'))
            surface=stage('boozer_surface_qs',sample)
        return TensorEvaluation(x,metrics,dict(device='cpu',cpu_threads=self.threads,nfp=self.nfp,
            configuration=config.to_dict(),field_dtype=str(self.dtype),gradient_method='local_reverse',
            flux_integration=physics['flux_validation'],native_centers=[snapshot['score_result']['status']],
            capture=dict(query_count=1,unique_preparations=1,shared_geometry=False,
                native_query_timings=[snapshot['score_result'].get('timing',{})],
                native_query_diagnostics=[snapshot['score_result'].get('diagnostics',{})],
                statuses=[snapshot['score_result']['status']])),timings,surface)

    def close(self):
        if self._closed:return
        self._check();self.backend.close();self._closed=True

    def __enter__(self):
        self._check()
        self._active_thread_scope=thread_scope(self.threads)
        self._active_thread_scope.__enter__()
        return self

    def __exit__(self,*exc):
        try:
            self.close()
        finally:
            self._active_thread_scope.__exit__(*exc)
