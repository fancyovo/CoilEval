"""PyTorch evaluation of coil metrics and local reverse gradients."""
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
import json
import numpy as np
from ._dependencies import require_dependencies, require_torch_cuda
require_dependencies('torch')
import torch

from . import _binding as native
from .config import EvaluationConfig
from .types import CoilSet
from .evaluator import CenterContextError, InvalidEvaluationError
from ._autograd.capture import capture_center
from ._autograd.field import NativeFieldLibrary, SegmentField, coil_segments
from ._autograd.front import build_front
from ._autograd.downstream import DownstreamConfig, downstream
from ._autograd.batch_field import BatchSegmentField, coil_segments_batch
from ._autograd.batch_front import build_batch_front
from ._autograd.two_stage_front import build_two_stage_front,build_batch_two_stage_front
from ._autograd.batch_downstream import batch_downstream
from ._autograd.engineering import _EngineeringMetrics, split_tokens
from ._tensor_config import validate_tensor_config, validate_tensor_batch_configs
from ._autograd.axis_batch import prepare_axis_batch


def pack_coils(coils: CoilSet, *, device='cuda:0', dtype=torch.float64):
    """Pack basic coils as physical [coil, x/y/z coefficients + I].

    The tensor contains no symmetry metadata; pass coils.nfp to TensorEvaluator.
    """
    if not isinstance(coils, CoilSet):
        raise TypeError('coils must be a CoilSet')
    if device is not None:
        device = torch.device(device)
        if device.type == 'cuda':
            require_torch_cuda(torch, device)
    values = np.concatenate((coils.coeffs_x, coils.coeffs_y, coils.coeffs_z,
                             coils.currents_a[:, None]), axis=1)
    return torch.tensor(values, device=device, dtype=dtype)


@dataclass(frozen=True)
class TensorMetrics:
    qs_volume: torch.Tensor
    iota_boundary: torch.Tensor
    relative_normal_drift_p95: torch.Tensor
    coil_gap_min: torch.Tensor
    coil_curvature_max: torch.Tensor
    coil_length_mean: torch.Tensor
    surface_volume: torch.Tensor
    psi_fit_rms: torch.Tensor
    qs_outer: torch.Tensor | None = None
    iota_min: torch.Tensor | None = None
    iota_max: torch.Tensor | None = None


@dataclass(frozen=True)
class TensorEvaluation:
    parameters: torch.Tensor
    metrics: TensorMetrics
    diagnostics: dict
    timings: dict
    surface_qs: object | None = None


class TensorEvaluator:
    """Single or independent-batch local response, with optional PyTorch autograd."""

    def __new__(cls,*args,device='cuda',threads=None,axis_grid=None,**kwargs):
        if device=='cpu':
            require_dependencies('cpu')
            from ._cpu.tensor import CpuTensorEvaluator
            return CpuTensorEvaluator(*args,device=device,threads=threads,axis_grid=axis_grid,**kwargs)
        if device!='cuda':
            raise ValueError("device must be 'cpu' or 'cuda'")
        if threads is not None:
            raise ValueError("threads is only supported with device='cpu'")
        if axis_grid is not None:
            raise ValueError("axis_grid is only supported with device='cpu'")
        return super().__new__(cls)

    def __init__(self, library=None, field_library=None, *, config: EvaluationConfig,
                 device_id=0, field_dtype=torch.float32, batch_axis=True, share_geometry=True,
                 surface_qs=None,device='cuda',threads=None,axis_grid=None,
                 capture_pipeline=True, batch_surface_qs=True, synchronize_stages=False,
                 engineering_overlap=True, capture_workers=1, nfp=4):
        self._validate_config(config)
        if field_dtype not in (torch.float32, torch.float64):
            raise ValueError('field_dtype must be torch.float32 or torch.float64')
        from operator import index
        if isinstance(nfp, (bool, np.bool_)):
            raise ValueError('nfp must be a positive integer')
        try:
            self.nfp = index(nfp)
        except TypeError as exc:
            raise ValueError('nfp must be a positive integer') from exc
        if self.nfp < 1:
            raise ValueError('nfp must be a positive integer')
        self.config = config
        self.device = torch.device('cuda', int(device_id))
        self.dtype = field_dtype
        self.batch_axis = bool(batch_axis)
        self.share_geometry = bool(share_geometry)
        from operator import index
        if isinstance(capture_workers, bool):
            raise ValueError('capture_workers must be an integer between 1 and 4')
        try:
            self.capture_workers = index(capture_workers)
        except TypeError as exc:
            raise ValueError('capture_workers must be an integer between 1 and 4') from exc
        if not 1 <= self.capture_workers <= 4:
            raise ValueError('capture_workers must be an integer between 1 and 4')
        self.capture_pipeline = bool(capture_pipeline)
        self.batch_surface_qs = bool(batch_surface_qs)
        self.synchronize_stages = bool(synchronize_stages)
        self.engineering_overlap = bool(engineering_overlap)
        from .qs import SurfaceQSConfig
        if surface_qs is not None and not isinstance(surface_qs,SurfaceQSConfig):
            raise TypeError('surface_qs must be SurfaceQSConfig or None')
        self.surface_qs = surface_qs
        from ._native_loader import resolve_library
        library = resolve_library(library)
        self.library_path = str(library)
        field_library = field_library or library.with_name('libcoil_eval_autograd.so')
        self.field_library = NativeFieldLibrary(field_library)
        self._thread = threading.get_ident()
        self._closed = False

    @staticmethod
    def _validate_config(config):
        validate_tensor_config(config)

    def _check(self):
        if self._closed or threading.get_ident() != self._thread:
            raise CenterContextError('use an open evaluator on its creating thread')

    def _capture(self, physical, config, *, nfp=None):
        nfp = self.nfp if nfp is None else nfp
        arrays = split_tokens(physical)
        if config.psi_fit == 'two_stage':
            from ._two_stage.native import TwoStageNative
            coils=CoilSet(*arrays,nfp=nfp)
            result,snapshot=TwoStageNative(self.library_path,self.device.index).evaluate(
                coils,config,capture=True,include_torsion=False)
            if snapshot is None:
                raise native.GpuError(f'two-stage center is not valid: {result["status"]}')
            return snapshot
        batch = native.BatchCoilFieldGpu(self.library_path, *(a[None] for a in arrays),
                                         nfp=nfp, device_id=self.device.index)
        try:
            return capture_center(batch, *arrays, target_helicity=config.helicity,
                                  config_overrides=config.native_options(),include_torsion=False)
        finally:
            batch.close()

    def _center(self, snapshot):
        for source in (snapshot, snapshot['config']):
            if 'nfp' in source:
                captured_nfp = np.asarray(source['nfp'])
                if captured_nfp.size != 1 or captured_nfp.item() != self.nfp:
                    raise ValueError('snapshot field periods do not match evaluator nfp')
        names = ('axis_samples', 'psi_coefficients', 'psi_scale', 'psi_upper',
                 'target_adjoint_weights', 'target_axis_weights', 'target_phi',
                 'target_axis_R', 'target_level', 'target_adjoint_residual', 'target_dVds')
        center={'config': {**snapshot['config'], 'nfp': self.nfp}, **{
            k: torch.tensor(snapshot[k], device=self.device, dtype=torch.float64).detach()
            for k in names}}
        if 'two_stage' in snapshot:
            center['two_stage']=snapshot['two_stage']
        return center

    def evaluate(self, parameters):
        """Evaluate physical parameters (nc, 3*ncoeff+1); metrics are scalars."""
        return self._evaluate(parameters, batch=False)

    def evaluate_batch(self, parameters, *, configs=None):
        """Evaluate physical parameters (B,nc,3*ncoeff+1); metrics have shape (B,)."""
        return self._evaluate(parameters, batch=True, configs=configs)

    def _choose_local_flux(self, parameters, snapshot):
        """Choose one center quadrature before evaluating finite perturbations.

        The center runs through its ordinary front response and flux check.
        Alpha/QS, surface quality and geometry metrics are not needed here.
        """
        if self.config.stokes_validation_rtol is None:
            return dict(method='stokes', validation='disabled', fallback=False,
                tolerance=None, edge_relative_error=None, derivative_relative_error=None), {}
        with torch.no_grad():
            return self._evaluate(parameters, batch=False, _snapshots=[snapshot],
                                  _parameter_grad=False, _engineering=False,
                                  _flux_choice_only=True)

    def _evaluate(self, parameters, *, batch, configs=None, _snapshots=None,
                  _local_value=False, _parameter_grad=True, _engineering=True,
                  _flux_choice_only=False, _flux_choice=None):
        self._check()
        if _flux_choice_only and (batch or _snapshots is None or _flux_choice is not None):
            raise ValueError('center flux selection requires one captured scalar center')
        if _flux_choice is not None:
            if _snapshots is None or not _local_value or _flux_choice.get('method') not in {'stokes','area'}:
                raise ValueError('a frozen flux choice requires captured local endpoints and an integration method')
        if isinstance(parameters, torch.Tensor):
            x = parameters
        else:
            if self.device.type == 'cuda':
                require_torch_cuda(torch, self.device)
            x = torch.as_tensor(parameters, device=self.device, dtype=torch.float64)
        if x.device != self.device:
            raise ValueError(
                f'parameters are on {x.device}, but this evaluator uses {self.device}. '
                'Match pack_coils(device=...) to the evaluator device_id.'
            )
        wanted_ndim = 3 if batch else 2
        width=x.shape[-1] if x.ndim else 0
        if (x.ndim != wanted_ndim or x.shape[-2] < 1 or
                width<10 or (width-1)%3 or ((width-1)//3)%2!=1 or x.numel() == 0):
            raise ValueError('parameters require nonempty coils with XYZ odd Fourier widths plus a current')
        if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
            raise ValueError('parameters must be finite float32 or float64 physical values')
        if not _parameter_grad:
            x=x.detach()
        elif not x.requires_grad:
            x = x.detach().requires_grad_(True)
        xs = x if batch else x[None]
        configs = validate_tensor_batch_configs(
            configs if configs is not None else [self.config]*len(xs),count=len(xs))
        timings = {}
        stage_events = []
        synchronized = getattr(self, 'synchronize_stages', True)
        def stage(name, fn):
            if synchronized:
                torch.cuda.synchronize(self.device)
                start=time.perf_counter(); result=fn(); torch.cuda.synchronize(self.device)
                timings[name]=time.perf_counter()-start
            else:
                # Keep the launch stream moving. Events are resolved together
                # at the public-call boundary, never between adjacent stages.
                stream = torch.cuda.current_stream(self.device)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                result = fn()
                end.record(stream)
                stage_events.append((name, begin, end))
            return result
        def finish_timings():
            if stage_events:
                stage_events[-1][2].synchronize()
                for name, begin, end in stage_events:
                    timings[name] = begin.elapsed_time(end) / 1000.
        captured_host = []
        def capture_all():
            from ._autograd.cuda_memory import prepare_native_memory
            capture_metadata['allocator_handoff']=prepare_native_memory(self.device)
            host = xs.detach().double().cpu().numpy()
            captured_host.append(host)
            # Only exact physical duplicates and configurations differing solely
            # in target volume share S1/domain/S2 work. Graph parameters remain
            # independent per query; sharing applies to constant host captures.
            groups = []
            keys = {}
            for i, (value,cfg) in enumerate(zip(host,configs)):
                options=cfg.to_dict();options.pop('target_volume')
                key=(value.tobytes(),json.dumps(options,sort_keys=True))
                if self.share_geometry and cfg.psi_fit=='two_stage' and key in keys:
                    groups[keys[key]].append(i)
                else:
                    keys[key]=len(groups);groups.append([i])
            representatives=[g[0] for g in groups]
            prepared=(prepare_axis_batch(host[representatives],[configs[i] for i in representatives],
                self.library_path,device=self.device.index,nfp=self.nfp) if self.batch_axis else None)
            captures = [None]*len(host)
            if ((getattr(self, 'capture_pipeline', False) or getattr(self, 'capture_workers', 1)>1) and len(groups)>1
                    and all(c.psi_fit == 'two_stage' for c in configs)):
                from ._two_stage.capture_batch import capture_two_stage_batch, CapturePipelineError
                try:
                    if getattr(self, 'capture_workers', 1)>1:
                        from ._two_stage.capture_batch import capture_two_stage_workers
                        pairs, pipeline = capture_two_stage_workers(host, configs, groups,
                            self.library_path, device_id=self.device.index, prepared_axis=prepared,
                            workers=self.capture_workers,nfp=self.nfp)
                    else:
                        pairs, pipeline = capture_two_stage_batch(host, configs, groups,
                            self.library_path, device_id=self.device.index, prepared_axis=prepared,nfp=self.nfp)
                except CapturePipelineError as exc:
                    raise InvalidEvaluationError(str(exc)) from exc
                except native.GpuError as exc:
                    raise InvalidEvaluationError(f'batched center capture: {exc}') from exc
                for i, (raw, snapshot) in enumerate(pairs):
                    if snapshot is None or raw['status'] != 'ok':
                        raise InvalidEvaluationError(f'geometry {i}: {raw["status"]}')
                    captures[i] = snapshot
                capture_metadata['pipeline'] = pipeline
            from contextlib import nullcontext
            for j,group in enumerate(groups if captures[0] is None else []):
                i=group[0]
                try:
                    with prepared.capture(j) if prepared is not None else nullcontext():
                        if len(group)>1:
                            from ._two_stage.native import TwoStageNative
                            coils=CoilSet(*split_tokens(host[i]),nfp=self.nfp)
                            results=TwoStageNative(self.library_path,self.device.index).evaluate_many(
                                coils,[configs[k] for k in group],capture=True,include_torsion=False)
                            if len(results)!=len(group):
                                raise InvalidEvaluationError('two-stage capture returned an incomplete query group')
                            for k,(raw,snapshot) in zip(group,results):
                                if snapshot is None or raw['status']!='ok':
                                    raise InvalidEvaluationError(f'geometry {k}: {raw["status"]}')
                                captures[k]=snapshot
                        else:
                            captures[i]=self._capture(host[i],configs[i])
                except native.GpuError as exc:
                    raise InvalidEvaluationError(f'geometry {i}: {exc}') from exc
                for k in group:
                    if captures[k]['score_result']['status'] != 'ok':
                        raise InvalidEvaluationError(f'geometry {k}: {captures[k]["score_result"]["status"]}')
            capture_metadata.update(query_count=len(host),unique_preparations=len(groups),
                shared_geometry=self.share_geometry,groups=groups,
                axis_batch=prepared.metadata if prepared is not None else {'enabled':False,'reason':'disabled_by_caller'},
                axis_consumption=prepared.observations() if prepared is not None else [],
                native_query_timings=[s['score_result'].get('timing',{}) for s in captures])
            return captures
        capture_metadata={}
        snapshots=stage('center_capture',capture_all) if _snapshots is None else list(_snapshots)
        if len(snapshots)!=len(xs):
            raise ValueError('provide one owned snapshot per query')
        from contextlib import nullcontext
        from ._autograd.engineering import prepare_engineering_batch
        overlap = _engineering and getattr(self, 'engineering_overlap', False)
        engineering_context = (prepare_engineering_batch(
            captured_host[0] if captured_host else xs.detach().double().cpu().numpy(),
            self.library_path, self.nfp) if overlap else nullcontext(None))
        with engineering_context as engineering_work:
            centers=stage('center_upload',lambda:[self._center(s) for s in snapshots])
            def make_field():
                if batch:
                    p,w=coil_segments_batch(x.double(),nfp=self.nfp,count=256)
                    return BatchSegmentField(p.to(self.dtype),w.to(self.dtype),self.field_library)
                p,w=coil_segments(x.double(),nfp=self.nfp,count=256)
                return SegmentField(p.to(self.dtype),w.to(self.dtype),self.field_library)
            field=stage('coil_segments',make_field)
            def make_front():
                methods={c.psi_fit for c in configs}
                if len(methods)!=1:
                    raise ValueError('group batches by psi_fit method')
                if configs[0].psi_fit!='two_stage':
                    return (build_batch_front(field,centers,dtype=torch.float64,nfp=self.nfp) if batch else
                            build_front(field,field.trace,centers[0],dtype=torch.float64))
                if not batch:
                    return build_two_stage_front(field,field.trace,centers[0],dtype=torch.float64,
                        local_value=_local_value,value_only=not _parameter_grad)
                return build_batch_two_stage_front(field,centers,dtype=torch.float64,
                    local_value=_local_value,value_only=not _parameter_grad,nfp=self.nfp)
            front=stage('axis_psi_surface',make_front)
            downstream_configs=[DownstreamConfig.from_mapping({**s['config'],'nfp':self.nfp,
                'stokes_validation_rtol':cfg.stokes_validation_rtol})
                for s,cfg in zip(snapshots,configs)]
            caches=[{name:torch.as_tensor(s[key],device=self.device,dtype=self.dtype).detach()
                     for name,key in (('upper','alpha_upper'),('scale','alpha_scale'),
                                      ('scaled_solution','alpha_center_scaled_solution'))} for s in snapshots]
            for cache,snapshot,cfg in zip(caches,snapshots,configs):
                if 'flux_coefficients' in snapshot and 'target_level' in snapshot:
                    cache['reference_flux_coefficients']=torch.tensor(
                        snapshot['flux_coefficients'],device=self.device,dtype=torch.float64)
                    cache['reference_flux_level']=torch.tensor(
                        snapshot['target_level'],device=self.device,dtype=torch.float64).reshape(())
                elif cfg.stokes_validation_rtol is not None:
                    raise InvalidEvaluationError('formal center lacks the flux reference required for Stokes validation')
                for name in ('volume_source_indices','volume_sampling_shape'):
                    if name in snapshot:
                        cache[name]=torch.tensor(snapshot[name],device=self.device,dtype=torch.int64)
            if _flux_choice_only:
                from ._autograd.downstream import flux_calibration
                from ._autograd.flux_validation import stokes_check
                flux=stage('flux_choice',lambda:flux_calibration(front.geometry,front.level,
                    field,field.potential,downstream_configs[0]))
                _,choice=stokes_check(flux,front.level,caches[0],configs[0].stokes_validation_rtol)
                finish_timings()
                return choice,timings
            if _flux_choice is not None:
                downstream_configs=[replace(c,stokes_flux=_flux_choice['method']=='stokes',
                                            stokes_validation_rtol=None) for c in downstream_configs]
            physics=stage('surface_metrics',lambda: batch_downstream(front,caches,field,downstream_configs)
                          if batch else downstream(front.geometry,front.level,caches[0],field,
                                                   field.trace,field.potential,downstream_configs[0]))
            if _flux_choice is not None:
                selected={**_flux_choice,'validation':'center_selected',
                          'center_validation':_flux_choice['validation']}
                physics['flux_validation']=([dict(selected) for _ in configs] if batch else selected)
            qs_result=None
            if getattr(self,'surface_qs',None) is not None:
                from dataclasses import asdict
                from ._autograd.boozer import sample_boozer_surface
                from .qs_metrics import boozer_spectrum_metrics,boozer_qs_metrics_field
                from .qs import SurfaceQSResult
                def evaluate_qs_surfaces():
                    # Spectral second derivatives amplify FP32 sampling noise.
                    # Build these field quadratures in FP64 while preserving the
                    # configured precision of the existing score and its gradient.
                    qp,qw=(coil_segments_batch(x.double(),nfp=self.nfp,count=256) if batch
                           else coil_segments(x.double(),nfp=self.nfp,count=256))
                    geometries = [front.geometry(i) if batch else front.geometry
                                  for i in range(len(configs))]
                    levels = list(front.levels.unbind()) if batch else [front.level]
                    physics_queries = []
                    for geometry, i in zip(geometries, range(len(configs))):
                        phys = {k:v[i] if batch and isinstance(v,torch.Tensor) else v
                                for k,v in physics.items()}
                        phys['major_radius'] = geometry.axis_samples[:,0].double().mean()
                        physics_queries.append(phys)
                    if batch and getattr(self, 'batch_surface_qs', False):
                        from ._autograd.batch_surface_qs import sample_boozer_surfaces
                        samples = sample_boozer_surfaces(geometries, levels, physics_queries,
                            BatchSegmentField(qp, qw, self.field_library), downstream_configs,
                            **self.surface_qs.sampler_options(),
                            field_derivatives=self.surface_qs.compute_f_T)
                    else:
                        samples = [sample_boozer_surface(geometry, level, phys,
                            SegmentField(qp[i] if batch else qp, qw[i] if batch else qw,
                                         self.field_library), downstream_configs[i],
                            **self.surface_qs.sampler_options(),
                            field_derivatives=self.surface_qs.compute_f_T)
                            for i, (geometry, level, phys) in enumerate(
                                zip(geometries, levels, physics_queries))]
                    records=[]
                    for i, (cfg, sample) in enumerate(zip(configs, samples)):
                        values=(boozer_qs_metrics_field(sample['B'],sample['dBdx'],
                            sample['x_theta'],sample['x_phi'],nfp=self.nfp,helicity=cfg.helicity,
                            iota=sample['iota'],G=sample['G'],major_radius=sample['major_radius'])
                            if self.surface_qs.compute_f_T else boozer_spectrum_metrics(
                                sample['B_magnitude'],nfp=self.nfp,helicity=cfg.helicity))
                        clean_diagnostics={k:(v.detach().cpu().item() if v.numel()==1
                            else v.detach().cpu().tolist()) if isinstance(v,torch.Tensor) else v
                            for k,v in sample['diagnostics'].items()}
                        records.append((values,dict(clean_diagnostics,
                            major_radius_m=float(sample['major_radius'].detach()),
                            iota_surface=float(sample['iota'].detach()),G_rad=float(sample['G'].detach()),
                            spectral_tail_fraction=float(values['spectral_tail_fraction'].detach()))))
                    return records
                records=stage('boozer_surface_qs',evaluate_qs_surfaces)
                qs_result=SurfaceQSResult(
                    torch.stack([v['f_B'] for v,d in records]) if batch else records[0][0]['f_B'],
                    (torch.stack([v['f_T'] for v,d in records]) if batch else records[0][0]['f_T'])
                        if self.surface_qs.compute_f_T else None,
                    {'configuration':asdict(self.surface_qs),'surfaces':[d for v,d in records],
                     'region':'target_surface','f_B_average':'uniform_boozer_angle_energy',
                     'f_T_average':'flux_shell_rms','f_T_normalization':'R_axis_mean^2 * T / B^4'})
            # The host metric worker is independent of optional target-surface
            # sampling. Join it after QS work so its result does not leave the
            # GPU idle while the surface sampler is still running.
            engineering=(stage('coil_metrics',lambda: engineering_work.tensor(xs)
                if engineering_work is not None else torch.stack([
                    _EngineeringMetrics.apply(v,self.library_path,self.nfp) for v in xs.unbind()]))
                if _engineering else xs.new_zeros((len(xs),3)))
            if not batch:
                engineering=engineering[0]
            metrics=TensorMetrics(physics['raw_qs'],physics['iota_boundary'],physics['target_quality_error'],
                                  *engineering.unbind(-1),physics['surface_volume'],front.psi_train_rms,
                                  physics['raw_qs_outer'],physics['iota_min'],physics['iota_max'])
            diagnostics={'nfp':self.nfp,'n_base_coils':x.shape[-2],'configuration':[c.to_dict() for c in configs] if batch else configs[0].to_dict(),
                         'field_dtype':str(self.dtype),'psi_dtype':'torch.float64',
                         'gradient_method':'local_reverse', 'native_centers':[
                             s['score_result']['status'] for s in snapshots],
                         'volume_sampling_source':[
                             'formal_capture' if 'volume_source_indices' in s else 'legacy_local'
                             for s in snapshots],
                         'flux_integration':physics['flux_validation'],
                         'capture':capture_metadata,
                         'engineering_overlap': engineering_work is not None,
                         'stage_timing': 'synchronized_wall' if synchronized else 'deferred_cuda_events'}
            finish_timings()
            return TensorEvaluation(x,metrics,diagnostics,timings,qs_result)

    def close(self):
        if threading.get_ident()!=self._thread:
            raise CenterContextError('close the evaluator on its creating thread')
        self._closed=True

    def __enter__(self):
        self._check();return self

    def __exit__(self,*exc):
        self.close()
