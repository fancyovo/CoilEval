"""PyTorch evaluation of coil metrics and local reverse gradients."""
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import numpy as np
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


def pack_coils(coils: CoilSet, *, device='cuda:0', dtype=torch.float64):
    """Pack four basic coils as physical [coil, x/y/z coefficients + I]."""
    if coils.nfp != 4 or coils.coeffs_x.shape[0] != 4:
        raise ValueError('tensor evaluation requires nfp=4 and four basic coils')
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


class TensorEvaluator:
    """Single or independent-batch local response, with optional PyTorch autograd."""

    def __init__(self, library=None, field_library=None, *, config: EvaluationConfig,
                 device_id=0, field_dtype=torch.float32):
        self._validate_config(config)
        if field_dtype not in (torch.float32, torch.float64):
            raise ValueError('field_dtype must be torch.float32 or torch.float64')
        self.config = config
        self.device = torch.device('cuda', int(device_id))
        self.dtype = field_dtype
        from ._native_loader import resolve_library
        library = resolve_library(library)
        self.library_path = str(library)
        field_library = field_library or library.with_name('libcoil_eval_autograd.so')
        self.field_library = NativeFieldLibrary(field_library)
        self._thread = threading.get_ident()
        self._closed = False

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, EvaluationConfig):
            raise TypeError('config must be an EvaluationConfig')
        if not config.fast or config.flux_method != 'stokes':
            raise ValueError('tensor evaluation requires fast=True and flux_method="stokes"')
        if (config.segments_per_coil, config.psi_iterations, config.alpha_iterations,
                config.helicity) != (256, 4, 4, (1, 4)):
            raise ValueError('tensor evaluation requires 256 segments, four fit iterations and helicity (1,4)')

    def _check(self):
        if self._closed or threading.get_ident() != self._thread:
            raise CenterContextError('use an open evaluator on its creating thread')

    def _capture(self, physical, config):
        arrays = split_tokens(physical)
        if config.psi_fit == 'two_stage':
            from ._two_stage.native import TwoStageNative
            coils=CoilSet(*arrays,nfp=4)
            result,snapshot=TwoStageNative(self.library_path,self.device.index).evaluate(coils,config,capture=True)
            if snapshot is None:
                raise native.GpuError(f'two-stage center is not valid: {result["status"]}')
            return snapshot
        batch = native.BatchCoilFieldGpu(self.library_path, *(a[None] for a in arrays),
                                         nfp=4, device_id=self.device.index)
        try:
            return capture_center(batch, *arrays, target_helicity=config.helicity,
                                  config_overrides=config.native_options())
        finally:
            batch.close()

    def _center(self, snapshot):
        names = ('axis_samples', 'psi_coefficients', 'psi_scale', 'psi_upper',
                 'target_adjoint_weights', 'target_axis_weights', 'target_phi',
                 'target_axis_R', 'target_level', 'target_adjoint_residual', 'target_dVds')
        center={'config': {**snapshot['config'], 'nfp': 4}, **{
            k: torch.tensor(snapshot[k], device=self.device, dtype=torch.float64).detach()
            for k in names}}
        if 'two_stage' in snapshot:
            center['two_stage']=snapshot['two_stage']
        return center

    def evaluate(self, parameters):
        """Evaluate physical parameters of shape (4,28); metrics are scalars."""
        return self._evaluate(parameters, batch=False)

    def evaluate_batch(self, parameters, *, configs=None):
        """Evaluate independent physical parameters (B,4,28); metrics have shape (B,)."""
        return self._evaluate(parameters, batch=True, configs=configs)

    def _evaluate(self, parameters, *, batch, configs=None, _snapshots=None,
                  _local_value=False, _parameter_grad=True, _engineering=True):
        self._check()
        x = parameters if isinstance(parameters, torch.Tensor) else torch.as_tensor(
            parameters, device=self.device, dtype=torch.float64)
        wanted_ndim = 3 if batch else 2
        width=x.shape[-1] if x.ndim else 0
        if (x.device != self.device or x.ndim != wanted_ndim or x.shape[-2] != 4 or
                width<10 or (width-1)%3 or ((width-1)//3)%2!=1 or x.numel() == 0):
            raise ValueError('parameters require four coils with XYZ odd Fourier widths plus a current on the evaluator GPU')
        if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
            raise ValueError('parameters must be finite float32 or float64 physical values')
        if not _parameter_grad:
            x=x.detach()
        elif not x.requires_grad:
            x = x.detach().requires_grad_(True)
        xs = x if batch else x[None]
        configs = list(configs) if configs is not None else [self.config]*len(xs)
        if len(configs) != len(xs):
            raise ValueError('provide one configuration per geometry')
        for c in configs:
            self._validate_config(c)
        timings = {}
        def stage(name, fn):
            torch.cuda.synchronize(self.device)
            start=time.perf_counter(); result=fn(); torch.cuda.synchronize(self.device)
            timings[name]=time.perf_counter()-start
            return result
        def capture_all():
            host = xs.detach().double().cpu().numpy()
            captures = []
            for i, (value, cfg) in enumerate(zip(host, configs)):
                try:
                    snapshot = self._capture(value, cfg)
                except native.GpuError as exc:
                    raise InvalidEvaluationError(f'geometry {i}: {exc}') from exc
                if snapshot['score_result']['status'] != 'ok':
                    raise InvalidEvaluationError(f'geometry {i}: {snapshot["score_result"]["status"]}')
                captures.append(snapshot)
            return captures
        snapshots=stage('center_capture',capture_all) if _snapshots is None else list(_snapshots)
        if len(snapshots)!=len(xs):
            raise ValueError('provide one owned snapshot per query')
        centers=stage('center_upload',lambda:[self._center(s) for s in snapshots])
        def make_field():
            if batch:
                p,w=coil_segments_batch(x.double(),nfp=4,count=256)
                return BatchSegmentField(p.to(self.dtype),w.to(self.dtype),self.field_library)
            p,w=coil_segments(x.double(),nfp=4,count=256)
            return SegmentField(p.to(self.dtype),w.to(self.dtype),self.field_library)
        field=stage('coil_segments',make_field)
        def make_front():
            methods={c.psi_fit for c in configs}
            if len(methods)!=1:
                raise ValueError('group batches by psi_fit method')
            if configs[0].psi_fit!='two_stage':
                return (build_batch_front(field,centers,dtype=torch.float64,nfp=4) if batch else
                        build_front(field,field.trace,centers[0],dtype=torch.float64))
            if not batch:
                return build_two_stage_front(field,field.trace,centers[0],dtype=torch.float64,
                    local_value=_local_value,value_only=not _parameter_grad)
            return build_batch_two_stage_front(field,centers,dtype=torch.float64,
                local_value=_local_value,value_only=not _parameter_grad,nfp=4)
        front=stage('axis_psi_surface',make_front)
        downstream_configs=[DownstreamConfig.from_mapping({**s['config'],'nfp':4}) for s in snapshots]
        caches=[{name:torch.as_tensor(s[key],device=self.device,dtype=self.dtype).detach()
                 for name,key in (('upper','alpha_upper'),('scale','alpha_scale'),
                                  ('scaled_solution','alpha_center_scaled_solution'))} for s in snapshots]
        physics=stage('surface_metrics',lambda: batch_downstream(front,caches,field,downstream_configs)
                      if batch else downstream(front.geometry,front.level,caches[0],field,
                                               field.trace,field.potential,downstream_configs[0]))
        engineering=(stage('coil_metrics',lambda: torch.stack([
            _EngineeringMetrics.apply(v,self.library_path,4) for v in xs.unbind()]))
            if _engineering else xs.new_zeros((len(xs),3)))
        if not batch:
            engineering=engineering[0]
        metrics=TensorMetrics(physics['raw_qs'],physics['iota_boundary'],physics['target_quality_error'],
                              *engineering.unbind(-1),physics['surface_volume'],front.psi_train_rms,
                              physics['raw_qs_outer'],physics['iota_min'],physics['iota_max'])
        diagnostics={'configuration':[c.to_dict() for c in configs] if batch else configs[0].to_dict(),
                     'field_dtype':str(self.dtype),'psi_dtype':'torch.float64',
                     'gradient_method':'local_reverse', 'native_centers':[
                         s['score_result']['status'] for s in snapshots]}
        return TensorEvaluation(x,metrics,diagnostics,timings)

    def close(self):
        self._check();self._closed=True

    def __enter__(self):
        self._check();return self

    def __exit__(self,*exc):
        self.close()
