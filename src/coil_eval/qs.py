"""Optional Boozer-surface QS configuration and value-only evaluation."""
from dataclasses import dataclass
import math
from numbers import Integral, Real
import numpy as np


@dataclass(frozen=True)
class SurfaceQSConfig:
    """Angular resolution and local inversion settings for one target surface.

    Angles use radians. ``nphi`` covers one field period; ``ntheta`` covers a
    full poloidal turn. Odd sizes avoid ambiguous Nyquist derivatives.
    """
    nphi: int = 129
    ntheta: int = 131
    nu_order: int = 24
    compute_f_T: bool = False
    alpha_order: int = 32
    alpha_ridge: float = 1e-12
    alpha_coordinates: str = 'geometric'
    alpha_fit_nphi: int | None = None
    alpha_fit_ntheta: int | None = None
    nu_method: str = 'potential'
    max_iterations: int = 30
    tolerance: float = 1e-11
    min_jacobian: float = 1e-6
    resonance_tolerance: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.compute_f_T,bool):
            raise ValueError('compute_f_T must be a boolean')
        if isinstance(self.alpha_order,(bool,np.bool_)) or not isinstance(self.alpha_order,Integral) or self.alpha_order<0:
            raise ValueError('alpha_order must be a nonnegative integer')
        object.__setattr__(self,'alpha_order',int(self.alpha_order))
        if not isinstance(self.alpha_coordinates,str) or self.alpha_coordinates not in ('straight','geometric'):
            raise ValueError('alpha_coordinates must be straight or geometric')
        if not isinstance(self.nu_method,str) or self.nu_method not in ('transport','potential'):
            raise ValueError('nu_method must be transport or potential')
        if isinstance(self.alpha_ridge,(bool,np.bool_)) or not isinstance(self.alpha_ridge,Real) or not math.isfinite(self.alpha_ridge) or self.alpha_ridge<0:
            raise ValueError('alpha_ridge must be finite and nonnegative')
        object.__setattr__(self,'alpha_ridge',float(self.alpha_ridge))
        for name in ('alpha_fit_nphi','alpha_fit_ntheta'):
            value=getattr(self,name)
            if value is not None:
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<5 or value%2!=1 or value<=2*self.alpha_order:
                    raise ValueError(f'{name} must be an odd integer resolving alpha_order')
                object.__setattr__(self,name,int(value))
        for name in ('nphi','ntheta','nu_order','max_iterations'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self,name,int(value))
        if any(n<5 or n%2!=1 for n in (self.nphi,self.ntheta)):
            raise ValueError('surface QS grids must have odd sizes at least five')
        if min(self.nphi,self.ntheta)<=2*self.nu_order:
            raise ValueError('surface QS grids must resolve all retained nu modes')
        for name in ('tolerance','min_jacobian','resonance_tolerance'):
            value=getattr(self,name)
            if (isinstance(value,bool) or not isinstance(value,Real)
                    or not math.isfinite(value) or value<=0):
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self,name,float(value))
        if self.min_jacobian>=1:
            raise ValueError('min_jacobian must be below one')

    def sampler_options(self):
        """Coordinate options shared by value and gradient evaluators."""
        from dataclasses import asdict
        options=asdict(self);options.pop('compute_f_T')
        return options

    @classmethod
    def preset(cls,accuracy='high',**overrides):
        """Select an angular accuracy/cost setting, with explicit overrides."""
        settings={'high':dict(nphi=129,ntheta=131,alpha_order=32,nu_order=24,alpha_coordinates='geometric'),
                  'balanced':dict(nphi=97,ntheta=99,alpha_order=24,nu_order=16,alpha_coordinates='geometric'),
                  'fast':dict(nphi=65,ntheta=67,alpha_order=8,nu_order=12,alpha_coordinates='straight')}
        if not isinstance(accuracy,str) or accuracy not in settings:
            raise ValueError('accuracy must be high, balanced or fast')
        return cls(**{**settings[accuracy],**overrides})


@dataclass(frozen=True)
class SurfaceQSResult:
    """Target-surface metrics; values are scalars or differentiable tensors."""
    f_B: object
    f_T: object
    diagnostics: dict


class SurfaceQSEvaluator:
    """Evaluate target-surface f_B/f_T without recording a reverse graph.

    PyTorch is loaded on construction for the selected CPU or CUDA device. Standard score calls do
    not import it. Inputs and physical units follow :class:`CoilSet`.
    """
    def __new__(cls,*args,device='cuda',threads=None,axis_grid=None,**kwargs):
        if device=='cpu':
            from ._dependencies import require_dependencies
            require_dependencies('cpu')
            from ._cpu.qs import CpuSurfaceQSEvaluator
            return CpuSurfaceQSEvaluator(*args,device=device,threads=threads,axis_grid=axis_grid,**kwargs)
        if device!='cuda':raise ValueError("device must be 'cpu' or 'cuda'")
        if threads is not None:raise ValueError("threads is only supported with device='cpu'")
        if axis_grid is not None:raise ValueError("axis_grid is only supported with device='cpu'")
        return super().__new__(cls)

    def __init__(self, library=None, field_library=None, *, config,
                 surface_qs=None, device_id=0, batch_axis=True, share_geometry=True,
                 device='cuda',threads=None,axis_grid=None):
        from ._dependencies import require_dependencies
        require_dependencies('torch')
        import torch
        from .torch import TensorEvaluator
        self._evaluator=TensorEvaluator(library,field_library,config=config,
            surface_qs=SurfaceQSConfig() if surface_qs is None else surface_qs,device_id=device_id,
            field_dtype=torch.float64,batch_axis=batch_axis,share_geometry=share_geometry)

    def evaluate(self, coils):
        """Reuse the native psi/alpha fit without constructing derivative factors."""
        import time
        import torch
        from .types import CoilSet
        engine=self._evaluator
        engine._check()
        if not isinstance(coils,CoilSet):
            raise TypeError('coils must be a CoilSet')
        from .torch import pack_coils
        parameters=pack_coils(coils,device=engine.device)
        with torch.no_grad():
            torch.cuda.synchronize(engine.device)
            start=time.perf_counter()
            source,alpha=self._capture_group(coils,[engine.config])[0]
            torch.cuda.synchronize(engine.device)
            elapsed=time.perf_counter()-start
            return self._evaluate_source(coils,engine.config,source,alpha,
                {'native_score_capture':elapsed},parameters=parameters)

    def _capture_group(self, coils, configs):
        """Capture owned source records; each group differs only in volume."""
        from .evaluator import EvaluationError, InvalidEvaluationError
        engine=self._evaluator
        if configs[0].psi_fit=='two_stage':
            from .simsopt.capture import capture_sources
            records=capture_sources(coils,configs,library=engine.library_path,
                device_id=engine.device.index,include_torsion=False)
            if len(records)!=len(configs):
                raise EvaluationError('source capture returned an incomplete query group')
            for index,record in enumerate(records):
                if not record.valid:
                    raise InvalidEvaluationError(f'query {index}: {record.result.status}',
                        results=tuple(r.result for r in records))
            return [(record.source,record.alpha) for record in records]
        import numpy as np
        from .simsopt.state import SourcePsi
        from .simsopt.clebsch import AlphaFit
        physical=np.concatenate((coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,
                                 coils.currents_a[:,None]),axis=1)
        records=[]
        for config in configs:
            snapshot=engine._capture(physical,config)
            status=snapshot.get('score_result',{}).get('status','ok')
            if status!='ok':
                raise InvalidEvaluationError(status)
            records.append((SourcePsi.from_snapshot(snapshot),AlphaFit.from_snapshot(snapshot)))
        return records

    def evaluate_many(self, coils, *, configs=None):
        """Evaluate independent coil sets, preserving input order.

        Axis preparation is batched across compatible geometries. When
        ``share_geometry=True``, identical two-stage coil inputs and settings
        differing only in target volume share their source fit. Surface QS
        sampling remains sequential per query. Results report the preparation
        groups and amortized shared times in their diagnostics.
        """
        from contextlib import nullcontext
        import json
        import time
        import numpy as np
        import torch
        from .types import CoilSet
        from .config import EvaluationConfig
        from ._tensor_config import validate_tensor_config
        from ._autograd.axis_batch import prepare_axis_batch
        engine=self._evaluator
        engine._check()
        coils=tuple(coils)
        configs=(engine.config,)*len(coils) if configs is None else tuple(configs)
        if len(configs)!=len(coils) or any(not isinstance(c,EvaluationConfig) for c in configs):
            raise ValueError('provide one EvaluationConfig per coil set')
        for coil,config in zip(coils,configs):
            if not isinstance(coil,CoilSet):
                raise TypeError('coils must contain CoilSet objects')
            if coil.nfp!=4 or coil.n_base_coils!=4:
                raise ValueError('surface QS evaluation requires nfp=4 and four basic coils')
            validate_tensor_config(config)
        if not coils:
            return ()
        physical=[np.concatenate((c.coeffs_x,c.coeffs_y,c.coeffs_z,c.currents_a[:,None]),axis=1)
                  for c in coils]
        groups=[];keys={}
        for index,(value,config) in enumerate(zip(physical,configs)):
            options=config.to_dict();options.pop('target_volume')
            key=(value.shape,value.tobytes(),json.dumps(options,sort_keys=True))
            if engine.share_geometry and config.psi_fit=='two_stage' and key in keys:
                groups[keys[key]].append(index)
            else:
                keys[key]=len(groups);groups.append([index])
        # All validated configurations share nfp, segments and fast trace
        # settings. Distinct Fourier widths need separate native batches.
        buckets={}
        for group_index,group in enumerate(groups):
            buckets.setdefault(physical[group[0]].shape,[]).append(group_index)
        axis={};axis_times={};results=[None]*len(coils)
        torch.cuda.synchronize(engine.device)
        batch_started=time.perf_counter()
        with torch.no_grad():
            for bucket in buckets.values():
                representatives=[groups[index][0] for index in bucket]
                torch.cuda.synchronize(engine.device)
                started=time.perf_counter()
                prepared=(prepare_axis_batch(np.stack([physical[i] for i in representatives]),
                    [configs[i] for i in representatives],engine.library_path,
                    device=engine.device.index,nfp=4) if engine.batch_axis else None)
                torch.cuda.synchronize(engine.device)
                elapsed=time.perf_counter()-started if prepared is not None else 0.
                requests=sum(len(groups[index]) for index in bucket)
                for position,index in enumerate(bucket):
                    axis[index]=(prepared,position)
                    axis_times[index]=elapsed/requests
            for group_index,group in enumerate(groups):
                prepared,position=axis[group_index]
                torch.cuda.synchronize(engine.device)
                started=time.perf_counter()
                try:
                    with prepared.capture(position) if prepared is not None else nullcontext():
                        captured=self._capture_group(coils[group[0]],[configs[index] for index in group])
                except Exception as exc:
                    from .evaluator import InvalidEvaluationError
                    if isinstance(exc,InvalidEvaluationError):
                        raise InvalidEvaluationError(f'source group {group}: {exc}',results=exc.results) from exc
                    raise
                torch.cuda.synchronize(engine.device)
                elapsed=time.perf_counter()-started
                if len(captured)!=len(group):
                    from .evaluator import EvaluationError
                    raise EvaluationError('source capture returned an incomplete query group')
                for index,(source,alpha) in zip(group,captured):
                    timings={'native_axis_batch':axis_times[group_index],
                             'native_score_capture':elapsed/len(group)}
                    try:
                        result=self._evaluate_source(coils[index],configs[index],source,alpha,timings)
                    except ValueError as exc:
                        raise ValueError(f'query {index}: {exc}') from exc
                    result.diagnostics['batch']={
                        'query_index':index,'query_count':len(coils),
                        'source_group_indices':list(group),'source_group_count':len(groups),
                        'shared_geometry':engine.share_geometry,
                        'axis_batch':prepared.metadata if prepared is not None else
                            {'enabled':False,'reason':'disabled_by_caller'},
                        'axis_consumption':prepared.observations()[position] if prepared is not None
                            and prepared.enabled else [],
                        'surface_execution':'sequential_per_query',
                        'timing_allocation':'axis batch divided over its requests; source capture divided over its group'}
                    results[index]=result
            torch.cuda.synchronize(engine.device)
        elapsed=time.perf_counter()-batch_started
        for result in results:
            result.diagnostics['batch']['total_wall_s']=elapsed
            result.diagnostics['batch']['amortized_wall_s']=elapsed/len(results)
        return tuple(results)

    def _evaluate_source(self, coils, query_config, source, alpha, timings, *, parameters=None):
        """Evaluate one owned source with the configuration that captured it."""
        import torch
        import time
        import numpy as np
        from dataclasses import asdict
        from ._autograd.front import PsiGeometry, native_modes
        from ._autograd.downstream import DownstreamConfig
        from ._autograd.field import SegmentField, coil_segments
        from ._autograd.boozer import sample_boozer_surface
        from .torch import pack_coils
        from .qs_metrics import boozer_spectrum_metrics,boozer_qs_metrics_field
        from .simsopt.clebsch import build_modes
        engine=self._evaluator
        if parameters is None:
            parameters=pack_coils(coils,device=engine.device)
        def timed(name,fn):
            torch.cuda.synchronize(engine.device)
            start=time.perf_counter();value=fn();torch.cuda.synchronize(engine.device)
            timings[name]=time.perf_counter()-start
            return value
        with torch.no_grad():
            def sample():
                def tensor(value):return torch.tensor(value,device=engine.device,dtype=torch.float64)
                degree=int(np.max(source.mode_a+source.mode_b));mtor=int(np.max(source.mode_m))
                modes=list(zip(source.mode_a,source.mode_b,source.mode_m,source.mode_kind))
                if modes!=native_modes(degree,mtor):
                    raise ValueError('source psi mode order differs from the native tensor basis')
                geometry=PsiGeometry(tensor(source.axis_samples),tensor(source.coefficients),
                    nfp=source.nfp,a=source.a,degree=degree,mtor=mtor)
                config=DownstreamConfig.from_mapping({**query_config.to_dict(),
                    **query_config.native_options(),'nfp':source.nfp})
                expected=build_modes(config.alpha_radial_order,config.alpha_poloidal_order,
                    config.alpha_toroidal_order)
                if (not np.array_equal(alpha.modes,np.asarray(expected)) or
                        alpha.radial_basis!=config.alpha_radial_basis or alpha.nfp!=source.nfp):
                    raise ValueError('source alpha mode order differs from the native tensor basis')
                physics={'alpha_coefficients':tensor(alpha.lambda_coefficients),
                    'iota_coefficients':tensor(alpha.iota_coefficients),
                    'major_radius':geometry.axis_samples[:,0].mean()}
                p,w=coil_segments(parameters,nfp=4,count=query_config.segments_per_coil)
                field=SegmentField(p,w,engine.field_library)
                return sample_boozer_surface(geometry,tensor(source.level),physics,field,config,
                    **engine.surface_qs.sampler_options(),field_derivatives=engine.surface_qs.compute_f_T)
            sampled=timed('boozer_surface',sample)
            def metrics():
                if engine.surface_qs.compute_f_T:
                    return boozer_qs_metrics_field(sampled['B'],sampled['dBdx'],
                        sampled['x_theta'],sampled['x_phi'],nfp=coils.nfp,
                        helicity=query_config.helicity,iota=sampled['iota'],
                        G=sampled['G'],major_radius=sampled['major_radius'])
                return boozer_spectrum_metrics(sampled['B_magnitude'],nfp=coils.nfp,
                    helicity=query_config.helicity)
            values=timed('qs_spectrum',metrics)
        diagnostics={k:v.detach().cpu().item() if isinstance(v,torch.Tensor) else v
            for k,v in sampled['diagnostics'].items()}
        diagnostics.update(major_radius_m=float(sampled['major_radius']),
            iota_surface=float(sampled['iota']),G_rad=float(sampled['G']),
            spectral_tail_fraction=float(values['spectral_tail_fraction']))
        return SurfaceQSResult(float(values['f_B']),float(values['f_T']) if 'f_T' in values else None,
            {'configuration':asdict(engine.surface_qs),'evaluation_configuration':query_config.to_dict(),
             'surfaces':[diagnostics],
             'region':'target_surface','f_B_average':'uniform_boozer_angle_energy',
             'f_T_average':'flux_shell_rms','f_T_normalization':'R_axis_mean^2 * T / B^4',
             'timings':timings})

    def close(self):self._evaluator.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
