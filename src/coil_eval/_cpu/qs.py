"""CPU value-only surface QS from one formal capture."""
from dataclasses import asdict
from time import perf_counter
import torch
from .tensor import CpuTensorEvaluator
from .autograd import CpuSegmentField
from .threads import thread_scope
from .._autograd.front import PsiGeometry
from .._autograd.field import coil_segments
from .._autograd.downstream import DownstreamConfig
from .._autograd.boozer import sample_boozer_surface
from ..qs_metrics import boozer_spectrum_metrics,boozer_qs_metrics_field
from ..evaluator import InvalidEvaluationError
from ..types import CoilSet
from .._tensor_config import validate_tensor_config


class CpuSurfaceQSEvaluator:
    def __init__(self,library=None,field_library=None,*,config,surface_qs=None,device_id=0,
                 batch_axis=True,share_geometry=True,device='cpu',threads=None,axis_grid=None):
        from ..qs import SurfaceQSConfig
        self._evaluator=CpuTensorEvaluator(library,field_library,config=config,
            surface_qs=SurfaceQSConfig() if surface_qs is None else surface_qs,threads=threads,axis_grid=axis_grid)

    def evaluate(self,coils):
        return self.evaluate_many([coils])[0]

    def evaluate_many(self,coils,*,configs=None):
        from ..qs import SurfaceQSResult
        from ..torch import pack_coils
        engine=self._evaluator;engine._check();coils=tuple(coils)
        configs=(engine.config,)*len(coils) if configs is None else tuple(configs)
        if len(configs)!=len(coils):raise ValueError('provide one configuration per coil set')
        for coil,config in zip(coils,configs):
            if not isinstance(coil,CoilSet):
                raise TypeError('surface QS evaluation requires CoilSet inputs')
            validate_tensor_config(config)
        results=[];batch_start=perf_counter()
        with thread_scope(engine.threads),torch.no_grad():
            for index,(coil,config) in enumerate(zip(coils,configs)):
                started=perf_counter()
                raw,snapshot=engine.backend._capture(coil,config)
                if snapshot is None:
                    raise InvalidEvaluationError(f'CPU query {index}: {raw["status"]}: {raw["diagnostics"].get("error_message","")}')
                timings=dict(native_score_capture=perf_counter()-started)
                options=snapshot['config']
                def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
                geometry=PsiGeometry(tensor(snapshot['axis_samples']),tensor(snapshot['psi_coefficients']),
                    nfp=coil.nfp,a=options['psi_a'],degree=options['psi_poly_degree'],mtor=options['psi_m_tor'])
                physics=dict(alpha_coefficients=tensor(snapshot['alpha_coefficients']),
                    iota_coefficients=tensor(snapshot['iota_coefficients']),
                    major_radius=geometry.axis_samples[:,0].mean())
                parameters=pack_coils(coil,device='cpu')
                p,w=coil_segments(parameters,nfp=coil.nfp,count=config.segments_per_coil)
                field=CpuSegmentField(p,w,engine.field_library)
                dcfg=DownstreamConfig.from_mapping(options)
                start=perf_counter()
                sample=sample_boozer_surface(geometry,tensor(snapshot['target_level']).reshape(()),physics,
                    field,dcfg,**engine.surface_qs.sampler_options(),field_derivatives=engine.surface_qs.compute_f_T)
                timings['boozer_surface']=perf_counter()-start;start=perf_counter()
                values=boozer_qs_metrics_field(sample['B'],sample['dBdx'],sample['x_theta'],sample['x_phi'],
                    nfp=coil.nfp,helicity=config.helicity,iota=sample['iota'],G=sample['G'],
                    major_radius=sample['major_radius']) if engine.surface_qs.compute_f_T else boozer_spectrum_metrics(
                    sample['B_magnitude'],nfp=coil.nfp,helicity=config.helicity)
                timings['qs_spectrum']=perf_counter()-start
                diagnostic={k:(v.item() if v.numel()==1 else v.tolist()) if isinstance(v,torch.Tensor) else v
                            for k,v in sample['diagnostics'].items()}
                diagnostic.update(iota_surface=float(sample['iota']),G_rad=float(sample['G']),
                    major_radius_m=float(sample['major_radius']),spectral_tail_fraction=float(values['spectral_tail_fraction']))
                results.append(SurfaceQSResult(float(values['f_B']),float(values['f_T']) if 'f_T' in values else None,
                    dict(configuration=asdict(engine.surface_qs),evaluation_configuration=config.to_dict(),
                        device='cpu',cpu_threads=engine.threads,nfp=coil.nfp,surfaces=[diagnostic],timings=timings,
                        capture=raw['timing'],region='target_surface',f_B_average='uniform_boozer_angle_energy',
                        f_T_average='flux_shell_rms',f_T_normalization='R_axis_mean^2 * T / B^4')))
        elapsed=perf_counter()-batch_start
        for i,result in enumerate(results):result.diagnostics['batch']=dict(query_index=i,query_count=len(results),
            total_wall_s=elapsed,amortized_wall_s=elapsed/len(results),surface_execution='sequential_samples_parallel_cpu_kernels')
        return tuple(results)

    def close(self):self._evaluator.close()
    def __enter__(self):self._evaluator._check();return self
    def __exit__(self,*exc):self.close()
