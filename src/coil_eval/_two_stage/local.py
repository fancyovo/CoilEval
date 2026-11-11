"""Batched local finite-difference values on a fixed discrete training branch."""
from time import perf_counter
import numpy as np


def nearby(backend,coils,candidates,center_raw,config):
    import torch
    from ..torch import TensorEvaluator,pack_coils
    from .._backend import _coil_arrays,_center_axis
    from .native import TwoStageNative
    started=perf_counter()
    driver=TwoStageNative(backend.library,backend.device_id)
    raw,snapshot=driver.evaluate(coils,config,capture=True,axis_hint=_center_axis(center_raw))
    if snapshot is None:
        raise RuntimeError(f'two-stage center capture is not valid: {raw["status"]}')
    capture_seconds=perf_counter()-started
    results=[];chunk_timings=[]
    ellipse=config.psi_fit=='near_axis_ellipse'
    mapping={'qs_volume':'qs_target_global_error_per_helicity',
        'qs_outer':'qs_target_edge_error_per_helicity','iota_boundary':'iota_target_surface',
        'iota_min':'iota_min','iota_max':'iota_max',
        'relative_normal_drift_p95':'target_quality_error','surface_volume':'surface_volume',
        'psi_fit_rms':'psi_train_rms'}
    with TensorEvaluator(backend.library,config=config,device_id=backend.device_id,nfp=coils.nfp) as evaluator:
        selection_started=perf_counter()
        choice,choice_timings=evaluator._choose_local_flux(
            pack_coils(coils,device=evaluator.device),snapshot)
        selection_seconds=perf_counter()-selection_started
        for begin in range(0,len(candidates),config.psi_fd_batch_size):
            group=candidates[begin:begin+config.psi_fd_batch_size]
            parameters=torch.stack([pack_coils(c,device=evaluator.device) for c in group])
            with torch.no_grad():
                response=evaluator._evaluate(parameters,batch=True,
                    _snapshots=[snapshot]*len(group),_local_value=True,
                    _parameter_grad=False,_engineering=False,_flux_choice=choice)
            values={name:getattr(response.metrics,name).detach().double().cpu().numpy()
                    for name in mapping}
            chunk_timings.append(response.timings)
            for index in range(len(group)):
                diagnostics=dict(raw['diagnostics'])
                diagnostics['flux_integration']=dict(response.diagnostics['flux_integration'][index])
                diagnostics.update({key:float(values[name][index]) for name,key in mapping.items()})
                volume_error=abs(diagnostics['surface_volume']/config.target_volume-1)
                valid=all(np.isfinite(values[name][index]) for name in mapping) and volume_error<=.01
                diagnostics.update(target_volume_relative_error=volume_error,
                    error_message='' if valid else 'local metric or target-volume check failed')
                if ellipse:
                    diagnostics['ellipse_neighborhood']='fixed disk nodes; moving axis and ellipse'
                else:
                    diagnostics['two_stage_neighborhood']='fixed active capacity/lattice membership; live S1, axis, domain and spacing'
                status='local_ellipse_rejected' if ellipse else 'local_two_stage_rejected'
                results.append({'status':'ok' if valid else status,
                    'components':dict(raw.get('components',{})),'diagnostics':diagnostics,'timing':{}})
            del response,parameters
    center_arrays=_coil_arrays(coils)
    arrays=tuple(np.ascontiguousarray(np.stack(items))
                 for items in zip(*[_coil_arrays(c) for c in candidates]))
    backend._add_geometry(results,center_arrays,arrays,int(coils.nfp))
    record={'scope':'shared_batch','query_count':len(candidates),'total_s':perf_counter()-started,
            'stages_s':{'center_capture':capture_seconds,'center_flux_selection':selection_seconds},
            'center_flux_choice':choice,'center_flux_selection_timings':choice_timings,'chunks':chunk_timings,
            'method':'moving_near_axis_ellipse' if ellipse else 'two_stage_fixed_discrete_branch',
            'batch_size':config.psi_fd_batch_size}
    for result in results:
        result['local_timing']=record
    return results
