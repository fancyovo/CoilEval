"""Full CPU evaluator: native host policy, CPU quadratures and QR fits."""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
from operator import index
import numpy as np
import torch
from .native import CpuLibrary, CpuField, pointer, SGPU_SCORE_TIMING_NAMES
from .threads import available_threads, thread_scope
from .fitting import psi_grid, fit_psi, fit_alpha
from .geometry import target_surface, formal_volume_points
from .._autograd.front import PsiGeometry
from .._autograd.downstream import (DownstreamConfig,flux_calibration,vacuum_G,
    qs_metric,surface_quality)


class CpuBackend:
    def __init__(self,library=None,*,threads=None,axis_grid=None):
        self.threads=available_threads(threads)
        if axis_grid is not None:
            try:
                if isinstance(axis_grid,bool) or index(axis_grid)<4:raise ValueError
            except (TypeError,ValueError) as exc:
                raise ValueError('axis_grid must be an integer >=4 or None') from exc
            axis_grid=index(axis_grid)
        self.axis_grid=axis_grid
        self.library=CpuLibrary(library)

    def close(self):
        pass

    def evaluate(self,coils,config):
        return self.capture(coils,config)[0]

    def evaluate_many(self,coils,config,configs=None):
        # Each sample owns the full CPU allocation; no nested oversubscription.
        configs=[config]*len(coils) if configs is None else configs
        start=perf_counter();records=[];times=[]
        for coil,cfg in zip(coils,configs):
            t=perf_counter();records.append(self.evaluate(coil,cfg));times.append(perf_counter()-t)
        timing=dict(query_s=times,total_s=perf_counter()-start,query_count=len(records),
                    execution='sequential_samples_parallel_cpu_kernels',threads=self.threads)
        for raw in records:raw['independent_batch_timing']=timing
        return records

    def nearby(self,coils,candidates,raw,config):
        # CPU finite differences currently use complete formal reevaluation.
        # This preserves all physical responses, including magnetic-axis motion.
        records=[]
        diagnostics=raw.get('diagnostics',{})
        hint=(diagnostics['axis_R'],diagnostics['axis_Z']) if 'axis_R' in diagnostics else None
        for candidate in candidates:
            result=self.capture(candidate,config,axis_hint=hint)[0]
            result['diagnostics'].update(geometry_model='formal_cpu',nearby_method='formal',nearby_backend='cpu',
                nearby_fallback_reason='CPU finite differences rebuild the formal fit')
            records.append(result)
        return records

    def capture(self,coils,config,*,axis_hint=None):
        with thread_scope(self.threads):
            return self._capture(coils,config,axis_hint=axis_hint)

    def _capture(self,coils,config,*,axis_hint=None):
        start=perf_counter();timing={};diagnostics=dict(device='cpu',cpu_threads=self.threads,
            cpu_openmp=self.library.openmp_enabled,
            cpu_avx2=self.library.avx2_enabled,
            cpu_field_threads=self.threads if self.library.openmp_enabled else 1)
        cfg=self.library.configuration(config,coils.nfp)
        if self.axis_grid is not None:cfg.axis_grid=self.axis_grid
        if axis_hint is not None:
            cfg.axis_hint_enabled=1;cfg.axis_hint_require_continuation=2
            cfg.axis_hint_R,cfg.axis_hint_Z=map(float,axis_hint)
        options={name:getattr(cfg,name) for name,_ in cfg._fields_}
        options.update(nfp=coils.nfp,psi_iterations=config.psi_iterations,
            alpha_iterations=config.alpha_iterations,alpha_radial_weighting=config.alpha_radial_weighting,
            alpha_weight_sigma=config.alpha_weight_sigma,alpha_count_floor=config.alpha_count_floor,
            alpha_radial_basis=config.alpha_radial_basis)
        options.update(psi_target_refit=config.psi_target_refit,
            psi_target_refit_factor=config.psi_target_refit_factor,
            psi_target_refit_ridge=config.psi_target_refit_ridge)
        if config.psi_refit_ridge is not None:
            options['psi_refit_ridge']=config.psi_refit_ridge
        field=CpuField(coils,cfg.segments_per_coil,self.threads,self.library)
        def stage(name,fn):
            t=perf_counter();out=fn();timing[name]=perf_counter()-t;return out
        status='internal_error'
        try:
            axis,axis_info,axis_times=stage('axis_search',lambda:field.axis(cfg));diagnostics.update(axis_info)
            detailed={name:float(value) for name,value in zip(SGPU_SCORE_TIMING_NAMES,axis_times)
                      if name.startswith('axis_')} if axis_times is not None else {}
            diagnostics.update(axis_grid=cfg.axis_grid,
                axis_seed_count=cfg.axis_grid**2 if detailed.get('axis_primary_grid_trace_s',0)>0 else 0,
                axis_fallback_grid=cfg.axis_fallback_grid,
                axis_fallback_seed_count=cfg.axis_fallback_grid**2 if detailed.get('axis_fallback_grid_trace_s',0)>0 else 0,
                axis_fallback_used=detailed.get('axis_fallback_grid_trace_s',0)>0,axis_timings=detailed)
            if axis is None:
                timing['total']=perf_counter()-start
                return dict(status=axis_info.get('status','no_axis'),diagnostics=diagnostics,timing=timing),None
            rz,phi=stage('psi_grid',lambda:psi_grid(axis,cfg,coils.nfp))
            coefficients,factor=stage('psi_fit',lambda:fit_psi(field,axis,rz,phi,cfg,coils.nfp))
            two=None
            if config.psi_fit=='two_stage':
                from .._two_stage.model import PsiModel
                from .._two_stage.boundary import max_closed_boundary
                from .._two_stage.sampling import sample_frozen_sections
                from .._two_stage.quadrature import sample_polar_quadrature
                model=PsiModel(coefficients,axis,nfp=coils.nfp,a=cfg.psi_a,
                    degree=cfg.psi_poly_degree,mtor=cfg.psi_m_tor)
                phis=np.arange(cfg.psi_n_phi)*2*np.pi/(coils.nfp*cfg.psi_n_phi)
                status='no_surface'
                boundary=stage('refit_boundary',lambda:max_closed_boundary(model,phis,ntheta=256,max_radius=cfg.psi_a))
                sample=stage('refit_points',lambda:sample_polar_quadrature(boundary,
                    target_count=config.psi_refit_points,nfp=coils.nfp) if config.psi_refit_sampling=='polar_quadrature'
                    else sample_frozen_sections(boundary['polygons_rz'],phis,
                        target_count=config.psi_refit_points,nfp=coils.nfp))
                two=dict(stage1_coefficients=coefficients,stage1_scale=factor['scale'],
                    stage1_upper=factor['upper'],boundary=boundary,sampling=sample)
                refit_cfg=type(cfg).from_buffer_copy(bytes(cfg))
                if config.psi_refit_ridge is not None:
                    refit_cfg.psi_ridge=config.psi_refit_ridge
                coefficients,factor=stage('psi_refit',lambda:fit_psi(field,axis,sample['rz'],sample['phi'],
                    refit_cfg,coils.nfp,sample.get('row_weights')))
                diagnostics.update(psi_refit_domain_volume=boundary['volume_m3'],
                    psi_refit_training_count=len(sample['rz']),psi_refit_ridge=refit_cfg.psi_ridge)
                if config.psi_target_refit:
                    from .._two_stage.target_domain import target_training_domain
                    model=PsiModel(coefficients,axis,nfp=coils.nfp,a=cfg.psi_a,
                        degree=cfg.psi_poly_degree,mtor=cfg.psi_m_tor)
                    target_metadata=stage('target_training_domain',lambda:target_training_domain(
                        model,phis,config.target_volume,factor=config.psi_target_refit_factor,
                        target_count=config.psi_refit_points))
                    target_metadata.update(stage2_coefficients=coefficients,
                        stage2_scale=factor['scale'],stage2_upper=factor['upper'])
                    sample=target_metadata['sampling']
                    refit_cfg.psi_ridge=config.psi_target_refit_ridge
                    coefficients,factor=stage('psi_target_refit',lambda:fit_psi(field,axis,
                        sample['rz'],sample['phi'],refit_cfg,coils.nfp,sample['row_weights']))
                    two['target_refit']=target_metadata
                    diagnostics['psi_target_refit_domain']=dict(
                        target_metadata['boundary']['diagnostics'],training_count=len(sample['rz']))
            diagnostics['psi_fit_stages']=3 if config.psi_target_refit else 2 if two is not None else 1
            geometry=PsiGeometry(torch.from_numpy(axis),torch.from_numpy(coefficients),coils.nfp,
                cfg.psi_a,cfg.psi_poly_degree,cfg.psi_m_tor,
                maximum_radius_scale=cfg.surface_max_radius_scale,root_iterations=cfg.surface_newton_iters,
                root_tolerance=cfg.surface_newton_tolerance)
            status='no_surface'
            level,target,target_info=stage('target_volume',lambda:target_surface(geometry,cfg,self.library))
            diagnostics.update(target_info)
            dcfg=DownstreamConfig.from_mapping({**options,'stokes_flux':False})
            status='flux_rejected'
            flux=stage('flux',lambda:flux_calibration(geometry,level,field,field.potential,dcfg))
            volume_error=float(flux['surface_volume'])/config.target_volume-1
            diagnostics['target_volume_relative_error']=volume_error
            if not np.isfinite(volume_error) or abs(volume_error)>cfg.target_volume_forward_tolerance:
                raise ValueError('target-volume forward precision tolerance exceeded')
            from .._autograd.downstream import polynomial
            fs=torch.arange(201,dtype=torch.float64)*level/200
            derivative=polynomial(flux['coefficients']*torch.arange(1,len(flux['coefficients'])+1),fs,constant=True)
            if not bool((derivative*flux['edge_flux'].sign()>0).all()):
                raise ValueError('calibrated flux derivative is not monotone over the fitted domain')
            points,selection,sampling_info=stage('volume_points',lambda:formal_volume_points(geometry,level,flux,dcfg,cfg))
            diagnostics.update(sampling_info)
            B,gradient=stage('volume_field',lambda:field(points['xyz'],gradient=True))
            alpha,alpha_factor=stage('alpha_fit',lambda:fit_alpha(points,B,dcfg))
            G=stage('vacuum_G',lambda:vacuum_G(geometry,level,flux['edge_flux'],field,dcfg))
            qs=stage('qs',lambda:qs_metric(points,B,gradient,alpha['iota_coefficients'],G,dcfg))
            quality=stage('surface_quality',lambda:surface_quality(geometry,level,field.trace,dcfg))
            values=np.empty(7)
            stage('coil_metrics',lambda:self.library.check(self.library.lib.ce_cpu_coil_metrics(
                *[pointer(a) for a in (coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,coils.currents_a)],
                *coils.coeffs_x.shape,coils.nfp,pointer(values))))
            diagnostics.update(dict(zip(('coil_length_mean','coil_curvature_p95','coil_curvature_max',
                'coil_min_intercoil_distance','coil_min_axis_distance','coil_high_mode_energy_fraction',
                'coil_current_abs_max_a'),values)))
            diagnostics.update(qs_target_global_error_per_helicity=float(qs['raw_qs']),
                qs_target_edge_error_per_helicity=float(qs['raw_qs_outer']),
                iota_target_surface=float(alpha['iota_boundary']),iota_min=float(alpha['iota_min']),
                iota_max=float(alpha['iota_max']),target_quality_error=float(quality['target_quality_error']),
                psi_train_rms=factor['train_rms'],alpha_relative_l2=float(alpha['relative_l2']),
                target_level=float(level),psi_fit_method=config.psi_fit,
                flux_edge=float(flux['edge_flux']),surface_volume=float(flux['surface_volume']))
            timing['total']=perf_counter()-start
            raw=dict(status='ok',diagnostics=diagnostics,timing=timing)
            snapshot=dict(config=options,axis_samples=axis,psi_coefficients=coefficients,
                psi_scale=factor['scale'],psi_upper=factor['upper'],**target,**selection,
                alpha_upper=alpha_factor['upper'],alpha_scale=alpha_factor['scale'],
                alpha_center_scaled_solution=alpha_factor['scaled_solution'],
                alpha_coefficients=alpha['alpha_coefficients'].numpy(),
                iota_coefficients=alpha['iota_coefficients'].numpy(),
                flux_coefficients=flux['coefficients'].numpy(),score_result=raw)
            if two is not None:snapshot['two_stage']=two
            return raw,snapshot
        except (ValueError,RuntimeError) as error:
            diagnostics['error_message']=str(error);timing['total']=perf_counter()-start
            return dict(status=status,diagnostics=diagnostics,timing=timing),None
        finally:
            field.close()
