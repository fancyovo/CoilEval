"""Standard Simsopt LS/Newton with independent final geometry validation."""
from __future__ import annotations
from dataclasses import dataclass
import inspect
import time
import numpy as np
from .geometry import surface_from_dofs
from .state import SurfaceSeed,StandardSurface,INITIALIZER_KIND,SOURCE_KIND,require_kind


@dataclass(frozen=True)
class SolverConfig:
    ls_tol:float=1e-10
    ls_maxiter:int=100
    newton_tol:float=1e-12
    newton_maxiter:int=30
    constraint_weight:float=1.0
    max_newton_residual_norm:float=1e-8
    max_final_relative_l2:float=1e-4
    max_final_normal_p95:float=1e-4
    validation_grids:tuple=(49,97)


def field_from_coils(coils,*,quadpoints=None):
    """Build a standard Simsopt BiotSavart field from physical CoilSet data."""
    from simsopt.field import BiotSavart,Current,coils_via_symmetries
    from simsopt.geo import CurveXYZFourier
    curves=[]
    order=(coils.coeffs_x.shape[1]-1)//2
    for x,y,z in zip(coils.coeffs_x,coils.coeffs_y,coils.coeffs_z):
        curve=CurveXYZFourier(quadpoints=quadpoints or max(256,15*order),order=order)
        curve.set_dofs(np.r_[x,y,z]);curves.append(curve)
    physical=coils_via_symmetries(curves,[Current(float(c)) for c in coils.currents_a],
                                 nfp=coils.nfp,stellsym=True)
    return BiotSavart(physical)


def _field_values(field,xyz):
    if hasattr(field,'set_points') and hasattr(field,'B'):
        field.set_points(np.ascontiguousarray(xyz.reshape(-1,3)))
        return np.asarray(field.B()).reshape(xyz.shape)
    if callable(field):
        value=field(np.ascontiguousarray(xyz.reshape(-1,3)))
        return np.asarray(value[0] if isinstance(value,tuple) else value).reshape(xyz.shape)
    raise TypeError('field must be a Simsopt field or Cartesian B callback')


def _oriented_target_volume(initial_signed_volume,requested_volume):
    initial=float(initial_signed_volume)
    if not np.isfinite(initial) or initial==0:
        raise ValueError('the initializer must enclose a finite nonzero signed volume')
    if requested_volume is None:return initial
    requested=float(requested_volume)
    if not np.isfinite(requested) or requested<=0:
        raise ValueError('target_volume must be finite and positive')
    # Alpha uses clockwise poloidal angle. A positive physical target must keep
    # the initializer's orientation when passed to Simsopt's signed Volume.
    return float(np.copysign(requested,initial))


def validate_surface(surface,field,*,iota,G,axis_samples=None,check_self_intersection=True):
    xyz=np.asarray(surface.gamma());gp=np.asarray(surface.gammadash1());gt=np.asarray(surface.gammadash2())
    B=_field_values(field,xyz);magnitude=np.linalg.norm(B,axis=-1)
    normal=np.cross(gp,gt);normal_norm=np.linalg.norm(normal,axis=-1)
    residual=G*B-magnitude[...,None]**2*(gp+iota*gt)
    weighted=residual/np.maximum(magnitude[...,None],1e-300)
    sine=abs(np.sum(B*normal,axis=-1))/np.maximum(magnitude*normal_norm,1e-300)
    angle=np.unwrap(np.arctan2(xyz[:,:,1],xyz[:,:,0]),axis=0)
    fraction=(xyz.shape[0]-1)/xyz.shape[0]
    winding=(angle[-1]-angle[0])/(fraction*2*np.pi/surface.nfp)
    finite=bool(all(np.isfinite(v).all() for v in (xyz,B,magnitude,normal_norm,residual,winding)))
    topology={'checked':False,'reason':'source_axis_unavailable'}
    if axis_samples is not None and finite:
        from .topology import angular_covering
        topology=angular_covering(xyz,axis_samples,surface.nfp)
    intersection={'checked':False,'confirmed_self_intersection':False,
                  'status':'not_checked','reason':'deferred_to_final_validation_grid'}
    if check_self_intersection:
        if finite and np.min(normal_norm)>1e-12:
            from .intersections import detect_self_intersection
            intersection=detect_self_intersection(surface)
        else:
            intersection['reason']='surface_nonfinite_or_degenerate'
    return {'finite':finite,'relative_l2':float(np.linalg.norm(weighted)/(max(abs(G),1e-300)*np.sqrt(magnitude.size))),
            'normal_B_sine_p95':float(np.percentile(sine,95)),
            'normal_min':float(np.min(normal_norm)),'toroidal_winding_min':float(np.min(winding)),
            'signed_volume_m3':float(surface.volume()),'point_count':magnitude.size,
            'angular_covering':topology,'self_intersection':intersection}


class StandardSimsoptSolver:
    def __init__(self,*,config=None):self.config=config or SolverConfig()

    def solve(self,seed:SurfaceSeed,field,*,target_volume=None,validation_field=None):
        require_kind(seed.kind,INITIALIZER_KIND,'standard Simsopt initialization')
        require_kind(seed.source_kind,SOURCE_KIND,'standard Simsopt source')
        from simsopt.geo import BoozerSurface,Volume,boozer_surface_residual
        cfg=self.config;start=time.perf_counter();exact=2*seed.order+1
        surface=surface_from_dofs(seed.dofs,nfp=seed.nfp,order=seed.order,nphi=exact,stellsym=seed.stellsym)
        initial_volume=float(surface.volume())
        volume=_oriented_target_volume(initial_volume,target_volume)
        problem=BoozerSurface(field,surface,Volume(surface),volume)
        kwargs=dict(tol=cfg.ls_tol,maxiter=cfg.ls_maxiter,iota=seed.iota,G=seed.G,
                    constraint_weight=cfg.constraint_weight)
        if 'weight_inv_modB' in inspect.signature(problem.minimize_boozer_penalty_constraints_ls).parameters:
            kwargs['weight_inv_modB']=True
        stage=time.perf_counter();ls=problem.minimize_boozer_penalty_constraints_ls(**kwargs)
        ls_seconds=time.perf_counter()-stage
        ls_dofs=np.asarray(surface.get_dofs()).copy()
        final=surface_from_dofs(ls_dofs,nfp=seed.nfp,order=seed.order,nphi=exact,stellsym=seed.stellsym)
        newton_problem=BoozerSurface(field,final,Volume(final),volume)
        stage=time.perf_counter()
        newton=newton_problem.solve_residual_equation_exactly_newton(tol=cfg.newton_tol,
            maxiter=cfg.newton_maxiter,iota=float(ls['iota']),G=float(ls['G']),verbose=False)
        newton_seconds=time.perf_counter()-stage
        iota,G=float(newton['iota']),float(newton['G'])
        dofs=np.asarray(final.get_dofs()).copy()
        residual_norm=float(np.linalg.norm(boozer_surface_residual(final,iota,G,field,derivatives=0)[0]))
        converged=bool(newton.get('success',False)) or residual_norm<=cfg.max_newton_residual_norm
        grids=[];stage=time.perf_counter()
        grid_sizes=sorted(set((exact,*cfg.validation_grids)))
        for size in grid_sizes:
            offset=(0.,0.) if size==exact else (.371,.371)
            dense=surface_from_dofs(dofs,nfp=seed.nfp,order=seed.order,nphi=size,
                                    stellsym=seed.stellsym,offset=offset)
            grids.append(dict(size=size,offset=offset[0],**validate_surface(dense,
                validation_field if validation_field is not None else field,iota=iota,G=G,
                axis_samples=getattr(seed,'source_axis_samples',None),
                check_self_intersection=(size==grid_sizes[-1]))))
        dense=grids[-1]
        hard={'newton_converged':converged,'finite':dense['finite'],
              'toroidal_winding':dense['toroidal_winding_min']>0,
              'normal_nonzero':dense['normal_min']>1e-12,'nonzero_volume':abs(dense['signed_volume_m3'])>0,
              'no_detected_self_intersection':not dense['self_intersection']['confirmed_self_intersection']}
        if getattr(seed,'source_axis_samples',None) is not None:
            hard['single_angular_covering']=all(row['angular_covering'].get('valid_convention',False)
                                               for row in grids)
        strict={'dense_relative_l2':dense['relative_l2']<=cfg.max_final_relative_l2,
                'dense_normal_field_p95':dense['normal_B_sine_p95']<=cfg.max_final_normal_p95,
                'self_intersection_audit_resolved':dense['self_intersection']['status']=='not_detected_in_sampled_sections'}
        accepted=all(hard.values())
        quality='strict_pass' if accepted and all(strict.values()) else 'accepted_with_quality_warning' if accepted else 'rejected'
        diagnostics={'least_squares':{'success':bool(ls.get('success',False)),
                     'iterations':int(ls.get('iter',-1)),'iota':float(ls['iota']),'G':float(ls['G'])},
                     'newton':{'success':bool(newton.get('success',False)),
                     'iterations':int(newton.get('iter',-1)),'residual_norm':residual_norm},
                     'grids':grids,'target_volume_m3':abs(volume),'target_volume_signed_m3':volume,
                     'initial_volume_signed_m3':initial_volume,'hard_downstream_checks':hard,
                     'strict_quality_checks':strict,
                     'timings':{'least_squares_s':ls_seconds,'newton_s':newton_seconds,
                                'validation_s':time.perf_counter()-stage,'total_s':time.perf_counter()-start}}
        return StandardSurface(seed,dofs,iota,G,accepted,quality,diagnostics)
