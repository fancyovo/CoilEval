"""Persistent source-psi -> alpha -> nu initialization with injected fields."""
from __future__ import annotations
from dataclasses import dataclass, replace
from operator import index
import re
import time
import numpy as np
from .clebsch import AlphaFit
from .fourier import fit_correction
from .geometry import SurfaceProjector,SurfaceGeometryCache,alpha_reparameterize,correct_surface,surface_from_dofs
from .state import SourcePsi,SurfaceSeed,SOURCE_KIND,require_kind


@dataclass(frozen=True)
class InitializerConfig:
    surface_order:int=12
    extract_order:int=24
    extract_maxiter:int=20
    extract_tol:float=1e-12
    nu_order:int=12
    fit_nphi:int=65
    fit_ntheta:int=67
    interpolation_size:int=97
    regularization:float=0.0
    device:str='cpu'
    nu_method:str='fft'
    projection_method:str='cached_qr'
    inversion_method:str='resident'
    geometry_method:str='reference'
    geometry_cache_bytes:int=512*1024**2
    reject_alpha_folds:bool=True
    max_radius_scale:float=1.0
    alpha_fold_policy:str='reject'
    alpha_min_jacobian:float=.05

    def __post_init__(self):
        if min(self.surface_order,self.extract_order,self.nu_order,self.extract_maxiter)<1 or self.extract_tol<=0:
            raise ValueError('surface, extraction and nu orders must be positive')
        if min(self.fit_nphi,self.fit_ntheta)<=2*self.nu_order:
            raise ValueError('nu fit grid must resolve every requested Fourier mode')
        if self.interpolation_size<3 or self.max_radius_scale<=0:
            raise ValueError('invalid interpolation size or extraction radius')
        if self.geometry_method not in ('reference','cached_basis') or self.geometry_cache_bytes<1:
            raise ValueError('invalid geometry method or cache byte limit')
        if self.alpha_fold_policy not in ('reject','damp') or not 0<self.alpha_min_jacobian<1:
            raise ValueError('invalid alpha fold policy or minimum Jacobian')


def _initializer_device(config, device_id):
    """Resolve one native/Torch GPU without importing an optional dependency."""
    if device_id is not None:
        try:
            if isinstance(device_id, (bool, np.bool_)):
                raise TypeError
            device_id = index(device_id)
        except TypeError as exc:
            raise ValueError('device_id must be a nonnegative integer') from exc
        if device_id < 0:
            raise ValueError('device_id must be a nonnegative integer')
    device = str(config.device)
    match = re.fullmatch(r'(cpu|cuda)(?::([0-9]+))?', device)
    if match is None:
        raise ValueError('initializer device must be cpu, cpu:<index>, cuda or cuda:<index>')
    kind, suffix = match.groups()
    if kind == 'cuda':
        explicit = None if suffix is None else int(suffix)
        if explicit is not None and device_id is not None and explicit != device_id:
            raise ValueError(f'InitializerConfig.device={device!r} conflicts with device_id={device_id}')
        device_id = explicit if explicit is not None else (0 if device_id is None else device_id)
        config = replace(config, device=f'cuda:{device_id}')
    elif device_id is None:
        device_id = 0
    return config, device_id


class AlphaNuInitializer:
    """Reuse imports and exact geometry factors across independent samples.

    ``field`` accepts Cartesian (N,3) arrays and returns B or (B,dBdx).
    ``prepare`` requires a fitted SourcePsi and its paired AlphaFit. The caller
    may use the existing native alpha capture or a separately validated dense
    alpha fit; the selected origin is retained in the result diagnostics.
    """
    def __init__(self,field,*,library=None,config=None,device_id=None):
        self.field=field;self.library=library
        self.config,self.device_id=_initializer_device(config or InitializerConfig(),device_id)
        if library is None and str(self.config.device).startswith('cuda'):
            from .._native_loader import resolve_library
            self.library=resolve_library()
        self.projector=SurfaceProjector(device=self.config.device,method=self.config.projection_method,
            cached_geometry=self.config.geometry_method=='cached_basis')
        self.geometry=SurfaceGeometryCache(device=self.config.device,method=self.config.geometry_method,
            max_bytes=self.config.geometry_cache_bytes)

    def close(self):self.projector.clear();self.geometry.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def prepare(self,source:SourcePsi,alpha:AlphaFit,*,rho=1.0):
        require_kind(source.kind,SOURCE_KIND,'alpha+nu source')
        if source.nfp!=alpha.nfp:raise ValueError('source and alpha field periods differ')
        cfg=self.config;timings={};started=time.perf_counter();stage=started
        xyz,radii,extraction=source.surface_points(cfg.extract_order,library=self.library,
            device_id=self.device_id,max_radius_scale=cfg.max_radius_scale,
            maxiter=cfg.extract_maxiter,tol=cfg.extract_tol)
        timings['surface_extraction_s']=time.perf_counter()-stage;stage=time.perf_counter()
        parameterized,transform=alpha_reparameterize(xyz,alpha,rho=rho,device=cfg.device,
            reject_folds=cfg.reject_alpha_folds,alpha_fold_policy=cfg.alpha_fold_policy,
            alpha_min_jacobian=cfg.alpha_min_jacobian)
        timings['alpha_reparameterization_s']=time.perf_counter()-stage;stage=time.perf_counter()
        alpha_surface,projection=self.projector.fit(parameterized,nfp=source.nfp,order=cfg.surface_order)
        timings['alpha_surface_projection_s']=time.perf_counter()-stage;stage=time.perf_counter()
        if cfg.geometry_method=='cached_basis':
            training=self.geometry.evaluate(alpha_surface.get_dofs(),nfp=source.nfp,order=cfg.surface_order,
                nphi=cfg.fit_nphi,ntheta=cfg.fit_ntheta,derivatives=True)
            points=training['gamma']
        else:
            training=surface_from_dofs(alpha_surface.get_dofs(),nfp=source.nfp,order=cfg.surface_order,
                                       nphi=cfg.fit_nphi,ntheta=cfg.fit_ntheta)
            points=np.asarray(training.gamma())
        timings['nu_training_geometry_s']=time.perf_counter()-stage;field_stage=time.perf_counter()
        values=self.field(np.ascontiguousarray(points.reshape(-1,3)))
        B=np.asarray(values[0] if isinstance(values,tuple) else values).reshape(points.shape)
        if not np.isfinite(B).all() or np.any(np.linalg.norm(B,axis=-1)==0):
            raise ValueError('invalid field on alpha surface')
        timings['nu_training_B_s']=time.perf_counter()-field_stage;derivative_stage=time.perf_counter()
        iota=float(np.asarray(alpha.iota(rho)))
        tangent=(training['gammadash1']+iota*training['gammadash2'] if cfg.geometry_method=='cached_basis'
                 else training.gammadash1()+iota*training.gammadash2())
        local_G=np.sum(B*tangent,axis=-1);G=float(np.mean(local_G))
        if not np.isfinite(G) or G==0:raise ValueError('zero or nonfinite covariant G')
        timings['nu_training_tangents_s']=time.perf_counter()-derivative_stage
        timings['nu_field_s']=time.perf_counter()-stage;stage=time.perf_counter()
        correction=fit_correction(local_G/G-1,iota=iota,nfp=source.nfp,order=cfg.nu_order,
                                   regularization=cfg.regularization,method=cfg.nu_method)
        timings['nu_fit_s']=time.perf_counter()-stage;stage=time.perf_counter()
        corrected,mapping=correct_surface(alpha_surface,correction,self.projector,order=cfg.surface_order,
            nphi=cfg.fit_nphi,ntheta=cfg.fit_ntheta,interpolation_size=cfg.interpolation_size,
            inversion_method=cfg.inversion_method,geometry_cache=self.geometry if cfg.geometry_method=='cached_basis' else None)
        timings['nu_correction_and_projection_s']=time.perf_counter()-stage
        timings['total_s']=time.perf_counter()-started
        return SurfaceSeed(np.asarray(corrected.get_dofs()).copy(),source.nfp,cfg.surface_order,
            iota,G,source.level,float(np.mean(radii)),mapping['spectral_fit_rms_m'],
            source_axis_samples=source.axis_samples.copy(),
            diagnostics={'timings':timings,'extraction':extraction,'alpha_transform':transform,
                         'alpha_projection':projection,'nu_mapping':mapping,'nu_fit':correction.diagnostics,
                         'alpha_origin':dict(alpha.diagnostics),'source_provenance':dict(source.provenance),
                         'projection_cache_hits':self.projector.hits,'projection_cache_misses':self.projector.misses,
                         'geometry_method':cfg.geometry_method,'geometry_cache_hits':self.geometry.hits,
                         'geometry_cache_misses':self.geometry.misses,'geometry_cache_bytes':self.geometry.bytes})
