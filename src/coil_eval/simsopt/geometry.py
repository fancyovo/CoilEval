"""Surface projection with reusable exact Simsopt coefficient factorizations."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import time


def surface_from_dofs(dofs,*,nfp,order,nphi,ntheta=None,stellsym=True,offset=(0.,0.)):
    from simsopt.geo import SurfaceXYZTensorFourier
    ntheta=nphi if ntheta is None else ntheta
    surface=SurfaceXYZTensorFourier(mpol=order,ntor=order,nfp=nfp,stellsym=stellsym,
        quadpoints_phi=(np.arange(nphi)+offset[0])/(nfp*nphi),
        quadpoints_theta=(np.arange(ntheta)+offset[1])/ntheta)
    if dofs is not None:surface.set_dofs(np.asarray(dofs,dtype=float))
    return surface


class SurfaceProjector:
    """Keep at most ``cache_size`` QR factors; release them with the worker.

    The tensor-Fourier coefficient basis is linear and depends only on the
    symmetry, resolution and quadrature grid, so independent geometries can
    share its factorization without sharing fitted coefficients.
    """
    def __init__(self,*,device='cpu',cache_size=4,method='cached_qr',cached_geometry=False):
        if method not in ('cached_qr','reference'):
            raise ValueError("projection method must be 'cached_qr' or 'reference'")
        if cache_size<1:raise ValueError('cache_size must be positive')
        self.device=device;self.cache_size=cache_size;self.method=method
        self.cached_geometry=bool(cached_geometry)
        self._cache=OrderedDict();self.hits=0;self.misses=0

    def clear(self):self._cache.clear()

    def fit(self,xyz,*,nfp,order,stellsym=True):
        xyz=np.asarray(xyz,dtype=float)
        if xyz.ndim!=3 or xyz.shape[-1]!=3 or not np.isfinite(xyz).all():
            raise ValueError('surface samples must be finite (phi,theta,3) arrays')
        nphi,ntheta=xyz.shape[:2]
        surface=surface_from_dofs(None,nfp=nfp,order=order,nphi=nphi,ntheta=ntheta,stellsym=stellsym)
        predicted=None
        if self.method=='reference':
            surface.least_squares_fit(xyz)
            reused=False
        else:
            key=(nfp,order,bool(stellsym),nphi,ntheta)
            reused=key in self._cache
            if reused:
                self.hits+=1;entry=self._cache.pop(key);self._cache[key]=entry
            else:
                self.misses+=1
                surface.set_dofs(np.zeros_like(surface.get_dofs()))
                base=surface.gamma().copy()
                design=np.asarray(surface.dgamma_by_dcoeff()).reshape(-1,len(surface.get_dofs()))
                if self.device=='cpu':
                    q,r=np.linalg.qr(design,mode='reduced')
                else:
                    import torch
                    matrix=torch.as_tensor(design,device=self.device,dtype=torch.float64)
                    q,r=torch.linalg.qr(matrix,mode='reduced')
                entry=(base,q,r)
                self._cache[key]=entry
                if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
            base,q,r=entry
            rhs=(xyz-base).reshape(-1,1)
            if self.device=='cpu':
                from scipy.linalg import solve_triangular
                coefficients=solve_triangular(r,q.T@rhs,lower=False)[:,0]
            else:
                import torch
                b=torch.as_tensor(rhs,device=self.device,dtype=torch.float64)
                fitted_coefficients=torch.linalg.solve_triangular(r,q.mT@b,upper=True)[:,0]
                coefficients=fitted_coefficients.cpu().numpy()
            if self.cached_geometry:
                if self.device=='cpu':
                    predicted=base+(q@(r@coefficients)).reshape(xyz.shape)
                else:
                    predicted=base+(q@(r@fitted_coefficients)).cpu().numpy().reshape(xyz.shape)
            surface.set_dofs(coefficients)
        distance=np.linalg.norm((surface.gamma() if predicted is None else predicted)-xyz,axis=-1)
        return surface,{'spectral_fit_rms_m':float(np.sqrt(np.mean(distance**2))),
                        'spectral_fit_max_m':float(np.max(distance)),'projection_factor_reused':reused}


class SurfaceGeometryCache:
    """Exact Simsopt DOF basis maps with bounded persistent storage.

    Bases come from Simsopt's actual coefficient Jacobians, including its
    symmetry and rotating Cartesian frame. A few unit DOFs independently check
    the coefficient and Cartesian layout when each cache entry is constructed.
    Cached maps are double precision on either NumPy or the selected device.
    """
    def __init__(self,*,device='cpu',method='reference',max_bytes=512*1024**2,cache_size=4):
        if method not in ('reference','cached_basis'):
            raise ValueError("geometry method must be 'reference' or 'cached_basis'")
        if max_bytes<1 or cache_size<1:raise ValueError('geometry cache limits must be positive')
        self.device=device;self.method=method;self.max_bytes=int(max_bytes);self.cache_size=int(cache_size)
        self._cache=OrderedDict();self.bytes=0;self.hits=0;self.misses=0

    def clear(self):self._cache.clear();self.bytes=0

    def evaluate(self,dofs,*,nfp,order,nphi,ntheta=None,stellsym=True,offset=(0.,0.),derivatives=False):
        ntheta=nphi if ntheta is None else ntheta
        dofs=np.asarray(dofs,dtype=float)
        names=('gamma','gammadash1','gammadash2') if derivatives else ('gamma',)
        options=dict(nfp=nfp,order=order,nphi=nphi,ntheta=ntheta,stellsym=stellsym,offset=offset)
        if self.method=='reference':
            surface=surface_from_dofs(dofs,**options)
            return {name:np.asarray(getattr(surface,name)()).copy() for name in names}
        key=(int(nfp),int(order),int(nphi),int(ntheta),bool(stellsym),tuple(offset),bool(derivatives))
        if key in self._cache:
            self.hits+=1;entry=self._cache.pop(key);self._cache[key]=entry
        else:
            self.misses+=1
            surface=surface_from_dofs(None,**options)
            count=len(surface.get_dofs())
            if dofs.shape!=(count,):raise ValueError('coefficient count does not match the Simsopt surface basis')
            surface.set_dofs(np.zeros(count))
            entry={name:(np.asarray(getattr(surface,name)()).copy(),
                np.asarray(getattr(surface,'d'+name+'_by_dcoeff')()).reshape(-1,count).copy()) for name in names}
            for index in sorted({0,count//2,count-1}):
                unit=np.zeros(count);unit[index]=1.;surface.set_dofs(unit)
                for name,(base,matrix) in entry.items():
                    expected=base+matrix[:,index].reshape(base.shape)
                    if not np.allclose(getattr(surface,name)(),expected,rtol=2e-13,atol=2e-13):
                        raise ValueError('Simsopt coefficient basis layout validation failed: '+name)
            if self.device!='cpu':
                import torch
                entry={name:(base,torch.as_tensor(matrix,device=self.device,dtype=torch.float64))
                       for name,(base,matrix) in entry.items()}
            size=sum(base.nbytes+(matrix.nbytes if self.device=='cpu' else matrix.numel()*matrix.element_size())
                     for base,matrix in entry.values())
            if size>self.max_bytes:raise ValueError('requested Simsopt geometry basis exceeds the cache byte limit')
            while self._cache and (len(self._cache)>=self.cache_size or self.bytes+size>self.max_bytes):
                _,(removed_size,_)=self._cache.popitem(last=False);self.bytes-=removed_size
            self._cache[key]=(size,entry);self.bytes+=size
        if isinstance(entry,tuple):_,entry=entry
        if dofs.ndim!=1 or any(matrix.shape[1]!=len(dofs) for base,matrix in entry.values()):
            raise ValueError('coefficient count does not match the cached Simsopt surface basis')
        if self.device=='cpu':return {name:base+(matrix@dofs).reshape(base.shape) for name,(base,matrix) in entry.items()}
        import torch
        coefficients=torch.as_tensor(dofs,device=self.device,dtype=torch.float64)
        return {name:base+(matrix@coefficients).cpu().numpy().reshape(base.shape) for name,(base,matrix) in entry.items()}


def periodic_resample(parameter,values,target):
    parameter=np.mod(np.asarray(parameter,dtype=float),2*np.pi)
    index=np.argsort(parameter);parameter=parameter[index];values=np.asarray(values)[index]
    keep=np.r_[True,np.diff(parameter)>1e-12]
    parameter,values=parameter[keep],values[keep]
    if len(parameter)<3:raise ValueError('alpha coordinate has fewer than three distinct nodes')
    xp=np.r_[parameter[-1]-2*np.pi,parameter,parameter[0]+2*np.pi]
    return np.column_stack([np.interp(target,xp,np.r_[values[-1,k],values[:,k],values[0,k]])
                            for k in range(values.shape[1])])


def _alpha_dense_theta_derivative(alpha,rho,nphi,ntheta):
    """Exact separable Fourier contraction on a fixed-rho validation grid."""
    theta=-np.arange(ntheta)*2*np.pi/ntheta
    phi=np.arange(nphi)*2*np.pi/(alpha.nfp*nphi)
    if not hasattr(alpha,'modes') or not hasattr(alpha,'lambda_coefficients'):
        th,ph=np.broadcast_arrays(theta[None,:],phi[:,None])
        return alpha.evaluate(np.full(th.shape,rho),th,ph,device='cpu')[1],'alpha.evaluate'
    from .clebsch import radial
    modes=np.asarray(alpha.modes,dtype=int)
    if not len(modes):return np.zeros((nphi,ntheta)),'separable_fourier'
    keys,inverse=np.unique(modes[:,:2],axis=0,return_inverse=True)
    values=np.asarray([float(radial(float(rho),l,m,getattr(alpha,'radial_basis','zernike'))) for l,m in keys])
    amplitude=np.asarray(alpha.lambda_coefficients)*values[inverse]
    mm,nn=modes[:,1],modes[:,2]
    ms,mi=np.unique(mm,return_inverse=True);ns,ni=np.unique(nn,return_inverse=True)
    # d_theta Re[(c_cos-i*c_sin)*exp(i*(m*theta-n*nfp*phi))].
    coefficients=np.zeros((len(ns),len(ms)),dtype=np.complex128)
    np.add.at(coefficients,(ni,mi),mm*amplitude*np.where(modes[:,3]==0,1j,1.))
    toroidal=np.exp(-1j*alpha.nfp*phi[:,None]*ns)
    poloidal=np.exp(1j*theta[:,None]*ms)
    return (toroidal@coefficients@poloidal.T).real,'separable_fourier'


def alpha_reparameterize(xyz,alpha,*,rho=1.0,device='cpu',reject_folds=True,
                         alpha_fold_policy='reject',alpha_min_jacobian=.05):
    """Reparameterize source points, optionally damping only the seed's lambda.

    The compatible reject policy keeps the original reject_folds behavior.
    Damp validates all dense nodes and the actual extraction nodes, preserving
    iota while shrinking lambda enough to retain a positive coordinate map.
    """
    nphi,ntheta=xyz.shape[:2]
    theta=np.arange(ntheta)*2*np.pi/ntheta
    phi=np.arange(nphi)*2*np.pi/(alpha.nfp*nphi)
    th,ph=np.broadcast_arrays(-theta[None,:],phi[:,None])
    lam,derivative,_=alpha.evaluate(np.full(th.shape,rho),th,ph,device=device)
    minimum=float(np.min(1+derivative))
    noninvertible=float(np.mean(1+derivative<=0))
    details={}
    if alpha_fold_policy=='reject':
        if reject_folds and minimum<=0:
            raise ValueError(f'noninvertible alpha map: minimum 1+lambda_theta={minimum}')
    elif alpha_fold_policy=='damp':
        if not 0<alpha_min_jacobian<1:raise ValueError('alpha_min_jacobian must lie strictly between zero and one')
        dense_nphi,dense_ntheta=max(97,nphi),max(257,ntheta)
        dense,method=_alpha_dense_theta_derivative(alpha,rho,dense_nphi,dense_ntheta)
        if not all(np.isfinite(value).all() for value in (lam,derivative,dense)):
            raise ValueError('nonfinite alpha coordinate map')
        original_dense_min=float(np.min(1+dense))
        limiting=min(float(np.min(derivative)),float(np.min(dense)))
        damping=min(1.,(1-alpha_min_jacobian)/(-limiting)) if limiting<0 else 1.
        if damping<1:
            # A small margin keeps the requested floor strict after FP64 sums.
            damping*=1-1e-12
        coarse_min=float(np.min(1+damping*derivative))
        dense_min=float(np.min(1+damping*dense))
        if not 0<damping<=1 or min(coarse_min,dense_min)<alpha_min_jacobian:
            raise ValueError('damped alpha coordinate map failed the positive-Jacobian postcheck')
        lam=damping*lam
        details=dict(alpha_fold_policy='damp',original_min_jacobian=min(minimum,original_dense_min),
            original_coarse_min_jacobian=minimum,original_dense_min_jacobian=original_dense_min,
            damping_factor=float(damping),damped_min_jacobian=min(coarse_min,dense_min),
            damped_dense_min_jacobian=dense_min,alpha_min_jacobian=float(alpha_min_jacobian),
            original_noninvertible_fraction=noninvertible,
            dense_grid=dict(nphi=dense_nphi,ntheta=dense_ntheta),dense_evaluation_method=method)
        minimum=coarse_min
        noninvertible=float(np.mean(1+damping*derivative<=0))
    else:raise ValueError("alpha_fold_policy must be 'reject' or 'damp'")
    transformed=np.stack([periodic_resample(-theta+lam[k],row,theta)
                          for k,row in enumerate(xyz)])
    return transformed,{'one_plus_lambda_theta_min':minimum,
                        'noninvertible_fraction':noninvertible,**details}


def correct_surface(source,correction,projector,*,order,nphi=65,ntheta=67,interpolation_size=97,
                    inversion_method='resident',geometry_cache=None):
    """Preserve alpha while inverting phi_B=phi+nu and projecting the surface."""
    from scipy.ndimage import map_coordinates
    from .fourier import invert_toroidal_map
    nfp=correction.nfp;stage=time.perf_counter();timings={}
    pb,tb=np.broadcast_arrays(np.arange(nphi)[:,None]/(nfp*nphi),
                             np.arange(ntheta)[None,:]/ntheta)
    alpha=tb-correction.iota*pb
    old,mapping=invert_toroidal_map(correction,alpha,pb,device=projector.device,method=inversion_method)
    theta=alpha+correction.iota*old
    timings['inversion_s']=time.perf_counter()-stage;stage=time.perf_counter()
    if geometry_cache is None:
        sampled=surface_from_dofs(source.get_dofs(),nfp=nfp,order=source.mpol,nphi=interpolation_size,
                                  stellsym=source.stellsym)
        xyz=sampled.gamma()
    else:
        xyz=geometry_cache.evaluate(source.get_dofs(),nfp=nfp,order=source.mpol,
            nphi=interpolation_size,stellsym=source.stellsym)['gamma']
    phi=np.arange(interpolation_size)[:,None]/(nfp*interpolation_size)
    radius=np.hypot(xyz[:,:,0],xyz[:,:,1])
    offset=np.angle(np.exp(1j*(np.arctan2(xyz[:,:,1],xyz[:,:,0])-2*np.pi*phi)))
    coordinates=np.vstack(((old*nfp%1).ravel()*interpolation_size,
                           (theta%1).ravel()*interpolation_size))
    timings['dense_geometry_s']=time.perf_counter()-stage;stage=time.perf_counter()
    fields=[map_coordinates(value,coordinates,order=5,mode='grid-wrap').reshape(pb.shape)
            for value in (radius,offset,xyz[:,:,2])]
    rr,off,zz=fields;angle=2*np.pi*old+off
    timings['interpolation_s']=time.perf_counter()-stage;stage=time.perf_counter()
    fitted,projection=projector.fit(np.stack((rr*np.cos(angle),rr*np.sin(angle),zz),axis=-1),
                                    nfp=nfp,order=order,stellsym=source.stellsym)
    timings['final_projection_s']=time.perf_counter()-stage
    return fitted,dict(projection,timings=timings,**mapping)
