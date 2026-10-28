"""CPU QR fits with native basis, row weights and column regularization."""
from __future__ import annotations
import numpy as np
import torch
from scipy.linalg import get_lapack_funcs, solve_triangular
from .._autograd.front import monomials, periodic_hermite, _trig


def _copy_columns(destination,source,scale=None):
    """Fill contiguous column slices without a large strided 2D copy."""
    for column in range(source.shape[1]):
        if scale is None:
            np.copyto(destination[:,column],source[:,column])
        else:
            np.divide(source[:,column],scale[column],out=destination[:,column])


def regularized_qr(matrix, rhs, ridge, *, scale_floor=1e-15, driver='torch'):
    """Householder QR of [A/D; sqrt(ridge) I], without materializing Q."""
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    scale = np.maximum(np.sqrt(np.einsum('ij,ij->j',matrix,matrix,dtype=np.float64)),scale_floor)
    system = np.zeros((rows+cols,cols),dtype=matrix.dtype,order='F')
    _copy_columns(system[:rows],matrix,scale.astype(matrix.dtype))
    system[rows+np.arange(cols),np.arange(cols)] = np.sqrt(ridge)
    b = np.zeros((rows+cols,1),dtype=matrix.dtype,order='F')
    b[:rows,0] = np.asarray(rhs).reshape(-1)
    if driver=='torch':
        packed=torch.from_numpy(system);tau=torch.empty(cols,dtype=packed.dtype)
        qr,tau=torch.geqrf(packed,out=(packed,tau))
        target=torch.from_numpy(b)
        transformed=torch.ormqr(qr,tau,target,transpose=True,out=target)
        upper_tensor=torch.triu(qr[:cols])
        z=torch.linalg.solve_triangular(upper_tensor,transformed[:cols],upper=True)[:,0].numpy()
        upper=upper_tensor.numpy()
    elif driver=='scipy':
        geqrf, ormqr = get_lapack_funcs(('geqrf','ormqr'),(system,))
        _,_,work,info = geqrf(system,lwork=-1,overwrite_a=True)
        if info: raise RuntimeError(f'CPU QR workspace query failed: {info}')
        qr,tau,_,info = geqrf(system,lwork=int(work[0]),overwrite_a=True)
        if info: raise RuntimeError(f'CPU QR failed: {info}')
        _,work,info = ormqr('L','T',qr,tau,b,lwork=-1,overwrite_c=True)
        transformed,_,info = ormqr('L','T',qr,tau,b,lwork=int(work[0]),overwrite_c=True)
        if info: raise RuntimeError(f'CPU QR right-hand side failed: {info}')
        upper = np.triu(qr[:cols]).copy()
        z = solve_triangular(upper,transformed[:cols,0],check_finite=False)
    else:raise ValueError('CPU QR driver must be torch or scipy')
    coefficient = z/scale.astype(matrix.dtype)
    residual = ((torch.from_numpy(matrix)@torch.from_numpy(coefficient)).numpy()
        if driver=='torch' else matrix@coefficient)-np.asarray(rhs).reshape(-1)
    return coefficient.astype(np.float64), dict(scale=scale,upper=upper,
        scaled_solution=z,train_rms=float(np.sqrt(np.mean(residual.astype(np.float64)**2))))


def psi_grid(axis, cfg, nfp):
    q=np.linspace(-cfg.psi_a,cfg.psi_a,cfg.psi_n_r)
    z=np.linspace(-cfg.psi_a,cfg.psi_a,cfg.psi_n_z)
    rr,zz=np.meshgrid(q,z,indexing='ij')
    keep=(np.hypot(rr,zz)>=cfg.psi_rho_min)&(np.hypot(rr,zz)<=cfg.psi_a)
    phi=np.arange(cfg.psi_n_phi)*2*np.pi/(nfp*cfg.psi_n_phi)
    ax=torch.from_numpy(axis); p=torch.from_numpy(phi)
    ar,_=periodic_hermite(p,ax[:,0],ax[:,2],nfp)
    az,_=periodic_hermite(p,ax[:,1],ax[:,3],nfp)
    R=(rr[keep,None]+ar.numpy()[None,:]).reshape(-1)
    Z=(zz[keep,None]+az.numpy()[None,:]).reshape(-1)
    return np.column_stack((R,Z)),np.tile(phi,int(keep.sum()))


def fit_psi(field, axis, rz, phi, cfg, nfp, weights=None):
    dtype=torch.float32 if cfg.psi_precision_mode==2 else torch.float64
    ax=torch.tensor(axis,dtype=dtype); ph=torch.tensor(phi,dtype=dtype)
    rz=torch.tensor(rz,dtype=dtype)
    ar,arp=periodic_hermite(ph,ax[:,0],ax[:,2],nfp)
    az,azp=periodic_hermite(ph,ax[:,1],ax[:,3],nfp)
    X,Y=(rz[:,0]-ar)/cfg.psi_a,(rz[:,1]-az)/cfg.psi_a
    cp,sp=ph.cos(),ph.sin()
    xyz=torch.stack((rz[:,0]*cp,rz[:,0]*sp,rz[:,1]),-1)
    magnetic=field(xyz).to(dtype)
    bphi=(-magnetic[:,0]*sp+magnetic[:,1]*cp)/rz[:,0]
    br=magnetic[:,0]*cp+magnetic[:,1]*sp-bphi*arp
    bz=magnetic[:,2]-bphi*azp
    root_weight=(torch.ones_like(X) if weights is None else torch.as_tensor(
        np.array(weights,copy=True),dtype=dtype).sqrt())
    br,bz,bphi=br*root_weight,bz*root_weight,bphi*root_weight
    T,Tp=_trig(ph,cfg.psi_m_tor,nfp),_trig(ph,cfg.psi_m_tor,nfp,derivative=True)
    xp=[X**k for k in range(cfg.psi_poly_degree+1)]
    yp=[Y**k for k in range(cfg.psi_poly_degree+1)]
    count=len(monomials(cfg.psi_poly_degree))*(2*cfg.psi_m_tor+1)-1
    matrix=np.empty((len(X),count),dtype=np.float32 if dtype==torch.float32 else np.float64,order='F')
    cursor=0
    for term,(a,b) in enumerate(monomials(cfg.psi_poly_degree)):
        p=xp[a]*yp[b]
        pr=a*xp[max(a-1,0)]*yp[b]/cfg.psi_a
        pz=b*xp[a]*yp[max(b-1,0)]/cfg.psi_a
        block=(br*pr+bz*pz)[:,None]*T+(bphi*p)[:,None]*Tp
        if term==0:block=block[:,1:]
        _copy_columns(matrix[:,cursor:cursor+block.shape[1]],block.numpy());cursor+=block.shape[1]
    rhs=(-2*X*br/cfg.psi_a).numpy()
    return regularized_qr(matrix,rhs,cfg.psi_ridge)


def fit_alpha(points,B,config):
    from .._autograd.downstream import (clebsch_modes,smooth_alpha_weights,
        _design_chunk,iota_profile_extrema)
    c=config;n=len(B);nf=min(c.alpha_fit_point_count,n)
    index=torch.arange(nf)*n//nf
    rho,theta,phi=[points[k][index].to(B.dtype) for k in ('rho','theta','phi')]
    magnetic=B[index];gs,gt,gp=[points[k][index].to(B.dtype) for k in ('grad_s','grad_theta','grad_phi')]
    norm2=magnetic.square().sum(-1)
    normal=(magnetic*gs).sum(-1)/gs.square().sum(-1).clamp_min(1e-30)
    tangent=magnetic-normal[:,None]*gs
    bt,bp=(tangent*gt).sum(-1),(tangent*gp).sum(-1)
    if c.alpha_radial_weighting=='smooth':weight,counts=smooth_alpha_weights(rho,norm2,c)
    else:
        bins=(rho*c.radial_bin_count).long().clamp(0,c.radial_bin_count-1)
        counts=torch.bincount(bins,minlength=c.radial_bin_count)
        weight=counts[bins].clamp_min(1).to(B.dtype).rsqrt()/norm2.clamp_min(1e-30).sqrt()
        weight=weight*(nf/weight.double().square().sum()).sqrt().to(B.dtype)
    modes=clebsch_modes(c.alpha_radial_order,c.alpha_poloidal_order,c.alpha_toroidal_order,c.alpha_radial_basis)
    metadata=tuple(torch.tensor(v,dtype=B.dtype) for v in modes);nm=len(modes[0]);cols=nm+c.iota_degree+1
    ones=torch.ones(cols,dtype=B.dtype)
    matrix=np.empty((nf,cols),dtype=np.float32,order='F')
    for lo in range(0,cols,c.alpha_chunk_columns):
        hi=min(lo+c.alpha_chunk_columns,cols)
        block=_design_chunk(rho,theta,phi,bt,bp,weight,ones,metadata,c.nfp,lo,hi,c.iota_degree,c.alpha_radial_basis).numpy()
        _copy_columns(matrix[:,lo:hi],block)
    rhs=(-weight*bt).numpy();coeff,cache=regularized_qr(matrix,rhs,c.alpha_ridge,scale_floor=1e-20)
    iota=torch.from_numpy(coeff[nm:].astype(np.float32))
    result=dict(alpha_coefficients=torch.from_numpy(coeff[:nm].astype(np.float32)),iota_coefficients=iota,
        iota_boundary=iota.sum(),**iota_profile_extrema(iota,c.volume_rho_min),
        relative_l2=cache['train_rms']*np.sqrt(nf)/max(np.linalg.norm(rhs.astype(np.float64)),1e-30),
        normal_B_relative_l2=float(((normal[:,None]*gs).double().square().sum()/magnetic.double().square().sum()).sqrt()))
    return result,cache
