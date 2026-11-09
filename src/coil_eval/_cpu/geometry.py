"""CPU target-volume policy and formal volume sampling."""
import math
import numpy as np
import torch
from .native import pointer
from .._autograd.front import monomials, _trig
from .._autograd.downstream import polynomial


def target_surface(geometry, cfg, library, *, fit_stage=1):
    # Match the native sampled guard; a resolved psi Fourier series does not
    # make the nonlinear radial-capacity envelope band limited to that order.
    n=cfg.flux_phi_count; guard=max(2*cfg.psi_n_phi,4*cfg.psi_m_tor+1)
    phi=np.concatenate((np.arange(n)/n,np.arange(guard)/guard))*2*np.pi/geometry.nfp
    theta=np.arange(cfg.flux_theta_count)*2*np.pi/cfg.flux_theta_count
    pp=torch.from_numpy(phi)[:,None];tt=torch.from_numpy(theta)[None,:]
    powers=np.arange(geometry.degree+1)
    poly=np.ascontiguousarray(geometry.rays(pp,tt).numpy()*geometry.a**powers)
    ar,az,_,_=geometry.axis_eval(torch.from_numpy(phi))
    out=np.zeros(6);radii=np.empty((n,len(theta)))
    code=library.lib.ce_cpu_target(pointer(poly),pointer(ar.numpy()),pointer(az.numpy()),
        len(phi),n,len(theta),geometry.degree,geometry.a,cfg.surface_max_radius_scale,
        cfg.target_volume,cfg.target_volume_relative_tolerance,pointer(out),pointer(radii))
    if code:
        from .._two_stage.boundary import MaxClosedBoundaryError
        from .._two_stage.model import PsiModel
        from .._two_stage.target_failure import explain_target_failure
        model=PsiModel(geometry.coefficients.numpy(),geometry.axis_samples.numpy(),nfp=geometry.nfp,
                       a=geometry.a,degree=geometry.degree,mtor=geometry.mtor)
        diagnostics=dict(error_message=f'target volume is outside the closed surface capacity ({out[0]:.9g} m^3)',
                         target_volume_requested=cfg.target_volume,target_volume_maximum=out[0])
        explain_target_failure(diagnostics,model,cfg,fit_stage,force_guard=True)
        raise MaxClosedBoundaryError(diagnostics['error_message'],diagnostics)
    level=out[2];ph=torch.from_numpy(phi[:n])[:,None];th=torch.from_numpy(theta)[None,:]
    r=torch.from_numpy(radii);R=ar[:n,None]+r*th.cos();Z=az[:n,None]+r*th.sin()
    s,gr,gz,_=geometry.psi_eval(R,Z,ph)
    psir=gr*th.cos()+gz*th.sin();weight=4*math.pi**2/r.numel()
    dw=-weight*R*r/psir
    trig=_trig(ph.expand_as(r),geometry.mtor,geometry.nfp)
    X,Y=r*th.cos()/geometry.a,r*th.sin()/geometry.a
    blocks=[]
    for i,(a,b) in enumerate(monomials(geometry.degree)):
        block=(dw*X**a*Y**b)[...,None]*trig
        if i==0:block=block[...,1:]
        blocks.append(block.sum((0,1)))
    snapshot=dict(target_adjoint_weights=torch.cat(blocks).numpy(),
        target_axis_weights=(.5*weight*r.square()).sum(-1).numpy(),target_phi=phi[:n],
        target_axis_R=ar[:n].numpy(),target_level=np.array([level]),
        target_adjoint_residual=np.array([float((dw*(s-level)).sum())]),
        target_dVds=np.array([out[4]]))
    return torch.tensor(level,dtype=torch.float64),snapshot,dict(
        target_capacity_volume=out[0],target_capacity_level=out[1],target_surface_volume=out[3])


def _formal_boundary_radii(geometry,polynomial,level):
    """Use the local first-branch solver without rejecting other valid rays.

    Ordinary batches solve together. A failed batch is split until unavailable
    rays are isolated; their residual remains infinite, so formal filtering
    removes their candidates before the existing ordered stratification.
    """
    shape=polynomial.shape[:-1]
    rays=polynomial.reshape(-1,polynomial.shape[-1])
    radii=polynomial.new_zeros(len(rays))
    valid=torch.zeros(len(rays),dtype=torch.bool,device=polynomial.device)
    pending=[(0,len(rays))]
    while pending:
        start,end=pending.pop()
        try:
            roots=geometry.solve_rays(rays[start:end],level)
        except ValueError:
            if end-start>1:
                mid=(start+end)//2
                pending.extend(((mid,end),(start,mid)))
            continue
        radii[start:end]=roots
        valid[start:end]=True
    powers=torch.arange(polynomial.shape[-1],device=polynomial.device,dtype=polynomial.dtype)
    residual=((rays*radii[:,None]**powers).sum(-1)-level).abs()
    residual=torch.where(valid,residual,torch.full_like(residual,float('inf')))
    return radii.reshape(shape),residual.reshape(shape)


def formal_volume_points(geometry,level,flux,c,cfg):
    rays=c.volume_phi_count*c.volume_theta_count
    nr=(math.ceil(c.volume_point_count*1.25)+rays-1)//rays
    count=rays*nr
    ip=torch.arange(c.volume_phi_count,dtype=torch.float32)[:,None]
    it=torch.arange(c.volume_theta_count,dtype=torch.float32)[None,:]
    phi=((ip+.417)*float(2*math.pi/(c.nfp*c.volume_phi_count))).expand(-1,c.volume_theta_count)
    theta=(it+.613+.217*ip).remainder(c.volume_theta_count)*float(2*math.pi/c.volume_theta_count)
    # The formal sampler retains valid rays individually before stratification.
    p=geometry.rays(phi.double(),theta.double())
    r,error=_formal_boundary_radii(geometry,p,level)
    boundary=r.float().flatten().repeat_interleave(nr)
    phi=phi.flatten().repeat_interleave(nr);theta=theta.flatten().repeat_interleave(nr)
    radial=torch.arange(nr,dtype=torch.float32).repeat(rays)
    radius=boundary*(c.volume_rho_min**2+(1-c.volume_rho_min**2)*(radial+.371)/nr).sqrt()
    # Native volume geometry is evaluated in FP32.
    local=type(geometry)(geometry.axis_samples.float(),geometry.coefficients.float(),geometry.nfp,
        geometry.a,geometry.degree,geometry.mtor)
    ar,az,arp,azp=local.axis_eval(phi)
    R,Z=ar+radius*theta.cos(),az+radius*theta.sin()
    s,gr,gz,gp=local.psi_eval(R,Z,phi)
    coeff=flux['coefficients'].float();fs=polynomial(coeff,s)
    fd=polynomial(coeff*torch.arange(1,len(coeff)+1),s,constant=True)
    rho=(fs/flux['edge_flux'].float()).clamp_min(0).sqrt()
    cp,sp=phi.cos(),phi.sin();r2=radius.square().clamp_min(1e-30)
    tr,tz=radius*theta.sin()/r2,-radius*theta.cos()/r2
    tp=(radius*theta.cos()*azp-radius*theta.sin()*arp)/r2
    points=dict(xyz=torch.stack((R*cp,R*sp,Z),-1),
        grad_s=torch.stack((gr*cp-gp/R*sp,gr*sp+gp/R*cp,gz),-1),
        grad_theta=torch.stack((tr*cp-tp/R*sp,tr*sp+tp/R*cp,tz),-1),
        grad_phi=torch.stack((-sp/R,cp/R,torch.zeros_like(R)),-1),
        rho=rho,theta=-theta,phi=phi,volume_weight=R*boundary.square(),flux_derivative=fd,s=s)
    valid=(torch.isfinite(s)&torch.isfinite(rho)&(R>0)&
        (error.flatten().repeat_interleave(nr)<=cfg.flux_boundary_tolerance)&
        (s>=float(level)*c.volume_rho_min**2)&(s<=float(level)*(1+2e-5))&(rho<=1.0001))
    available=valid.nonzero().flatten();minimum=max(c.alpha_fit_point_count,c.volume_point_count,math.ceil(.95*count))
    if len(available)<minimum:
        raise ValueError(f'volume sampler found {len(available)} valid points; minimum is {minimum}')
    source=available[torch.arange(c.volume_point_count)*len(available)//c.volume_point_count]
    selected={k:v[source] for k,v in points.items()}
    return selected,dict(volume_source_indices=source.numpy(),
        volume_sampling_shape=np.array([c.volume_phi_count,c.volume_theta_count,nr,count])),dict(
        volume_candidate_count=count,volume_available_count=len(available),volume_valid_fraction=len(available)/count)
