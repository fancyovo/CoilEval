"""Batched axis, psi and target-volume response."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .front import (PsiGeometry,monomials,periodic_hermite,_trig,
                               _coefficient_matrix,predict_target_level,_tensor_product_column_norms,_frozen_fit_tensor)


def differentiable_batch_axis(field,captured_starts,*,nfp=4,trace_steps=240,
                              sample_count=240,condition_limit=1e10):
    """Independent 2x2 implicit periodic-root solves plus moving sampled axes."""
    if captured_starts.ndim!=2 or captured_starts.shape[-1]!=2:
        raise ValueError('captured axis starts must be [query,2]')
    y=captured_starts.detach()[:,None,:].requires_grad_()
    end=field.trace(y,0.,2*math.pi/nfp,trace_steps,return_samples=False)
    # Batch field/trace kernels have no cross-query dependencies. Consequently
    # summing the same output coordinate gives every diagonal 2x2 block at once.
    J=torch.stack([torch.autograd.grad(end[:,0,k].sum(),y,retain_graph=True)[0][:,0,:]
                   for k in range(2)],1)
    matrix=torch.eye(2,device=y.device,dtype=y.dtype)[None]-J.detach()
    condition=torch.linalg.cond(matrix)
    valid=torch.isfinite(condition)&(condition<condition_limit)
    if not bool(valid.all()):
        indices=torch.nonzero(~valid).flatten().tolist()
        raise ValueError(f'ill-conditioned periodic axis for query indices {indices}')
    correction=torch.linalg.solve(matrix,(end-end.detach())[:,0,:,None])[...,0]
    starts=y.detach()[:,0,:]+correction
    substeps=(trace_steps+sample_count-1)//sample_count
    total_steps=sample_count*substeps
    history=field.trace(starts[:,None,:],0.,2*math.pi/nfp,total_steps,return_samples=True)
    expected=(total_steps+1,len(starts),1,2)
    if history.shape!=expected:
        raise ValueError(f'batch trace history must have shape {expected}')
    rz=history[:-1:substeps,:,0,:].transpose(0,1)
    phi=torch.arange(sample_count,device=y.device,dtype=y.dtype)*(2*math.pi/(nfp*sample_count))
    R,Z=rz.unbind(-1);c,s=phi.cos(),phi.sin()
    xyz=torch.stack((R*c,R*s,Z),-1)
    B=field(xyz).to(rz.dtype)
    br=B[...,0]*c+B[...,1]*s;bp=-B[...,0]*s+B[...,1]*c
    floor=torch.where(bp<0,-torch.ones_like(bp),torch.ones_like(bp))*1e-14
    denominator=torch.where(bp.abs()>1e-14,bp,floor)
    slope=torch.stack((R*br/denominator,R*B[...,2]/denominator),-1)
    samples=torch.cat((rz,slope),-1)
    return samples,dict(monodromy=J.detach(),condition=condition.detach(),
        root_residual=torch.linalg.vector_norm(end-y,dim=-1)[:,0].detach(),valid=valid.detach())


class BatchTensorProductPsiOperator:
    """Full tensor-product rows; per-query radial exclusion is zero padding."""
    def __init__(self,field,axis_samples,a,*,nfp=4,degree=10,mtor=12,
                 radial_grid=48,vertical_grid=48,phi_grid=48,rho_min=.002,
                 dtype=torch.float32):
        if axis_samples.ndim!=3 or axis_samples.shape[-1]!=4:
            raise ValueError('axis samples require [query,sample,4]')
        device=axis_samples.device;queries=axis_samples.shape[0]
        a64=torch.as_tensor(a,device=device,dtype=torch.float64).expand(queries).detach()
        rho64=torch.as_tensor(rho_min,device=device,dtype=torch.float64).expand(queries).detach()
        if not bool((a64>0).all()&(rho64>=0).all()):
            raise ValueError('positive fitting scales and nonnegative exclusion radii required')
        self.degree,self.mtor,self.nfp=degree,mtor,nfp
        self.a=a64.to(dtype)
        # Each query has the same normalized Cartesian lattice. Keeping all
        # unit-disc rows and zeroing native-excluded rows preserves its operator
        # while allowing different physical a and rho_min within one batch.
        rr,zz=torch.meshgrid(torch.linspace(-1.,1.,radial_grid,device=device,dtype=torch.float64),
                            torch.linspace(-1.,1.,vertical_grid,device=device,dtype=torch.float64),indexing='ij')
        disc=torch.hypot(rr,zz)<=1.
        X64,Y64=rr[disc],zz[disc]
        mask=torch.hypot(X64,Y64)[None,:]*a64[:,None]>=rho64[:,None]
        if not bool(mask.any(-1).all()):raise ValueError('one query has no native psi training rows')
        self.row_mask=mask
        X,Y=X64.to(dtype),Y64.to(dtype)
        phi=(torch.arange(phi_grid,device=device,dtype=torch.float64)*(2*math.pi/(nfp*phi_grid))).to(dtype)
        self.X,self.Y,self.phi=X,Y,phi
        mono=monomials(degree)
        self.P=torch.stack([X**ax*Y**bz for ax,bz in mono],-1)
        self.PR=torch.stack([ax*X**max(ax-1,0)*Y**bz for ax,bz in mono],-1)
        self.PZ=torch.stack([bz*X**ax*Y**max(bz-1,0) for ax,bz in mono],-1)
        self.T=_trig(phi,mtor,nfp);self.Tp=_trig(phi,mtor,nfp,derivative=True)
        ax=axis_samples.to(dtype)
        ra,rap=periodic_hermite(phi,ax[:,:,0],ax[:,:,2],nfp)
        za,zap=periodic_hermite(phi,ax[:,:,1],ax[:,:,3],nfp)
        scale=self.a[:,None,None]
        R=ra[:,None,:]+scale*X[None,:,None]
        Z=za[:,None,:]+scale*Y[None,:,None]
        self.points=torch.stack((R*phi.cos(),R*phi.sin(),Z),-1)
        B=field(self.points.reshape(queries,-1,3)).reshape(self.points.shape).to(dtype)
        cp=(-B[...,0]*phi.sin()+B[...,1]*phi.cos())/R
        cr=B[...,0]*phi.cos()+B[...,1]*phi.sin()-cp*rap[:,None,:]
        cz=B[...,2]-cp*zap[:,None,:]
        active=mask[:,:,None].to(dtype)
        self.features=cr/scale*active,cz/scale*active,cp*active
        self.rhs=-2*X[None,:,None]*self.features[0]
        self.valid_rows=mask.sum(-1)*phi_grid
        self.coefficient_count=len(mono)*(2*mtor+1)-1

    def apply(self,coefficients):
        C=_coefficient_matrix(coefficients,self.degree,self.mtor)
        cr,cz,cp=self.features
        return cr*(self.PR@C@self.T.T)+cz*(self.PZ@C@self.T.T)+cp*(self.P@C@self.Tp.T)

    def transpose(self,rows):
        cr,cz,cp=self.features
        C=self.PR.T@(cr*rows)@self.T+self.PZ.T@(cz*rows)@self.T+self.P.T@(cp*rows)@self.Tp
        return C.flatten(1)[:,1:]

    def column_norms(self):
        return _tensor_product_column_norms(self.features,self.PR,self.PZ,self.P,self.T,self.Tp)


def batch_four_step_pcgls(operator,coefficients,scales,uppers,*,ridge=1e-6,iterations=4,
                           live_column_scale=False):
    dtype,device=operator.rhs.dtype,operator.rhs.device
    c0,scale,upper=[_frozen_fit_tensor(v,device=device,dtype=dtype) for v in (coefficients,scales,uppers)]
    if c0.shape!=scale.shape or upper.shape!=(*c0.shape,c0.shape[-1]):
        raise ValueError('every query requires independent c0/scale[C] and upper[C,C]')
    if not bool((scale>0).all()):raise ValueError('positive psi normalization scales required')
    sqrt_ridge=torch.as_tensor(ridge,device=device,dtype=dtype).expand(c0.shape[0]).sqrt()
    live_scale=operator.column_norms() if live_column_scale else scale
    ratio=live_scale/scale
    operator.column_scales=live_scale.detach()
    operator.column_scale_ratio=ratio.detach()
    def triangular(v,transpose=False):
        return torch.linalg.solve_triangular(upper.transpose(-1,-2) if transpose else upper,
                                             v[...,None],upper=not transpose)[...,0]
    z=c0*scale
    residual=operator.rhs-operator.apply(z/scale)
    rr=-sqrt_ridge[:,None]*ratio*z
    def normal(r,rr):return triangular(operator.transpose(r)/scale+sqrt_ridge[:,None]*ratio*rr,True)
    n=normal(residual,rr);direction=n;gamma=n.square().sum(-1)
    for iteration in range(iterations):
        transformed=triangular(direction)
        image=operator.apply(transformed/scale);ri=sqrt_ridge[:,None]*ratio*transformed
        delta=image.square().sum((-2,-1))+ri.square().sum(-1)
        alpha=gamma/delta.clamp_min(1e-30)
        z=z+alpha[:,None]*transformed
        residual=residual-alpha[:,None,None]*image
        rr=rr-alpha[:,None]*ri
        if iteration+1==iterations:break
        n=normal(residual,rr);next_gamma=n.square().sum(-1)
        direction=n+(next_gamma/gamma.clamp_min(1e-30))[:,None]*direction
        gamma=next_gamma
    rms=torch.sqrt(residual.square().sum((-2,-1))/operator.valid_rows)
    return z/scale,rms


@dataclass
class BatchFrontResult:
    axis_samples:torch.Tensor
    coefficients:torch.Tensor
    levels:torch.Tensor
    psi_train_rms:torch.Tensor
    axis_diagnostics:dict
    operator:BatchTensorProductPsiOperator
    geometries:list

    def geometry(self,index):return self.geometries[index]
    def __len__(self):return len(self.geometries)


def build_batch_front(field,centers,*,config_overrides=None,dtype=torch.float32,nfp=4):
    if not centers or len(centers)!=field.positions.shape[0]:
        raise ValueError('one independent center snapshot required for every field query')
    cfgs=[{**c.get('config',{}),**(config_overrides or {})} for c in centers]
    defaults=dict(axis_trace_steps=240,axis_sample_count=240,psi_poly_degree=10,psi_m_tor=12,
        psi_n_r=48,psi_n_z=48,psi_n_phi=48,psi_live_column_scale=False,surface_max_radius_scale=1.,surface_newton_iters=20,surface_newton_tolerance=1e-12)
    common={}
    for name,default in defaults.items():
        values=[cfg.get(name,default) for cfg in cfgs]
        if any(value!=values[0] for value in values[1:]):
            raise ValueError(f'group compatible batches before solving: differing {name}')
        common[name]=values[0]
    device=field.positions.device
    def stack(name):return torch.stack([torch.as_tensor(c[name],device=device,dtype=torch.float64) for c in centers])
    axes,diagnostics=differentiable_batch_axis(field,stack('axis_samples')[:,0,:2],nfp=nfp,
        trace_steps=int(common['axis_trace_steps']),sample_count=int(common['axis_sample_count']))
    radii=[float(c['psi_a']) for c in cfgs]
    op=BatchTensorProductPsiOperator(field,axes,radii,nfp=nfp,degree=int(common['psi_poly_degree']),mtor=int(common['psi_m_tor']),
        radial_grid=int(common['psi_n_r']),vertical_grid=int(common['psi_n_z']),phi_grid=int(common['psi_n_phi']),
        rho_min=[float(c.get('psi_rho_min',.002)) for c in cfgs],dtype=dtype)
    coefficients,rms=batch_four_step_pcgls(op,stack('psi_coefficients'),stack('psi_scale'),stack('psi_upper'),
        ridge=[float(c.get('psi_ridge',1e-6)) for c in cfgs],iterations=4,
        live_column_scale=bool(common['psi_live_column_scale']))
    diagnostics['psi_column_scale_relative_max']=(op.column_scale_ratio-1).abs().amax(-1).detach()
    geometries=[PsiGeometry(axes[i],coefficients[i],nfp,radii[i],int(common['psi_poly_degree']),int(common['psi_m_tor']),
        maximum_radius_scale=float(common['surface_max_radius_scale']),root_iterations=int(common['surface_newton_iters']),
        root_tolerance=float(common['surface_newton_tolerance'])) for i in range(len(centers))]
    levels=torch.stack([predict_target_level(g,c) for g,c in zip(geometries,centers)])
    return BatchFrontResult(axes,coefficients,levels,rms,diagnostics,op,geometries)
