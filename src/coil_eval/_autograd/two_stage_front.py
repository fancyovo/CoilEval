"""Local reverse path through both psi fits and the moving second-fit domain.

The captured member set, active capacity ray and count-adjustment decisions
define a smooth local branch. Axis motion, first-fit coefficients, boundary
roots, section areas and lattice spacing remain differentiable on that branch.
``local_value=True`` evaluates the corresponding first-order root continuation
for local finite-difference checks; full domain-rebuilding differences are a
separate acceptance check because membership and active branches can change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from .front import (FrontResult, PsiGeometry, TensorProductPsiOperator,
                    _coefficient_matrix, _trig, _monomial_features, differentiable_axis,
                    cylindrical_rhs, four_step_pcgls, monomials, periodic_hermite,
                    predict_target_level,_section_column_norms)
from .batch_front import (BatchFrontResult, BatchTensorProductPsiOperator,
                          batch_four_step_pcgls, differentiable_batch_axis)


def _live_tensor(value, *, device, dtype):
    if not getattr(getattr(value,'flags',None),'writeable',True):
        value=value.copy()
    return torch.as_tensor(value,device=device,dtype=dtype)


def _stack_live(values, *, device, dtype):
    return torch.stack([_live_tensor(value,device=device,dtype=dtype) for value in values])


class SectionPsiOperator:
    """Irregular-point psi design, factored as 63 monomials x 25 trig terms.

    Rows are padded independently per geometric-phi section. Only O(N*63)
    polynomial storage is formed; apply/transpose use section GEMMs and never
    materialize the O(N*1574) design matrix. The field is evaluated only at
    actual points. Padded rows and their derivatives are identically zero.
    """
    def __init__(self, field, axis_samples, rz, phis, section_index, *,
                 nfp=4, a=.15, degree=10, mtor=12, dtype=torch.float32,row_weights=None):
        device=axis_samples.device
        self.degree,self.mtor,self.nfp,self.a=degree,mtor,nfp,a
        rz=_live_tensor(rz,device=device,dtype=dtype)
        phis=_live_tensor(phis,device=device,dtype=dtype)
        section=_live_tensor(section_index,device=device,dtype=torch.long)
        if rz.ndim!=2 or rz.shape[1]!=2 or section.shape!=(len(rz),) or not len(rz):
            raise ValueError('irregular training points require aligned (N,2) RZ and (N,) section indices')
        if phis.ndim!=1 or not len(phis) or bool(((section<0)|(section>=len(phis))).any()):
            raise ValueError('section indices must refer to the supplied phi vector')
        counts=torch.bincount(section,minlength=len(phis))
        if bool((counts==0).any()):
            raise ValueError('each training section must contain at least one point')
        width=int(counts.max())
        order=torch.argsort(section,stable=True)
        ordered_section=section[order]
        first=torch.cumsum(counts,0)-counts
        slots=torch.arange(len(rz),device=device)-torch.repeat_interleave(first,counts)
        padded_indices=torch.zeros((len(phis),width),device=device,dtype=torch.long)
        mask=torch.zeros_like(padded_indices,dtype=torch.bool)
        padded_indices[ordered_section,slots]=order
        mask[ordered_section,slots]=True
        self.mask,self.padded_indices,self.section_index=mask,padded_indices,section
        self.point_count=len(rz)
        weights=(torch.ones(len(rz),device=device,dtype=dtype) if row_weights is None else
                 _live_tensor(row_weights,device=device,dtype=dtype))
        if weights.shape!=(len(rz),) or not bool(torch.isfinite(weights).all() & (weights>0).all()):
            raise ValueError('one finite positive least-squares row weight required per training point')
        self.row_weights=weights
        root_weight=weights[padded_indices].sqrt()
        self.phis,self.rz=phis,rz
        point_phi=phis[section]
        self.points=torch.stack((rz[:,0]*point_phi.cos(),rz[:,0]*point_phi.sin(),rz[:,1]),-1)
        magnetic=field(self.points,gradient=False).to(dtype)
        if magnetic.shape!=self.points.shape:
            raise ValueError('field must return one Cartesian B vector per training point')
        ax=axis_samples.to(dtype)
        ra,rap=periodic_hermite(phis,ax[:,0],ax[:,2],nfp)
        za,zap=periodic_hermite(phis,ax[:,1],ax[:,3],nfp)
        rpad=rz[:,0][padded_indices]
        zpad=rz[:,1][padded_indices]
        R=torch.where(mask,rpad,torch.ones_like(rpad))
        X=torch.where(mask,(rpad-ra[:,None])/a,torch.zeros_like(rpad))
        Y=torch.where(mask,(zpad-za[:,None])/a,torch.zeros_like(zpad))
        B=magnetic[padded_indices]
        phi=phis[:,None]
        cp=(-B[...,0]*phi.sin()+B[...,1]*phi.cos())/R
        cr=B[...,0]*phi.cos()+B[...,1]*phi.sin()-cp*rap[:,None]
        cz=B[...,2]-cp*zap[:,None]
        self.features=tuple(torch.where(mask,value*root_weight,torch.zeros_like(value)) for value in (cr,cz,cp))
        self.X,self.Y=X,Y
        self.monomials=monomials(degree)
        self.P,self.PR,self.PZ=_monomial_features(X,Y,degree,a)
        self.T,self.Tp=_trig(phis,mtor,nfp),_trig(phis,mtor,nfp,derivative=True)
        self.rhs=-2*X*self.features[0]/a
        self.coefficient_count=len(self.monomials)*(2*mtor+1)-1

    def apply(self,coefficients):
        C=_coefficient_matrix(coefficients,self.degree,self.mtor)
        amplitude=self.T@C.T
        amplitude_phi=self.Tp@C.T
        radial=torch.bmm(self.PR,amplitude[...,None])[...,0]
        vertical=torch.bmm(self.PZ,amplitude[...,None])[...,0]
        toroidal=torch.bmm(self.P,amplitude_phi[...,None])[...,0]
        cr,cz,cp=self.features
        return cr*radial+cz*vertical+cp*toroidal

    def transpose(self,rows):
        if rows.shape!=self.rhs.shape:
            raise ValueError('transpose rows must have the operator padded section shape')
        cr,cz,cp=self.features
        radial=torch.bmm(self.PR.transpose(1,2),(cr*rows)[...,None])[...,0]
        vertical=torch.bmm(self.PZ.transpose(1,2),(cz*rows)[...,None])[...,0]
        toroidal=torch.bmm(self.P.transpose(1,2),(cp*rows)[...,None])[...,0]
        C=(radial+vertical).T@self.T+toroidal.T@self.Tp
        return C.reshape(-1)[1:]

    def residual_rms(self,coefficients):
        residual=self.apply(coefficients)-self.rhs
        return torch.sqrt(residual.square().sum()/self.point_count)

    def column_norms(self):
        return _section_column_norms(self.features,self.PR,self.PZ,self.P,self.T,self.Tp)


class BatchSectionPsiOperator:
    """Independent irregular-section operators evaluated by batched GEMMs.

    Query point counts and section counts may differ. Only the field call's
    point dimension and each section's rows are padded; dummy field points are
    positive-R copies and all dummy design rows have zero contribution. Normal
    equations, preconditioners and residual norms remain independent per query.
    """
    def __init__(self,field,axis_samples,rz,phis,section_index,a,*,
                 nfp=4,degree=10,mtor=12,dtype=torch.float32,row_weights=None):
        if axis_samples.ndim!=3 or axis_samples.shape[-1]!=4:
            raise ValueError('batch axis samples require [query,sample,4]')
        device=axis_samples.device;queries=len(axis_samples)
        if not queries or not (len(rz)==len(phis)==len(section_index)==queries):
            raise ValueError('every query requires its own RZ points, phis and section indices')
        self.degree,self.mtor,self.nfp=degree,mtor,nfp
        self.a=torch.as_tensor(a,device=device,dtype=dtype).expand(queries).detach()
        if not bool((self.a>0).all()):
            raise ValueError('positive per-query psi scales required')
        point_lists=[_live_tensor(p,device=device,dtype=dtype) for p in rz]
        phi_lists=[_live_tensor(p,device=device,dtype=dtype) for p in phis]
        section_lists=[_live_tensor(p,device=device,dtype=torch.long).detach() for p in section_index]
        if row_weights is None:
            row_weights=[None]*queries
        if len(row_weights)!=queries:
            raise ValueError('one independent row-weight vector required per batch query')
        weight_lists=[torch.ones(len(p),device=device,dtype=dtype) if w is None else
                      _live_tensor(w,device=device,dtype=dtype) for p,w in zip(point_lists,row_weights)]
        for points,weights in zip(point_lists,weight_lists):
            if weights.shape!=(len(points),) or not bool(torch.isfinite(weights).all() & (weights>0).all()):
                raise ValueError('one finite positive least-squares row weight required per training point')
        counts=[]
        for points,phi,section in zip(point_lists,phi_lists,section_lists):
            if points.ndim!=2 or points.shape[1]!=2 or not len(points) or section.shape!=(len(points),):
                raise ValueError('batch irregular training points require aligned nonempty RZ and section indices')
            if phi.ndim!=1 or not len(phi) or bool(((section<0)|(section>=len(phi))).any()):
                raise ValueError('batch section indices must refer to their own supplied phi vector')
            count=torch.bincount(section,minlength=len(phi))
            if not bool((count>0).all()):
                raise ValueError('each training section must contain at least one point')
            counts.append(count)
        sections=max(len(p) for p in phi_lists)
        width=max(int(c.max()) for c in counts)
        maximum_points=max(len(p) for p in point_lists)
        padded_indices=torch.zeros((queries,sections,width),device=device,dtype=torch.long)
        mask=torch.zeros_like(padded_indices,dtype=torch.bool)
        point_batches=[];point_phi_batches=[];phi_batches=[];weight_batches=[]
        ra_batches=[];za_batches=[];rap_batches=[];zap_batches=[]
        for query,(points,phi,section,count) in enumerate(zip(point_lists,phi_lists,section_lists,counts)):
            order=torch.argsort(section,stable=True)
            offsets=torch.cumsum(count,0)-count
            slots=torch.arange(len(points),device=device)-torch.repeat_interleave(offsets,count)
            padded_indices[query,section[order],slots]=order
            mask[query,section[order],slots]=True
            padding=maximum_points-len(points)
            point_batches.append(torch.cat((points,points[:1].expand(padding,-1))))
            weights=weight_lists[query]
            weight_batches.append(torch.cat((weights,weights[:1].expand(padding))))
            point_phi=phi[section]
            point_phi_batches.append(torch.cat((point_phi,point_phi[:1].expand(padding))))
            missing=sections-len(phi)
            phi_batches.append(torch.cat((phi,phi.new_zeros(missing))))
            axis=axis_samples[query].to(dtype)
            ra,rap=periodic_hermite(phi,axis[:,0],axis[:,2],nfp)
            za,zap=periodic_hermite(phi,axis[:,1],axis[:,3],nfp)
            for values,container in ((ra,ra_batches),(za,za_batches),(rap,rap_batches),(zap,zap_batches)):
                container.append(torch.cat((values,values.new_zeros(missing))))
        point_batch=torch.stack(point_batches)
        point_phi=torch.stack(point_phi_batches)
        self.points=torch.stack((point_batch[...,0]*point_phi.cos(),
                                 point_batch[...,0]*point_phi.sin(),point_batch[...,1]),-1)
        magnetic=field(self.points,gradient=False).to(dtype)
        if magnetic.shape!=self.points.shape:
            raise ValueError('batch field must return one Cartesian B vector per point/query')
        query_index=torch.arange(queries,device=device)[:,None,None]
        rpad=point_batch[:,:,0][query_index,padded_indices]
        zpad=point_batch[:,:,1][query_index,padded_indices]
        R=torch.where(mask,rpad,torch.ones_like(rpad))
        ra,za,rap,zap=[torch.stack(v)[:,:,None] for v in (ra_batches,za_batches,rap_batches,zap_batches)]
        scale=self.a[:,None,None]
        X=torch.where(mask,(rpad-ra)/scale,torch.zeros_like(rpad))
        Y=torch.where(mask,(zpad-za)/scale,torch.zeros_like(zpad))
        self.phis=torch.stack(phi_batches)
        phi=self.phis[:,:,None]
        B=magnetic[query_index,padded_indices]
        cp=(-B[...,0]*phi.sin()+B[...,1]*phi.cos())/R
        cr=B[...,0]*phi.cos()+B[...,1]*phi.sin()-cp*rap
        cz=B[...,2]-cp*zap
        root_weight=torch.stack(weight_batches)[query_index,padded_indices].sqrt()
        self.row_weights=weight_lists
        self.features=tuple(torch.where(mask,v*root_weight,torch.zeros_like(v)) for v in (cr,cz,cp))
        self.mask,self.padded_indices=mask,padded_indices
        self.X,self.Y=X,Y
        self.valid_rows=torch.tensor([len(p) for p in point_lists],device=device,dtype=torch.long)
        self.monomials=monomials(degree)
        self.P,self.PR,self.PZ=_monomial_features(X,Y,degree,scale)
        self.T,self.Tp=_trig(self.phis,mtor,nfp),_trig(self.phis,mtor,nfp,derivative=True)
        self.rhs=-2*X*self.features[0]/scale
        self.coefficient_count=len(self.monomials)*(2*mtor+1)-1

    def apply(self,coefficients):
        C=_coefficient_matrix(coefficients,self.degree,self.mtor)
        amplitude=self.T@C.transpose(-1,-2)
        amplitude_phi=self.Tp@C.transpose(-1,-2)
        radial=(self.PR@amplitude[...,None])[...,0]
        vertical=(self.PZ@amplitude[...,None])[...,0]
        toroidal=(self.P@amplitude_phi[...,None])[...,0]
        cr,cz,cp=self.features
        return cr*radial+cz*vertical+cp*toroidal

    def transpose(self,rows):
        if rows.shape!=self.rhs.shape:
            raise ValueError('batch transpose rows require the padded [query,section,row] shape')
        cr,cz,cp=self.features
        radial=(self.PR.transpose(-1,-2)@(cr*rows)[...,None])[...,0]
        vertical=(self.PZ.transpose(-1,-2)@(cz*rows)[...,None])[...,0]
        toroidal=(self.P.transpose(-1,-2)@(cp*rows)[...,None])[...,0]
        C=(radial+vertical).transpose(-1,-2)@self.T+toroidal.transpose(-1,-2)@self.Tp
        return C.flatten(1)[:,1:]

    def residual_rms(self,coefficients):
        residual=self.apply(coefficients)-self.rhs
        return torch.sqrt(residual.square().sum((-2,-1))/self.valid_rows)

    def column_norms(self):
        return _section_column_norms(self.features,self.PR,self.PZ,self.P,self.T,self.Tp)


def _tensor(value,reference,*,dtype=torch.float64):
    return _live_tensor(value,device=reference.device,dtype=dtype).detach()


def _value_with_response(reference,value,reference_value,local_value):
    # A direct subtraction at the center preserves native values while retaining
    # a live derivative. A local-value continuation uses the frozen center's
    # evaluation, so finite differences see the same first-order response.
    return reference+value-(reference_value if local_value else value.detach())


def two_stage_sampling_geometry(stage1: PsiGeometry, metadata: Mapping, *,
                                reference_axis=None, local_value=False):
    """Propagate S1 -> capacity -> inward boundary -> polygon area -> lattice h.

    ``metadata['boundary']`` and ``metadata['sampling']`` are the dictionaries
    returned by ``_two_stage.max_closed_boundary`` / ``sample_frozen_sections``.
    The captured active fold radius can be frozen in the envelope derivative
    because p'(u)=0 there. An R-positive cap instead retains its axis derivative.
    """
    boundary,sampling=metadata['boundary'],metadata['sampling']
    reference=stage1.coefficients
    phi=_tensor(boundary['phis'],reference)
    theta=_tensor(boundary['theta'],reference)
    radii=_tensor(boundary['radii'],reference)
    diagnostics=boundary['diagnostics']
    limiting=diagnostics['limiting_ray']
    ip,it=int(limiting['phi_index']),int(limiting['theta_index'])
    source=diagnostics['active_capacity_source']
    if source not in ('fold','fixed_radius_cap','R_positive_cap'):
        raise ValueError(f'unsupported active capacity source: {source}')
    geometry=PsiGeometry(stage1.axis_samples.double(),reference.double(),stage1.nfp,
                         stage1.a,stage1.degree,stage1.mtor)
    polynomial=geometry.rays(phi[:,None],theta[None,:])
    powers=torch.arange(stage1.degree+1,device=reference.device,dtype=torch.float64)
    active_u=_tensor(diagnostics['active_capacity_normalized_radius'],reference)
    active_radius=active_u*stage1.a
    axis_R,axis_Z,_,_=geometry.axis_eval(phi)
    if source=='R_positive_cap':
        if reference_axis is None:
            polygons=_tensor(boundary['polygons_rz'],reference)
            reference_R=(polygons[:,:,0]-radii*theta.cos()).mean(-1)
        else:
            axis=_tensor(reference_axis,reference)
            reference_R,_=periodic_hermite(phi,axis[:,0],axis[:,2],stage1.nfp)
        response=(axis_R[ip]-reference_R[ip])/(-theta[it].cos())
        if not local_value:
            response=response-response.detach()
        active_radius=active_radius+response
    capacity=(polynomial[ip,it]*active_radius**powers).sum()
    margin=float(diagnostics.get('inward_s_level_fraction',1e-6))
    level_ref=_tensor(boundary['level'],reference)
    capacity_ref=level_ref/(1-margin)
    level=(1-margin)*_value_with_response(capacity_ref,capacity,capacity_ref,local_value)
    # Reference polynomial is in normalized u; convert to physical-radius
    # coefficients for the simple-root denominator and center residual.
    reference_coeff=metadata.get('stage1_reference_coefficients',metadata['stage1_coefficients'])
    reference_geom=PsiGeometry(geometry.axis_samples.detach(),_tensor(reference_coeff,reference),
                              stage1.nfp,stage1.a,stage1.degree,stage1.mtor)
    captured_polynomial=reference_geom.rays(phi[:,None],theta[None,:])
    p_ref=(captured_polynomial*radii[...,None]**powers).sum(-1)
    derivative=(captured_polynomial[...,1:]*powers[1:]*radii[...,None]**(powers[1:]-1)).sum(-1)
    if not bool(torch.isfinite(derivative).all() & (derivative.abs()>1e-12).all()):
        raise ValueError('second-fit boundary has a singular radial root')
    residual=(polynomial*radii[...,None]**powers).sum(-1)-level
    center_residual=p_ref-level_ref
    roots=radii-(residual-(center_residual if local_value else residual.detach()))/derivative.detach()
    polygons=torch.stack((axis_R[:,None]+roots*theta.cos(),axis_Z[:,None]+roots*theta.sin()),-1)
    following=torch.roll(polygons,-1,1)
    areas=.5*(polygons[:,:,0]*following[:,:,1]-following[:,:,0]*polygons[:,:,1]).sum(-1)
    if not bool((areas>0).all()):
        raise ValueError('second-fit domain has a nonpositive section area')
    area=areas.sum()
    common=dict(phis=phi,area=area,section_areas=areas,polygons_rz=polygons,radii=roots,level=level,
                capacity=capacity,active_capacity_source=source)
    if sampling.get('kind',sampling.get('metadata',{}).get('kind'))=='polar_quadrature':
        indices=_tensor(sampling['theta_indices'],reference,dtype=torch.long)
        u=_tensor(sampling['radial_u'],reference)
        radial_weights=_tensor(sampling['radial_weights'],reference)
        angles=theta[indices]
        edge=roots[:,indices]
        radius=edge[...,None]*u.sqrt()
        R=axis_R[:,None,None]+radius*angles.cos()[None,:,None]
        Z=axis_Z[:,None,None]+radius*angles.sin()[None,:,None]
        rz=torch.stack((R,Z),-1).reshape(-1,2)
        weights=(edge[...,None].square()*radial_weights).reshape(-1)
        weights=weights/weights.mean()
        section=torch.arange(len(phi),device=reference.device).repeat_interleave(len(indices)*len(u))
        return dict(common,kind='polar_quadrature',rz=rz,section_index=section,row_weights=weights,
                    spacing=None,equivalent_spacing=torch.sqrt(area/len(rz)),
                    derivative_scope='fixed polar nodes and active capacity; live axis/S1/boundary/area weights')
    area_ref=_tensor(diagnostics['section_areas_m2'],reference).sum()
    h_ref=_tensor(sampling['spacing'],reference)
    if sampling['metadata'].get('fixed_spacing',False):
        spacing=h_ref
    else:
        h_raw=h_ref*torch.sqrt(area/area_ref)
        spacing=_value_with_response(h_ref,h_raw,h_ref,local_value)
    indices=torch.as_tensor(sampling['lattice_indices'],device=reference.device,dtype=torch.long).detach()
    shift=_tensor(sampling['lattice_shift'],reference)
    section=torch.as_tensor(sampling['section_index'],device=reference.device,dtype=torch.long).detach()
    rz=spacing*(indices.to(torch.float64)+shift)
    return dict(common,kind='cartesian',rz=rz,section_index=section,spacing=spacing,row_weights=None,
                derivative_scope='fixed active ray, members and count-adjustment branch; live axis/S1/domain/spacing')


@dataclass
class TwoStageFrontResult(FrontResult):
    stage1_geometry: PsiGeometry
    domain: dict


def _axis_local_value(field,trace,captured_start,*,nfp,trace_steps,sample_count):
    """One local periodic-root continuation for meaningful axis FD values."""
    y=captured_start.detach().reshape(1,2).requires_grad_(True)
    end=trace(y,0.,2*math.pi/nfp,trace_steps,return_samples=False)
    jac=torch.stack([torch.autograd.grad(end[0,k],y,retain_graph=True)[0][0] for k in range(2)])
    matrix=torch.eye(2,device=y.device,dtype=y.dtype)-jac.detach()
    condition=torch.linalg.cond(matrix)
    if not bool(torch.isfinite(condition) & (condition<1e10)):
        raise ValueError('periodic magnetic-axis root has singular/ill-conditioned monodromy')
    start=y.detach()[0]+torch.linalg.solve(matrix,(end-y.detach())[0])
    substeps=(trace_steps+sample_count-1)//sample_count
    steps=sample_count*substeps
    states=trace(start[None],0.,2*math.pi/nfp,steps,return_samples=True)
    if states.shape!=(steps+1,1,2):
        raise ValueError('trace(return_samples=True) must include every step and initial point')
    rz=states[:-1:substeps,0]
    phi=torch.arange(sample_count,device=rz.device,dtype=rz.dtype)*(2*math.pi/(nfp*sample_count))
    rhs=cylindrical_rhs(field,rz,phi)
    return torch.cat((rz,rhs),-1),dict(monodromy=jac.detach(),condition=condition.detach(),
        root_residual=torch.linalg.vector_norm(end-y).detach(),axis_local_value=True)


def build_two_stage_front(field,trace,center: Mapping,*,config=None,
                          dtype=torch.float32,local_value=False,value_only=False):
    """Compose both center-preconditioned fits and the differentiable domain.

    S1 factors live under ``center['two_stage']['stage1_*']``. S2 factors and
    target-volume adjoints use the usual top-level ``psi_*`` / ``target_*``
    fields. ``stage1_reference_coefficients`` optionally distinguishes the
    captured S1 solution from a deliberately different PCGLS warm start.
    ``value_only`` detaches the temporary periodic-axis root graph before both
    fits. Supply a field built from non-grad physical tensors for a tape-free
    finite-difference endpoint; numerical values are unchanged.
    """
    cfg=dict(center.get('config',{}));cfg.update(config or {})
    two=center['two_stage']
    third=two.get('target_refit')
    axis_center=center['axis_samples']
    if not isinstance(axis_center,torch.Tensor):
        raise TypeError('center axis_samples must be a device tensor')
    nfp,a=int(cfg.get('nfp',4)),float(cfg['psi_a'])
    degree,mtor=int(cfg.get('psi_poly_degree',10)),int(cfg.get('psi_m_tor',12))
    axis_builder=_axis_local_value if local_value else differentiable_axis
    with torch.enable_grad():
        axis,diagnostics=axis_builder(field,trace,axis_center[0,:2],nfp=nfp,
            trace_steps=int(cfg.get('axis_trace_steps',240)),sample_count=int(cfg.get('axis_sample_count',240)))
    if value_only:
        axis=axis.detach()
    op1=TensorProductPsiOperator(field,axis,nfp=nfp,a=a,degree=degree,mtor=mtor,
        radial_grid=int(cfg.get('psi_n_r',48)),vertical_grid=int(cfg.get('psi_n_z',48)),
        phi_grid=int(cfg.get('psi_n_phi',48)),rho_min=float(cfg.get('psi_rho_min',.002)),dtype=dtype)
    c1,rms1=four_step_pcgls(op1,two['stage1_coefficients'],two['stage1_scale'],two['stage1_upper'],
        ridge=float(cfg.get('psi_ridge',1e-6)),iterations=int(cfg.get('psi_iterations',4)),
        live_column_scale=bool(cfg.get('psi_live_column_scale',True)))
    geometry1=PsiGeometry(axis,c1,nfp,a,degree,mtor)
    domain=two_stage_sampling_geometry(geometry1,two,reference_axis=axis_center,local_value=local_value)
    op2=SectionPsiOperator(field,axis,domain['rz'],domain['phis'],domain['section_index'],
        nfp=nfp,a=a,degree=degree,mtor=mtor,dtype=dtype,row_weights=domain['row_weights'])
    stage2=({key:third['stage2_'+key] for key in ('coefficients','scale','upper')}
            if third is not None else {key:center['psi_'+key] for key in ('coefficients','scale','upper')})
    coefficients,_=four_step_pcgls(op2,stage2['coefficients'],stage2['scale'],stage2['upper'],
        ridge=float(cfg['psi_refit_ridge'] if cfg.get('psi_refit_ridge') is not None else cfg.get('psi_ridge',1e-6)),
        iterations=int(cfg.get('psi_iterations',4)),
        live_column_scale=bool(cfg.get('psi_live_column_scale',True)))
    final_operator=op2
    if third is not None:
        from .target_domain import target_sampling_geometry
        target_domain=target_sampling_geometry(PsiGeometry(axis,coefficients,nfp,a,degree,mtor),
            third,reference_axis=axis_center,local_value=local_value)
        op3=SectionPsiOperator(field,axis,target_domain['rz'],target_domain['phis'],target_domain['section_index'],
            nfp=nfp,a=a,degree=degree,mtor=mtor,dtype=dtype,row_weights=target_domain['row_weights'])
        coefficients,_=four_step_pcgls(op3,center['psi_coefficients'],center['psi_scale'],center['psi_upper'],
            ridge=float(cfg['psi_target_refit_ridge']),iterations=int(cfg.get('psi_iterations',4)),
            live_column_scale=bool(cfg.get('psi_live_column_scale',True)))
        final_operator=op3
        diagnostics.update(third_fit_point_count=op3.point_count,
            third_fit_column_scale_relative_max=(op3.column_scale_ratio-1).abs().max().detach(),
            target_fit_domain_derivatives=target_domain['derivative_scope'])
    geometry=PsiGeometry(axis,coefficients,nfp,a,degree,mtor,
        maximum_radius_scale=float(cfg.get('surface_max_radius_scale',1.)),
        root_iterations=int(cfg.get('surface_newton_iters',20)),
        root_tolerance=float(cfg.get('surface_newton_tolerance',1e-12)),
        root_library=getattr(field, 'library', None))
    diagnostics.update(stage1_psi_train_rms=rms1.detach(),
                       stage1_column_scale_relative_max=(op1.column_scale_ratio-1).abs().max().detach(),
                       psi_column_scale_relative_max=(op2.column_scale_ratio-1).abs().max().detach(),
                       second_fit_spacing=None if domain['spacing'] is None else domain['spacing'].detach(),
                       second_fit_point_count=op2.point_count,
                       second_fit_padded_rows=op2.rhs.numel(),
                       second_fit_sampling=domain['kind'],
                       two_stage_domain_derivatives=domain['derivative_scope'])
    return TwoStageFrontResult(geometry,predict_target_level(geometry,center),
                               final_operator.residual_rms(coefficients),diagnostics,final_operator,geometry1,domain)


def _batch_axis_local_value(field,captured_starts,*,nfp,trace_steps,sample_count):
    """Simultaneous independent periodic-root continuations for local FD."""
    y=captured_starts.detach()[:,None,:].requires_grad_()
    end=field.trace(y,0.,2*math.pi/nfp,trace_steps,return_samples=False)
    jac=torch.stack([torch.autograd.grad(end[:,0,k].sum(),y,retain_graph=True)[0][:,0]
                     for k in range(2)],1)
    matrix=torch.eye(2,device=y.device,dtype=y.dtype)[None]-jac.detach()
    condition=torch.linalg.cond(matrix)
    valid=torch.isfinite(condition)&(condition<1e10)
    if not bool(valid.all()):
        raise ValueError('one batch periodic magnetic-axis root is singular/ill-conditioned')
    correction=torch.linalg.solve(matrix,(end-y.detach())[:,0,:,None])[...,0]
    starts=y.detach()[:,0]+correction
    substeps=(trace_steps+sample_count-1)//sample_count
    total_steps=sample_count*substeps
    history=field.trace(starts[:,None],0.,2*math.pi/nfp,total_steps,return_samples=True)
    if history.shape!=(total_steps+1,len(starts),1,2):
        raise ValueError('batch trace history must include every step and initial points')
    rz=history[:-1:substeps,:,0].transpose(0,1)
    phi=torch.arange(sample_count,device=y.device,dtype=y.dtype)*(2*math.pi/(nfp*sample_count))
    R,Z=rz.unbind(-1);c,s=phi.cos(),phi.sin()
    points=torch.stack((R*c,R*s,Z),-1)
    B=field(points,gradient=False).to(rz.dtype)
    br=B[...,0]*c+B[...,1]*s;bp=-B[...,0]*s+B[...,1]*c
    floor=torch.where(bp<0,-torch.ones_like(bp),torch.ones_like(bp))*1e-14
    denominator=torch.where(bp.abs()>1e-14,bp,floor)
    slopes=torch.stack((R*br/denominator,R*B[...,2]/denominator),-1)
    return torch.cat((rz,slopes),-1),dict(monodromy=jac.detach(),condition=condition.detach(),
        root_residual=torch.linalg.vector_norm(end-y,dim=-1)[:,0].detach(),
        valid=valid.detach(),axis_local_value=True)


@dataclass
class BatchTwoStageFrontResult(BatchFrontResult):
    stage1_geometries:list
    domains:list


def build_batch_two_stage_front(field,centers,*,config_overrides=None,
                                dtype=torch.float32,local_value=False,value_only=False,nfp=4):
    """Batched axis tracing, both fits and independent moving S2 domains.

    Group common grid/iteration settings before calling. Fitting scales,
    irregular point membership, factorization snapshots and target adjoints
    are independent per query. Bound memory with query chunks (typically 8-16)
    in the caller, rather than serializing hundreds of finite-difference rays.
    With non-grad physical parameters, ``value_only`` discards the temporary
    periodic-axis Jacobian tape before either fit, including local-value FD.
    """
    if not centers or len(centers)!=field.positions.shape[0]:
        raise ValueError('one independent center required per batch field query')
    cfgs=[{**c.get('config',{}),**(config_overrides or {})} for c in centers]
    thirds=[c['two_stage'].get('target_refit') for c in centers]
    if any(t is not None for t in thirds) and not all(t is not None for t in thirds):
        raise ValueError('group two-stage and target-refit captures separately')
    defaults=dict(axis_trace_steps=240,axis_sample_count=240,psi_poly_degree=10,psi_m_tor=12,
        psi_n_r=48,psi_n_z=48,psi_n_phi=48,psi_iterations=4,psi_live_column_scale=True,surface_max_radius_scale=1.,
        surface_newton_iters=20,surface_newton_tolerance=1e-12)
    common={}
    for name,default in defaults.items():
        values=[c.get(name,default) for c in cfgs]
        if any(value!=values[0] for value in values[1:]):
            raise ValueError(f'group compatible two-stage batches first: differing {name}')
        common[name]=values[0]
    if any(int(c.get('nfp',nfp))!=nfp for c in cfgs):
        raise ValueError('batch field and all center configs must share nfp')
    device=field.positions.device
    def stack(name):
        return _stack_live([c[name] for c in centers],device=device,dtype=torch.float64)
    def stack_stage1(name):
        return _stack_live([c['two_stage']['stage1_'+name] for c in centers],
                          device=device,dtype=torch.float64)
    starts=stack('axis_samples')[:,0,:2]
    axis_builder=_batch_axis_local_value if local_value else differentiable_batch_axis
    with torch.enable_grad():
        axes,diagnostics=axis_builder(field,starts,nfp=nfp,trace_steps=int(common['axis_trace_steps']),
                                      sample_count=int(common['axis_sample_count']))
    if value_only:
        axes=axes.detach()
    radii=[float(c['psi_a']) for c in cfgs]
    ridge=[float(c.get('psi_ridge',1e-6)) for c in cfgs]
    degree,mtor=int(common['psi_poly_degree']),int(common['psi_m_tor'])
    op1=BatchTensorProductPsiOperator(field,axes,radii,nfp=nfp,degree=degree,mtor=mtor,
        radial_grid=int(common['psi_n_r']),vertical_grid=int(common['psi_n_z']),
        phi_grid=int(common['psi_n_phi']),rho_min=[float(c.get('psi_rho_min',.002)) for c in cfgs],dtype=dtype)
    coefficients1,rms1=batch_four_step_pcgls(op1,stack_stage1('coefficients'),stack_stage1('scale'),
        stack_stage1('upper'),ridge=ridge,iterations=int(common['psi_iterations']),
        live_column_scale=bool(common['psi_live_column_scale']))
    geometries1=[PsiGeometry(axes[i],coefficients1[i],nfp,radii[i],degree,mtor) for i in range(len(centers))]
    domains=[two_stage_sampling_geometry(g,c['two_stage'],reference_axis=c['axis_samples'],local_value=local_value)
             for g,c in zip(geometries1,centers)]
    op2=BatchSectionPsiOperator(field,axes,[d['rz'] for d in domains],[d['phis'] for d in domains],
        [d['section_index'] for d in domains],radii,nfp=nfp,degree=degree,mtor=mtor,dtype=dtype,
        row_weights=[d['row_weights'] for d in domains])
    def stack_stage2(name):
        return (_stack_live([t['stage2_'+name] for t in thirds],device=device,dtype=torch.float64)
                if thirds[0] is not None else stack('psi_'+name))
    coefficients,_=batch_four_step_pcgls(op2,stack_stage2('coefficients'),stack_stage2('scale'),stack_stage2('upper'),
        ridge=[float(c['psi_refit_ridge'] if c.get('psi_refit_ridge') is not None else c.get('psi_ridge',1e-6)) for c in cfgs],
        iterations=int(common['psi_iterations']),
        live_column_scale=bool(common['psi_live_column_scale']))
    final_operator=op2
    if thirds[0] is not None:
        from .target_domain import target_sampling_geometry
        target_domains=[target_sampling_geometry(PsiGeometry(axes[i],coefficients[i],nfp,radii[i],degree,mtor),
            thirds[i],reference_axis=centers[i]['axis_samples'],local_value=local_value) for i in range(len(centers))]
        op3=BatchSectionPsiOperator(field,axes,[d['rz'] for d in target_domains],[d['phis'] for d in target_domains],
            [d['section_index'] for d in target_domains],radii,nfp=nfp,degree=degree,mtor=mtor,dtype=dtype,
            row_weights=[d['row_weights'] for d in target_domains])
        coefficients,_=batch_four_step_pcgls(op3,stack('psi_coefficients'),stack('psi_scale'),stack('psi_upper'),
            ridge=[float(c['psi_target_refit_ridge']) for c in cfgs],iterations=int(common['psi_iterations']),
            live_column_scale=bool(common['psi_live_column_scale']))
        final_operator=op3
        diagnostics.update(third_fit_point_count=op3.valid_rows.detach(),
            third_fit_column_scale_relative_max=(op3.column_scale_ratio-1).abs().amax(-1).detach(),
            target_fit_domain_derivatives=[d['derivative_scope'] for d in target_domains])
    geometries=[PsiGeometry(axes[i],coefficients[i],nfp,radii[i],degree,mtor,
        maximum_radius_scale=float(common['surface_max_radius_scale']),
        root_iterations=int(common['surface_newton_iters']),
        root_tolerance=float(common['surface_newton_tolerance']),
        root_library=getattr(field, 'library', None)) for i in range(len(centers))]
    levels=torch.stack([predict_target_level(g,c) for g,c in zip(geometries,centers)])
    diagnostics.update(stage1_psi_train_rms=rms1.detach(),
        stage1_column_scale_relative_max=(op1.column_scale_ratio-1).abs().amax(-1).detach(),
        psi_column_scale_relative_max=(op2.column_scale_ratio-1).abs().amax(-1).detach(),second_fit_point_count=op2.valid_rows.detach(),
        second_fit_padded_rows=op2.rhs.shape[1]*op2.rhs.shape[2],
        second_fit_spacing=(torch.stack([d['spacing'] for d in domains]).detach()
                            if all(d['spacing'] is not None for d in domains) else
                            [None if d['spacing'] is None else d['spacing'].detach() for d in domains]),
        second_fit_sampling=[d['kind'] for d in domains],
        two_stage_domain_derivatives=[d['derivative_scope'] for d in domains])
    return BatchTwoStageFrontResult(axes,coefficients,levels,final_operator.residual_rms(coefficients),
                                    diagnostics,final_operator,geometries,geometries1,domains)
