"""Area quadrature and moving-domain weighted least-squares derivatives."""
from dataclasses import replace
import math

import numpy as np
import pytest
torch=pytest.importorskip('torch')

from coil_eval import EvaluationConfig
from coil_eval._two_stage.quadrature import sample_polar_quadrature
from coil_eval._two_stage.model import PsiModel
from coil_eval._two_stage.boundary import max_closed_boundary
from coil_eval._autograd.front import PsiGeometry
from coil_eval._autograd.two_stage_front import (
    SectionPsiOperator, BatchSectionPsiOperator, two_stage_sampling_geometry,
    build_batch_two_stage_front, build_two_stage_front,
)
from test_two_stage_front import axis, field, make_domain, folded_coefficients
from test_two_stage_batch_front import ToyField, factors, toy_centers

DT=torch.float64


def circular_boundary():
    phi=np.arange(4)*2*np.pi/(4*4);theta=np.arange(256)*2*np.pi/256
    radius=np.array([.11,.14,.09,.16])
    R=np.array([1.,1.1,.98,1.05]);Z=np.array([.02,-.03,.01,.04])
    radii=np.broadcast_to(radius[:,None],(4,256)).copy()
    polygons=np.stack((R[:,None]+radii*np.cos(theta),Z[:,None]+radii*np.sin(theta)),-1)
    return dict(phis=phi,theta=theta,radii=radii,polygons_rz=polygons),radius,R,Z


def test_polar_quadrature_integrates_cartesian_area_moments_and_keeps_section_weights():
    boundary,radii,R,Z=circular_boundary()
    sample=sample_polar_quadrature(boundary,target_count=2575)
    assert sample['rz'].shape==(2560,2) and sample['metadata']['radial_count']==5
    assert sample['spacing'] is None and sample['equivalent_spacing_m']>0
    assert sample['lattice_indices'].shape==(0,2)
    weights=sample['row_weights'];section=sample['section_index'];points=sample['rz']
    assert (weights>0).all() and weights.mean()==pytest.approx(1.,abs=1e-15)
    area=np.pi*radii**2;expected_R=np.dot(area,R)/area.sum()
    assert np.mean(weights*points[:,0])==pytest.approx(expected_R,abs=1e-14)
    expected_second=np.sum(area*radii**2/4)/area.sum()
    assert np.mean(weights*(points[:,0]-R[section])**2)==pytest.approx(expected_second,abs=1e-15)
    assert sample['cell_volume_weights_full'].sum()==pytest.approx(2*np.pi*np.mean(area*R),rel=2e-14)
    section_weight=np.array([weights[section==i].sum() for i in range(4)])
    np.testing.assert_allclose(section_weight/section_weight.sum(),area/area.sum(),rtol=1e-14)
    assert sample['radial_u'].min()>0 and sample['radial_u'].max()<1
    assert not sample['row_weights'].flags.writeable


def test_sampling_config_default_and_input_guards():
    config=EvaluationConfig(.05,psi_refit_sampling='cartesian')
    assert config.psi_refit_sampling=='cartesian'
    assert replace(config,psi_refit_sampling='polar_quadrature').to_dict()['psi_refit_sampling']=='polar_quadrature'
    with pytest.raises(ValueError,match='psi_refit_sampling'):
        replace(config,psi_refit_sampling='unknown')
    boundary,*_=circular_boundary()
    with pytest.raises(ValueError,match='divisor'):
        sample_polar_quadrature(boundary,theta_count=72)
    with pytest.raises(ValueError,match='target_count'):
        sample_polar_quadrature(boundary,target_count=0)


def test_weighted_section_operator_matches_independent_row_scaled_design_and_weight_vjp():
    phi=torch.tensor([0.,.2,.6],dtype=DT);section=torch.tensor([0,2,1,0,0,1,2])
    points=torch.tensor([[.98,.02],[1.04,-.03],[.95,.01],[1.06,.04],[1.02,-.01],[.99,.03],[1.03,-.04]],dtype=DT)
    weights=torch.linspace(.4,1.6,len(points),dtype=DT,requires_grad=True)
    unweighted=SectionPsiOperator(field,axis(),points,phi,section,degree=4,mtor=1,dtype=DT)
    weighted=SectionPsiOperator(field,axis(),points,phi,section,degree=4,mtor=1,dtype=DT,row_weights=weights)
    coefficients=torch.linspace(-.01,.01,weighted.coefficient_count,dtype=DT)
    root=weights[weighted.padded_indices].sqrt()
    torch.testing.assert_close(weighted.apply(coefficients),unweighted.apply(coefficients)*root)
    torch.testing.assert_close(weighted.rhs,unweighted.rhs*root)
    rows=torch.arange(weighted.rhs.numel(),dtype=DT).reshape_as(weighted.rhs)/10
    torch.testing.assert_close(weighted.transpose(rows),unweighted.transpose(rows*root))
    residual=unweighted.apply(coefficients)-unweighted.rhs
    square=(residual.square()*weights[weighted.padded_indices])[weighted.mask].sum()
    torch.testing.assert_close(weighted.residual_rms(coefficients),torch.sqrt(square/len(points)))
    actual,=torch.autograd.grad((weighted.apply(coefficients)-weighted.rhs).square().sum(),weights)
    expected=torch.zeros_like(weights)
    expected[weighted.padded_indices[weighted.mask]]=residual[weighted.mask].square()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='positive'):
        SectionPsiOperator(field,axis(),points,phi,section,dtype=DT,row_weights=torch.zeros(len(points)))


def test_polar_live_axis_boundary_and_normalized_weights_vjp_match_full_numpy_domain_rebuild():
    coefficients=folded_coefficients();samples=axis().numpy()
    metadata=make_domain(coefficients,samples)
    metadata['sampling']=sample_polar_quadrature(metadata['boundary'],target_count=480,theta_count=16)
    generator=np.random.default_rng(21)
    direction=generator.normal(size=len(coefficients))*.003
    axis_direction=np.zeros_like(samples)
    phi=np.arange(len(samples))*2*np.pi/(4*len(samples))
    axis_direction[:,0]=.03+.02*np.cos(4*phi)
    axis_direction[:,1]=.01*np.sin(4*phi)
    axis_direction[:,2]=-.08*np.sin(4*phi);axis_direction[:,3]=.04*np.cos(4*phi)
    def evaluate(t):
        geometry=PsiGeometry(torch.tensor(samples)+t*torch.tensor(axis_direction),
            torch.tensor(coefficients)+t*torch.tensor(direction),a=.15,degree=4,mtor=1)
        return two_stage_sampling_geometry(geometry,metadata,reference_axis=torch.tensor(samples))
    t=torch.tensor(0.,dtype=DT,requires_grad=True);domain=evaluate(t)
    sample=metadata['sampling']
    torch.testing.assert_close(domain['rz'],torch.tensor(sample['rz']),rtol=1e-14,atol=2e-15)
    torch.testing.assert_close(domain['row_weights'],torch.tensor(sample['row_weights']),rtol=2e-14,atol=2e-14)
    assert domain['row_weights'].requires_grad and domain['rz'].requires_grad
    def torch_value(d):
        R,Z=d['rz'].unbind(-1)
        return (d['row_weights']*(torch.exp(R)+.3*torch.sin(7*Z)+R*Z)).mean()
    actual,=torch.autograd.grad(torch_value(domain),t,retain_graph=True)
    weight_response,=torch.autograd.grad((domain['row_weights']*torch.arange(len(sample['rz']),dtype=DT)).sum(),t)
    assert abs(float(weight_response))>1e-4
    def rebuilt(value):
        model=PsiModel(coefficients+value*direction,samples+value*axis_direction,a=.15,degree=4,mtor=1)
        boundary=max_closed_boundary(model,metadata['boundary']['phis'],ntheta=32,max_radius=.15)
        s=sample_polar_quadrature(boundary,target_count=480,theta_count=16)
        R,Z=s['rz'].T
        return np.mean(s['row_weights']*(np.exp(R)+.3*np.sin(7*Z)+R*Z))
    h=1e-6;expected=(rebuilt(h)-rebuilt(-h))/(2*h)
    torch.testing.assert_close(actual,torch.tensor(expected),rtol=2e-5,atol=2e-8)
    assert abs(float(actual))>1e-4


def polar_centers():
    centers=toy_centers();base=ToyField(torch.zeros((2,2),dtype=DT))
    for q,center in enumerate(centers):
        metadata=center['two_stage'];sample=sample_polar_quadrature(metadata['boundary'],480,theta_count=16)
        metadata['sampling']=sample
        magnetic,_=base.single(q)
        op=SectionPsiOperator(magnetic,center['axis_samples'],torch.tensor(sample['rz']),
            torch.tensor(metadata['boundary']['phis']),sample['section_index'],a=center['config']['psi_a'],
            degree=2,mtor=0,dtype=DT,row_weights=torch.tensor(sample['row_weights']))
        center['psi_scale'],center['psi_upper']=factors(op)
    return centers


def polar_losses(front):
    values=[]
    for q,d in enumerate(front.domains):
        values.append(front.coefficients[q].square().sum()+.3*front.psi_train_rms[q].square()+
            .02*(d['row_weights']*d['rz'][:,0].square()).mean()+.01*front.axis_samples[q,:,0].mean())
    return torch.stack(values)


def test_weighted_batched_two_fit_builder_matches_single_vjp_and_local_fd_without_cross_query_gradients():
    centers=polar_centers();parameters=torch.zeros((2,2),dtype=DT,requires_grad=True)
    field_batch=ToyField(parameters);front=build_batch_two_stage_front(field_batch,centers,dtype=DT)
    values=polar_losses(front)
    gradients,=torch.autograd.grad(values.sum(),parameters,retain_graph=True)
    isolated,=torch.autograd.grad(values[0],parameters)
    assert not isolated[1].count_nonzero()
    single_parameters=parameters.detach().requires_grad_();single_field=ToyField(single_parameters)
    single_values=[]
    for q,center in enumerate(centers):
        magnetic,trace=single_field.single(q)
        one=build_two_stage_front(magnetic,trace,center,dtype=DT)
        torch.testing.assert_close(front.coefficients[q],one.geometry.coefficients,rtol=2e-9,atol=2e-10)
        d=one.domain
        single_values.append(one.geometry.coefficients.square().sum()+.3*one.psi_train_rms.square()+
            .02*(d['row_weights']*d['rz'][:,0].square()).mean()+.01*one.geometry.axis_samples[:,0].mean())
    expected,=torch.autograd.grad(torch.stack(single_values).sum(),single_parameters)
    torch.testing.assert_close(gradients,expected,rtol=2e-6,atol=2e-9)
    h=1e-5;numeric=torch.zeros_like(parameters)
    for q in range(2):
        for k in range(2):
            delta=torch.zeros_like(parameters);delta[q,k]=h
            def local(p):
                return polar_losses(build_batch_two_stage_front(ToyField(p),centers,dtype=DT,
                    local_value=True,value_only=True))[q]
            numeric[q,k]=(local(parameters.detach()+delta)-local(parameters.detach()-delta))/(2*h)
    torch.testing.assert_close(gradients,numeric,rtol=2e-4,atol=2e-7)


def test_batch_weighted_operator_independent_weights_and_padding_adjoint():
    points=[torch.tensor([[.98,.02],[1.04,-.03],[.95,.01],[1.06,.04]],dtype=DT),
            torch.tensor([[1.08,.03],[1.05,-.01],[1.07,.02]],dtype=DT)]
    phis=[torch.tensor([0.,.3],dtype=DT)]*2
    sections=[torch.tensor([0,1,0,1]),torch.tensor([0,1,0])]
    weights=[torch.tensor([.7,1.1,1.3,.9],dtype=DT,requires_grad=True),
             torch.tensor([.4,1.8,.8],dtype=DT,requires_grad=True)]
    class BatchField:
        def __call__(self,p,gradient=False):return field(p)
    axes=torch.stack((axis(),axis(R=1.06)))
    op=BatchSectionPsiOperator(BatchField(),axes,points,phis,sections,[.15,.12],degree=4,mtor=1,
        dtype=DT,row_weights=weights)
    coefficients=torch.linspace(-.002,.002,op.coefficient_count,dtype=DT).expand(2,-1)
    rows=torch.arange(op.rhs.numel(),dtype=DT).reshape_as(op.rhs)
    for q in range(2):
        single=SectionPsiOperator(field,axes[q],points[q],phis[q],sections[q],a=[.15,.12][q],
            degree=4,mtor=1,dtype=DT,row_weights=weights[q])
        S,K=single.rhs.shape
        torch.testing.assert_close(op.apply(coefficients)[q,:S,:K],single.apply(coefficients[q]))
        torch.testing.assert_close(op.transpose(rows)[q],single.transpose(rows[q,:S,:K]))
    ga,gb=torch.autograd.grad(op.apply(coefficients)[0].square().sum(),weights)
    assert ga.abs().sum()>0 and gb.count_nonzero()==0


def test_mixed_cartesian_and_polar_queries_match_independent_builders():
    centers=toy_centers();centers[1]=polar_centers()[1]
    parameters=torch.zeros((2,2),dtype=DT,requires_grad=True)
    batch_field=ToyField(parameters);batch=build_batch_two_stage_front(batch_field,centers,dtype=DT)
    assert batch.domains[0]['kind']=='cartesian' and batch.domains[1]['kind']=='polar_quadrature'
    assert batch.axis_diagnostics['second_fit_spacing'][1] is None
    for q in range(2):
        magnetic,trace=batch_field.single(q)
        one=build_two_stage_front(magnetic,trace,centers[q],dtype=DT)
        torch.testing.assert_close(batch.coefficients[q],one.geometry.coefficients,rtol=2e-9,atol=2e-10)


def test_polar_value_only_has_identical_values_and_no_temporary_axis_graph():
    centers=polar_centers();parameters=torch.zeros((2,2),dtype=DT)
    normal=build_batch_two_stage_front(ToyField(parameters),centers,dtype=DT,local_value=True)
    value=build_batch_two_stage_front(ToyField(parameters),centers,dtype=DT,local_value=True,value_only=True)
    torch.testing.assert_close(value.coefficients,normal.coefficients,rtol=0,atol=0)
    for q in range(2):
        for name in ('rz','row_weights','radii','area'):
            torch.testing.assert_close(value.domains[q][name],normal.domains[q][name],rtol=0,atol=0)
            assert not value.domains[q][name].requires_grad and value.domains[q][name].grad_fn is None
    assert not value.coefficients.requires_grad and not value.psi_train_rms.requires_grad
