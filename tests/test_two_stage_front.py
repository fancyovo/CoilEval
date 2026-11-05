"""Matrix-free second fit and complete moving-domain local derivatives."""
import math

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from coil_eval._autograd.front import PsiGeometry,TensorProductPsiOperator,four_step_pcgls,native_modes
from coil_eval._autograd.two_stage_front import SectionPsiOperator,build_two_stage_front,two_stage_sampling_geometry
from coil_eval._two_stage.model import PsiModel
from coil_eval._two_stage.boundary import max_closed_boundary
from coil_eval._two_stage.sampling import sample_frozen_sections


DT=torch.float64


def axis(count=12,R=1.):
    phi=torch.arange(count,dtype=DT)*2*math.pi/(4*count)
    return torch.stack((R+.012*torch.cos(4*phi),.008*torch.sin(4*phi),
                        -.048*torch.sin(4*phi),.032*torch.cos(4*phi)),-1)


def field(xyz,gradient=False):
    x,y,z=xyz.unbind(-1)
    return torch.stack((.7+.1*x+.03*z*z,.3+.2*y+.04*x*z,.2+.1*z+.02*x*x),-1)


def test_irregular_sections_match_full_1574_column_design_and_adjoint():
    generator=torch.Generator().manual_seed(728)
    phi=torch.tensor([.02,.23,.48],dtype=DT)
    section=torch.tensor([2,0,2,1,0,1,2,0,0,2,0])
    rz=torch.stack((1+.06*torch.randn(len(section),generator=generator,dtype=DT),
                       .04*torch.randn(len(section),generator=generator,dtype=DT)),-1)
    op=SectionPsiOperator(field,axis(),rz,phi,section,dtype=DT)
    assert op.coefficient_count==1574
    X,Y=op.X,op.Y
    cr,cz,cp=op.features
    columns=[]
    for a,b,m,kind in native_modes():
        phase=4*m*phi[:,None]
        trig=phase.cos() if kind==0 else phase.sin()
        derivative=(-phase.sin() if kind==0 else phase.cos())*(4*m)
        columns.append((cr*a*X**max(a-1,0)*Y**b+cz*b*X**a*Y**max(b-1,0))*trig/.15
                       +cp*X**a*Y**b*derivative)
    dense=torch.stack(columns,-1).reshape(-1,1574)
    coefficients=torch.randn(1574,generator=generator,dtype=DT)
    rows=torch.randn(op.rhs.shape,generator=generator,dtype=DT)
    torch.testing.assert_close(op.apply(coefficients).reshape(-1),dense@coefficients,rtol=3e-12,atol=2e-10)
    torch.testing.assert_close(op.transpose(rows),dense.T@rows.reshape(-1),rtol=3e-12,atol=2e-10)
    assert not torch.count_nonzero(op.apply(coefficients)[~op.mask])
    # Each row represents one real collocation point; padding cannot dilute RMS.
    expected=(op.apply(coefficients)-op.rhs)[op.mask].square().mean().sqrt()
    torch.testing.assert_close(op.residual_rms(coefficients),expected)


def test_irregular_point_axis_and_field_motion_all_reach_vjp():
    phi=torch.tensor([0.,.31,.72],dtype=DT)
    section=torch.tensor([0,1,2,2,0,1,0])
    base=torch.tensor([[.94,.02],[1.05,-.03],[1.02,.05],[.98,-.02],[1.08,.01],[.96,.03],[1.03,-.04]],dtype=DT)
    coefficients=torch.linspace(-.002,.002,1574,dtype=DT)
    directions=torch.zeros_like(axis());directions[:,0]=.3;directions[:,2]=torch.linspace(-.1,.1,12)
    def value(parameter):
        def magnetic(points,gradient=False):
            return field(points)*(1+.2*parameter)
        op=SectionPsiOperator(magnetic,axis()+parameter*directions,
            base*(1+.1*parameter),phi,section,dtype=DT)
        return (op.apply(coefficients)-op.rhs).square().sum()
    parameter=torch.tensor(.015,dtype=DT,requires_grad=True)
    actual,=torch.autograd.grad(value(parameter),parameter)
    h=1e-6
    expected=(value(parameter.detach()+h)-value(parameter.detach()-h))/(2*h)
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-6)
    assert abs(float(actual))>1e-4


def make_domain(coefficients,samples,*,a=.15,degree=4,mtor=1,fixed_spacing=False):
    phi=np.arange(5)*2*np.pi/(4*5)
    model=PsiModel(coefficients,samples,nfp=4,a=a,degree=degree,mtor=mtor)
    boundary=max_closed_boundary(model,phi,ntheta=32,max_radius=a)
    sampling=sample_frozen_sections(boundary['polygons_rz'],phi,target_count=480,
        count_tolerance=.15,nfp=4)
    if fixed_spacing:
        sampling=sample_frozen_sections(boundary['polygons_rz'],phi,target_count=480,
            count_tolerance=.15,nfp=4,spacing=sampling['spacing'])
    return dict(stage1_coefficients=coefficients,boundary=boundary,sampling=sampling)


def folded_coefficients():
    modes=native_modes(degree=4,mtor=1)
    coefficients=np.zeros(len(modes))
    coefficients[modes.index((0,2,0,0))]=1.3
    coefficients[modes.index((3,0,0,0))]=.04
    coefficients[modes.index((4,0,0,0))]=-.8
    coefficients[modes.index((4,0,1,0))]=-.08
    return coefficients


def test_fold_capacity_boundary_area_and_spacing_match_rebuilt_domain_derivative():
    coefficients=folded_coefficients()
    samples=axis().numpy()
    metadata=make_domain(coefficients,samples)
    assert metadata['boundary']['diagnostics']['active_capacity_source']=='fold'
    direction=np.zeros_like(coefficients)
    direction[native_modes(degree=4,mtor=1).index((0,2,0,0))]=.6
    direction[native_modes(degree=4,mtor=1).index((4,0,0,0))]=.2
    parameter=torch.tensor(0.,dtype=DT,requires_grad=True)
    geometry=PsiGeometry(torch.tensor(samples),torch.tensor(coefficients)+parameter*torch.tensor(direction),a=.15,degree=4,mtor=1)
    domain=two_stage_sampling_geometry(geometry,metadata,reference_axis=torch.tensor(samples))
    captured=metadata['sampling']
    torch.testing.assert_close(domain['rz'],torch.tensor(captured['rz']),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(domain['radii'],torch.tensor(metadata['boundary']['radii']),rtol=1e-14,atol=1e-14)
    derivative,=torch.autograd.grad(domain['spacing'],parameter)
    h=1e-6
    def rebuilt(t):
        model=PsiModel(coefficients+t*direction,samples,a=.15,degree=4,mtor=1)
        boundary=max_closed_boundary(model,metadata['boundary']['phis'],ntheta=32,max_radius=.15)
        area=np.sum(boundary['diagnostics']['section_areas_m2'])
        reference_area=np.sum(metadata['boundary']['diagnostics']['section_areas_m2'])
        return captured['spacing']*np.sqrt(area/reference_area)
    expected=(rebuilt(h)-rebuilt(-h))/(2*h)
    torch.testing.assert_close(derivative,torch.tensor(expected),rtol=2e-5,atol=2e-9)
    assert abs(float(derivative))>1e-5


def test_fixed_branch_local_value_matches_analytic_domain_vjp():
    coefficients=folded_coefficients();samples=axis().numpy()
    metadata=make_domain(coefficients,samples)
    generator=torch.Generator().manual_seed(29)
    direction=torch.randn(len(coefficients),generator=generator,dtype=DT)*.002
    point_weights=torch.randn(metadata['sampling']['rz'].shape,generator=generator,dtype=DT)
    def value(parameter,local=False):
        geometry=PsiGeometry(torch.tensor(samples),torch.tensor(coefficients)+parameter*direction,a=.15,degree=4,mtor=1)
        domain=two_stage_sampling_geometry(geometry,metadata,reference_axis=torch.tensor(samples),local_value=local)
        return (domain['rz']*point_weights).sum()+.03*domain['area']+.02*domain['level']
    parameter=torch.tensor(0.,dtype=DT,requires_grad=True)
    actual,=torch.autograd.grad(value(parameter),parameter)
    h=1e-6
    expected=(value(torch.tensor(h,dtype=DT),True)-value(torch.tensor(-h,dtype=DT),True))/(2*h)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-8)
    assert abs(float(actual))>1e-5


def test_R_positive_capacity_includes_axis_derivative():
    modes=native_modes(degree=2,mtor=0)
    coefficients=np.zeros(len(modes));coefficients[modes.index((0,2,0,0))]=1.
    samples=axis(R=.11).numpy();samples[:,1]=0.;samples[:,3]=0.
    metadata=make_domain(coefficients,samples,a=.3,degree=2,mtor=0)
    assert metadata['boundary']['diagnostics']['active_capacity_source']=='R_positive_cap'
    parameter=torch.tensor(0.,dtype=DT,requires_grad=True)
    shift=torch.zeros_like(torch.tensor(samples));shift[:,0]=1.
    geometry=PsiGeometry(torch.tensor(samples)+parameter*shift,torch.tensor(coefficients),a=.3,degree=2,mtor=0)
    domain=two_stage_sampling_geometry(geometry,metadata,reference_axis=torch.tensor(samples))
    actual,=torch.autograd.grad(domain['spacing'],parameter)
    limiting=metadata['boundary']['diagnostics']['limiting_ray']
    phi=metadata['boundary']['phis'][limiting['phi_index']]
    original_R=PsiModel(coefficients,samples,a=.3,degree=2,mtor=0).axis_at(phi)[0]
    expected=metadata['sampling']['spacing']/original_R
    torch.testing.assert_close(actual,torch.tensor(expected),rtol=2e-10,atol=1e-12)


def test_explicit_fixed_lattice_spacing_has_zero_domain_response():
    coefficients=folded_coefficients();samples=axis().numpy()
    metadata=make_domain(coefficients,samples,fixed_spacing=True)
    parameter=torch.tensor(0.,dtype=DT,requires_grad=True)
    geometry=PsiGeometry(torch.tensor(samples),torch.tensor(coefficients)*(1+parameter*.01),a=.15,degree=4,mtor=1)
    domain=two_stage_sampling_geometry(geometry,metadata,reference_axis=torch.tensor(samples))
    assert not domain['spacing'].requires_grad
    assert not domain['rz'].requires_grad
    assert domain['area'].requires_grad


def test_section_operator_rejects_missing_or_invalid_phi_membership():
    rz=torch.ones((2,2),dtype=DT)
    with pytest.raises(ValueError,match='at least one'):
        SectionPsiOperator(field,axis(),rz,torch.tensor([0.,.1,.2]),torch.tensor([0,1]),dtype=DT)
    with pytest.raises(ValueError,match='indices'):
        SectionPsiOperator(field,axis(),rz,torch.tensor([0.,.1]),torch.tensor([0,3]),dtype=DT)


@pytest.mark.parametrize('refit_ridge',[None,1e-3])
@pytest.mark.parametrize('target_refit',[False,True])
def test_complete_two_fit_builder_local_fd_includes_moving_periodic_axis(refit_ridge,target_refit):
    # A smooth integrable toy field supplies an exact periodic axis and an
    # elliptical psi. Both coefficient and axis parameters are perturbed.
    samples=torch.zeros((8,4),dtype=DT);samples[:,0]=1.
    def magnetic(parameters):
        k=1.4+.2*parameters[0];root=1+.05*parameters[1]
        def evaluate(points,gradient=False):
            x,y,z=points.unbind(-1);R=torch.sqrt(x*x+y*y)
            br=-k*z/R;bp=1/R;bz=(R-root)/R
            return torch.stack((br*x/R-bp*y/R,br*y/R+bp*x/R,bz),-1)
        return evaluate
    def tracing(parameters):
        root=torch.stack((1+.05*parameters[1],parameters[1]*0))
        def trace(rz,phi0,phi1,steps,return_samples=False):
            if return_samples:
                return rz[None].expand(steps+1,-1,-1)
            return .2*rz+.8*root[None]
        return trace
    def factors(op,ridge=1e-6):
        identity=torch.eye(op.coefficient_count,dtype=DT)
        A=torch.stack([op.apply(column).reshape(-1) for column in identity],-1)
        scale=torch.linalg.vector_norm(A,dim=0).clamp_min(1e-10)
        upper=torch.linalg.qr(torch.cat((A/scale,math.sqrt(ridge)*identity)),mode='r').R
        return scale.detach(),upper.detach()
    zero=torch.zeros(2,dtype=DT)
    op1=TensorProductPsiOperator(magnetic(zero),samples,a=.15,degree=2,mtor=0,
                                radial_grid=7,vertical_grid=8,phi_grid=5,dtype=DT)
    scale1,upper1=factors(op1)
    coefficients1,_=four_step_pcgls(op1,zero,scale1,upper1)
    metadata=make_domain(coefficients1.detach().numpy(),samples.numpy(),degree=2,mtor=0)
    metadata.update(stage1_coefficients=zero,stage1_reference_coefficients=coefficients1.detach(),
                    stage1_scale=scale1,stage1_upper=upper1)
    sampling=metadata['sampling'];boundary=metadata['boundary']
    op2=SectionPsiOperator(magnetic(zero),samples,torch.tensor(sampling['rz']),
                           torch.tensor(boundary['phis']),sampling['section_index'],a=.15,
                           degree=2,mtor=0,dtype=DT)
    ridge2=1e-6 if refit_ridge is None else refit_ridge
    scale2,upper2=factors(op2,ridge2)
    phi=torch.tensor(boundary['phis'])
    center=dict(axis_samples=samples,psi_coefficients=zero,psi_scale=scale2,psi_upper=upper2,
        two_stage=metadata,target_level=.2,target_dVds=.3,target_adjoint_residual=0.,
        target_adjoint_weights=torch.tensor([.02,.01],dtype=DT),target_axis_weights=torch.full((5,),.015,dtype=DT),
        target_phi=phi,target_axis_R=torch.ones(5,dtype=DT),
        config=dict(nfp=4,psi_a=.15,psi_poly_degree=2,psi_m_tor=0,psi_n_r=7,psi_n_z=8,
                    psi_n_phi=5,axis_trace_steps=8,axis_sample_count=8))
    center['config']['psi_refit_ridge']=refit_ridge
    if target_refit:
        from coil_eval._two_stage.target_domain import target_training_domain
        coefficients2,_=four_step_pcgls(op2,zero,scale2,upper2,ridge=ridge2)
        model=PsiModel(coefficients2.detach().numpy(),samples.numpy(),nfp=4,a=.15,degree=2,mtor=0)
        third=target_training_domain(model,boundary['phis'],.02,ntheta=32,theta_count=16,target_count=480,workers=1)
        third.update(stage2_coefficients=coefficients2.detach(),stage2_scale=scale2,stage2_upper=upper2)
        sample3=third['sampling'];boundary3=third['boundary']
        op3=SectionPsiOperator(magnetic(zero),samples,sample3['rz'],boundary3['phis'],sample3['section_index'],
            a=.15,degree=2,mtor=0,dtype=DT,row_weights=sample3['row_weights'])
        scale3,upper3=factors(op3,1e-10)
        center.update(psi_scale=scale3,psi_upper=upper3)
        metadata['target_refit']=third
        center['config']['psi_target_refit_ridge']=1e-10
    built=build_two_stage_front(magnetic(zero),tracing(zero),center,dtype=DT)
    final_op,final_scale,final_ridge=(op3,scale3,1e-10) if target_refit else (op2,scale2,ridge2)
    A=torch.stack([final_op.apply(column).reshape(-1) for column in torch.eye(2,dtype=DT)],-1)
    scaled=A/final_scale
    expected_coefficients=torch.linalg.solve(scaled.T@scaled+final_ridge*torch.eye(2,dtype=DT),
        scaled.T@final_op.rhs.reshape(-1))/final_scale
    torch.testing.assert_close(built.geometry.coefficients,expected_coefficients,rtol=2e-8,atol=2e-9)
    def value(parameters,local=False):
        out=build_two_stage_front(magnetic(parameters),tracing(parameters),center,dtype=DT,local_value=local)
        return out.geometry.coefficients.square().sum()+.4*out.level+3*out.domain['spacing']+.02*out.geometry.axis_samples[:,0].mean()
    parameters=torch.zeros(2,dtype=DT,requires_grad=True)
    actual,=torch.autograd.grad(value(parameters),parameters)
    h=1e-5;expected=[]
    for direction in torch.eye(2,dtype=DT):
        expected.append((value(h*direction,True)-value(-h*direction,True))/(2*h))
    expected=torch.stack(expected)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-7)
    assert abs(float(actual[0]))>.05
    assert abs(float(actual[1]))>1e-4
