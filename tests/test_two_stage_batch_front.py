"""Batched two-stage fits preserve query isolation and local domain responses."""
import math

import pytest
torch=pytest.importorskip('torch')

from coil_eval._autograd.front import TensorProductPsiOperator,four_step_pcgls
from coil_eval._autograd.two_stage_front import (
    BatchSectionPsiOperator,SectionPsiOperator,build_batch_two_stage_front,build_two_stage_front,
)
from test_two_stage_front import axis,field,make_domain

DT=torch.float64


def test_batch_irregular_sections_match_full_basis_single_operators_and_cross_gradients():
    generator=torch.Generator().manual_seed(289)
    axes=torch.stack((axis(),axis()+torch.tensor([.06,.015,0.,0.]))).requires_grad_()
    phases=[torch.tensor([.03,.21,.51],dtype=DT),torch.tensor([.01,.18,.39,.62],dtype=DT)]
    sections=[torch.tensor([2,0,1,0,2,1,0,0,2]),torch.tensor([3,0,2,1,0,1,3])]
    points=[torch.stack((1+.03*q+.05*torch.randn(len(s),generator=generator,dtype=DT),
                             .03*torch.randn(len(s),generator=generator,dtype=DT)),-1) for q,s in enumerate(sections)]
    parameters=torch.tensor([.02,-.03],dtype=DT,requires_grad=True)
    class Field:
        def __call__(self,p,gradient=False):
            return field(p)*(1+parameters[:,None,None])
    op=BatchSectionPsiOperator(Field(),axes,points,phases,sections,[.15,.12],dtype=DT)
    coefficients=torch.randn((2,1574),generator=generator,dtype=DT)
    rows=torch.randn(op.rhs.shape,generator=generator,dtype=DT)
    actual=op.apply(coefficients);transpose=op.transpose(rows)
    for q in range(2):
        def one(p,gradient=False):return field(p)*(1+parameters[q])
        single=SectionPsiOperator(one,axes[q],points[q],phases[q],sections[q],a=[.15,.12][q],dtype=DT)
        S,K=single.rhs.shape
        torch.testing.assert_close(actual[q,:S,:K],single.apply(coefficients[q]),rtol=5e-12,atol=5e-10)
        torch.testing.assert_close(transpose[q],single.transpose(rows[q,:S,:K]),rtol=5e-12,atol=5e-10)
        assert int(op.valid_rows[q])==len(points[q])
    assert not torch.count_nonzero(actual[~op.mask])
    ga,gp=torch.autograd.grad(actual[0].square().sum(),(axes,parameters))
    assert not torch.count_nonzero(ga[1]) and gp[1]==0
    assert torch.linalg.vector_norm(ga[0])>0 and gp[0]!=0


class ToyField:
    def __init__(self,parameters):
        self.parameters=parameters
        self.positions=parameters.new_zeros((len(parameters),1,3))
        self.base_R=parameters.new_tensor([1.,1.07])[:len(parameters)]
        self.base_k=parameters.new_tensor([1.4,1.7])[:len(parameters)]

    def _evaluate(self,points,parameters,root,k):
        x,y,z=points.unbind(-1);R=torch.sqrt(x*x+y*y)
        br=-k*z/R;bp=1/R;bz=(R-root)/R
        return torch.stack((br*x/R-bp*y/R,br*y/R+bp*x/R,bz),-1)

    def __call__(self,points,gradient=False):
        root=(self.base_R+.05*self.parameters[:,1])[:,None]
        k=(self.base_k+.2*self.parameters[:,0])[:,None]
        return self._evaluate(points,self.parameters,root,k)

    def trace(self,rz,phi0,phi1,steps,return_samples=False):
        root=torch.stack((self.base_R+.05*self.parameters[:,1],self.parameters[:,1]*0),-1)
        if return_samples:
            return rz[None].expand(steps+1,-1,-1,-1)
        return .2*rz+.8*root[:,None]

    def single(self,index):
        def evaluate(points,gradient=False):
            p=self.parameters[index]
            return self._evaluate(points,p,self.base_R[index]+.05*p[1],self.base_k[index]+.2*p[0])
        def trace(rz,phi0,phi1,steps,return_samples=False):
            p=self.parameters[index]
            root=torch.stack((self.base_R[index]+.05*p[1],p[1]*0))
            if return_samples:
                return rz[None].expand(steps+1,-1,-1)
            return .2*rz+.8*root[None]
        return evaluate,trace


def factors(op):
    identity=torch.eye(op.coefficient_count,dtype=DT)
    A=torch.stack([op.apply(c).reshape(-1) for c in identity],-1)
    scale=torch.linalg.vector_norm(A,dim=0).clamp_min(1e-10)
    upper=torch.linalg.qr(torch.cat((A/scale,math.sqrt(1e-6)*identity)),mode='r').R
    return scale.detach(),upper.detach()


def toy_centers():
    batch=ToyField(torch.zeros((2,2),dtype=DT));centers=[]
    for query,a in enumerate((.15,.12)):
        samples=torch.zeros((8,4),dtype=DT);samples[:,0]=batch.base_R[query]
        magnetic,_=batch.single(query)
        op1=TensorProductPsiOperator(magnetic,samples,a=a,degree=2,mtor=0,
                                    radial_grid=7,vertical_grid=8,phi_grid=5,dtype=DT)
        scale1,upper1=factors(op1);warm=torch.zeros(2,dtype=DT)
        coefficients1,_=four_step_pcgls(op1,warm,scale1,upper1)
        metadata=make_domain(coefficients1.detach().numpy(),samples.numpy(),a=a,degree=2,mtor=0)
        metadata.update(stage1_coefficients=warm,stage1_reference_coefficients=coefficients1.detach(),
                        stage1_scale=scale1,stage1_upper=upper1)
        sample,boundary=metadata['sampling'],metadata['boundary']
        op2=SectionPsiOperator(magnetic,samples,torch.tensor(sample['rz']),torch.tensor(boundary['phis']),
                              sample['section_index'],a=a,degree=2,mtor=0,dtype=DT)
        scale2,upper2=factors(op2)
        centers.append(dict(axis_samples=samples,psi_coefficients=warm,psi_scale=scale2,psi_upper=upper2,
            two_stage=metadata,target_level=.2,target_dVds=.3,target_adjoint_residual=0.,
            target_adjoint_weights=torch.tensor([.02,.01],dtype=DT),
            target_axis_weights=torch.full((5,),.015,dtype=DT),target_phi=torch.tensor(boundary['phis']),
            target_axis_R=torch.full((5,),float(batch.base_R[query]),dtype=DT),
            config=dict(nfp=4,psi_a=a,psi_poly_degree=2,psi_m_tor=0,psi_n_r=7,psi_n_z=8,
                        psi_n_phi=5,axis_trace_steps=8,axis_sample_count=8)))
    return centers


def losses(front):
    spacing=torch.stack([d['spacing'] for d in front.domains])
    return front.coefficients.square().sum(-1)+.4*front.levels+3*spacing+.02*front.axis_samples[:,:,0].mean(-1)


def test_batched_two_fit_builder_matches_single_values_vjps_and_zero_cross_query_response():
    centers=toy_centers()
    parameters=torch.zeros((2,2),dtype=DT,requires_grad=True)
    batch_field=ToyField(parameters)
    out=build_batch_two_stage_front(batch_field,centers,dtype=DT)
    values=losses(out)
    actual_grad,=torch.autograd.grad(values.sum(),parameters,retain_graph=True)
    isolated,=torch.autograd.grad(values[0],parameters)
    assert not torch.count_nonzero(isolated[1])
    single_parameters=parameters.detach().requires_grad_();single_field=ToyField(single_parameters)
    singles=[];single_values=[]
    for q in range(2):
        magnetic,trace=single_field.single(q)
        single=build_two_stage_front(magnetic,trace,centers[q],dtype=DT)
        singles.append(single)
        single_values.append(single.geometry.coefficients.square().sum()+.4*single.level+
                             3*single.domain['spacing']+.02*single.geometry.axis_samples[:,0].mean())
    expected_values=torch.stack(single_values)
    expected_grad,=torch.autograd.grad(expected_values.sum(),single_parameters)
    torch.testing.assert_close(out.coefficients,torch.stack([s.geometry.coefficients for s in singles]),rtol=1e-9,atol=2e-10)
    torch.testing.assert_close(values,expected_values,rtol=1e-9,atol=2e-10)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-7,atol=2e-9)
    assert out.axis_diagnostics['second_fit_point_count'].shape==(2,)


def test_batched_local_value_fd_retains_both_moving_axes_and_domain_spacing():
    centers=toy_centers()
    parameters=torch.zeros((2,2),dtype=DT,requires_grad=True)
    out=build_batch_two_stage_front(ToyField(parameters),centers,dtype=DT)
    actual,=torch.autograd.grad(losses(out).sum(),parameters)
    def value(p):
        return losses(build_batch_two_stage_front(ToyField(p),centers,dtype=DT,local_value=True)).sum()
    h=1e-5;expected=[]
    for direction in torch.eye(4,dtype=DT).reshape(4,2,2):
        expected.append((value(h*direction)-value(-h*direction))/(2*h))
    torch.testing.assert_close(actual,torch.stack(expected).reshape(2,2),rtol=3e-5,atol=3e-7)
    assert bool((actual[:,0].abs()>.05).all())
    assert bool((actual[:,1].abs()>1e-4).all())


@pytest.mark.parametrize('local_value',[False,True])
def test_value_only_single_and_batch_preserve_values_without_axis_fit_or_domain_tapes(local_value):
    centers=toy_centers()
    parameters=torch.tensor([[.001,-.002],[-.003,.0015]],dtype=DT)
    magnetic=ToyField(parameters)
    taped=build_batch_two_stage_front(magnetic,centers,dtype=DT,local_value=local_value)
    assert taped.axis_samples.requires_grad
    plain=build_batch_two_stage_front(magnetic,centers,dtype=DT,local_value=local_value,value_only=True)
    for name in ('axis_samples','coefficients','levels','psi_train_rms'):
        expected,actual=getattr(taped,name),getattr(plain,name)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert not actual.requires_grad and actual.grad_fn is None
    for index,center in enumerate(centers):
        f,trace=magnetic.single(index)
        single_taped=build_two_stage_front(f,trace,center,dtype=DT,local_value=local_value)
        single=build_two_stage_front(f,trace,center,dtype=DT,local_value=local_value,value_only=True)
        torch.testing.assert_close(single.geometry.coefficients,single_taped.geometry.coefficients,rtol=0,atol=0)
        torch.testing.assert_close(single.geometry.axis_samples,single_taped.geometry.axis_samples,rtol=0,atol=0)
        torch.testing.assert_close(single.level,single_taped.level,rtol=0,atol=0)
        assert not single.geometry.coefficients.requires_grad
        assert not single.geometry.axis_samples.requires_grad
        assert not single.stage1_geometry.coefficients.requires_grad
        assert not single.psi_train_rms.requires_grad and not single.level.requires_grad
        for domain,reference in ((single.domain,single_taped.domain),(plain.domains[index],taped.domains[index])):
            for name in ('rz','spacing','radii','area','polygons_rz','level'):
                torch.testing.assert_close(domain[name],reference[name],rtol=0,atol=0)
                assert not domain[name].requires_grad and domain[name].grad_fn is None
    # A surrounding inference context must not break the temporary 2x2 axis
    # Jacobian calculation; only that small calculation reenables gradients.
    with torch.no_grad():
        inference=build_batch_two_stage_front(magnetic,centers,dtype=DT,local_value=local_value,value_only=True)
    torch.testing.assert_close(inference.coefficients,plain.coefficients,rtol=0,atol=0)
    assert not inference.coefficients.requires_grad
