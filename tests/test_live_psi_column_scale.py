"""Formal refit ridge derivatives, independent dense solves and field scaling."""
import math
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval._autograd.front import TensorProductPsiOperator,four_step_pcgls
from coil_eval._autograd.batch_front import BatchTensorProductPsiOperator,batch_four_step_pcgls
from coil_eval._autograd.two_stage_front import SectionPsiOperator,BatchSectionPsiOperator


DT=torch.float64


def dense_matrix(operator,batch=False):
    identity=torch.eye(operator.coefficient_count,dtype=operator.rhs.dtype)
    if batch:
        return torch.stack([operator.apply(v.expand(len(operator.rhs),-1)).flatten(1) for v in identity],-1)
    return torch.stack([operator.apply(v).flatten() for v in identity],-1)


def physical_operator(kind,parameter,dtype):
    phi=torch.arange(12,dtype=DT)*2*math.pi/(4*12)
    axis=torch.stack((1+.01*(4*phi).cos(),.008*(4*phi).sin(),-.04*(4*phi).sin(),.032*(4*phi).cos()),-1)
    axis=axis+parameter*torch.tensor([.03,-.02,.01,.04],dtype=DT)
    def field(points,gradient=False):
        x,y,z=points.unbind(-1)
        return torch.stack((.7+.1*x+.03*z*z,.3+.2*y+.04*x*z,.2+.1*z+.02*x*x),-1)*(1+.2*parameter)
    options=dict(nfp=4,degree=3,mtor=1,dtype=dtype)
    if kind=='tensor':
        return TensorProductPsiOperator(field,axis,a=.15,radial_grid=6,vertical_grid=7,phi_grid=5,**options)
    axes=torch.stack((axis,axis+axis.new_tensor([.04,.01,0.,0.])))
    if kind=='batch_tensor':
        return BatchTensorProductPsiOperator(field,axes,[.15,.18],radial_grid=6,vertical_grid=7,phi_grid=5,**options)
    sections=torch.arange(15)%3
    phis=torch.tensor([0.,.21,.43],dtype=DT)
    rz=torch.stack((torch.linspace(.92,1.08,15,dtype=DT),.04*torch.sin(torch.arange(15,dtype=DT))),-1)
    weights=torch.nn.functional.softplus(torch.linspace(-1.,1.,15,dtype=DT)+parameter*torch.linspace(.2,-.1,15,dtype=DT))
    weights=weights/weights.mean()
    if kind=='section':return SectionPsiOperator(field,axis,rz,phis,sections,a=.15,row_weights=weights,**options)
    return BatchSectionPsiOperator(field,axes,[rz,rz[:12]+.02],[phis,phis],[sections,sections[:12]],
        [.15,.18],row_weights=[weights,weights[:12]/weights[:12].mean()],**options)


@pytest.mark.parametrize('kind',['tensor','section','batch_tensor','batch_section'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_factored_live_column_norms_and_vjp_match_full_design(kind,dtype):
    parameter=torch.tensor(.03,dtype=DT,requires_grad=True)
    operator=physical_operator(kind,parameter,dtype)
    batch=kind.startswith('batch_')
    dense=dense_matrix(operator,batch)
    expected=dense.double().square().sum(-2).clamp_min(1e-30).sqrt().to(dtype)
    calculated=operator.column_norms()
    tolerance=3e-5 if dtype==torch.float32 else 2e-12
    torch.testing.assert_close(calculated,expected,rtol=tolerance,atol=tolerance*.01)
    actual_grad,=torch.autograd.grad(calculated.square().sum(),parameter,retain_graph=True)
    dense_grad,=torch.autograd.grad(expected.square().sum(),parameter)
    torch.testing.assert_close(actual_grad,dense_grad,rtol=tolerance*3,atol=tolerance)


class DenseOperator:
    def __init__(self,A,b):self.A,self.rhs=A,b;self.coefficient_count=A.shape[-1]
    def apply(self,c):return self.A@c
    def transpose(self,r):return self.A.T@r
    def column_norms(self):return self.A.square().sum(0).clamp_min(1e-30).sqrt()


def fit_constants(A,b,ridge):
    scale=A.square().sum(0).sqrt()
    aug=torch.cat((A/scale,math.sqrt(ridge)*torch.eye(A.shape[-1],dtype=A.dtype)))
    upper=torch.linalg.qr(aug,mode='r').R
    c=torch.linalg.solve(A.T@A+ridge*torch.diag(scale.square()),A.T@b)
    return c,scale,upper


def test_live_ridge_matches_independent_dense_weighted_refit_derivative():
    generator=torch.Generator().manual_seed(245)
    A=torch.randn(23,5,generator=generator,dtype=DT)
    delta=torch.randn(23,5,generator=generator,dtype=DT)*.05
    b=torch.randn(23,generator=generator,dtype=DT)
    db=torch.randn(23,generator=generator,dtype=DT)*.08
    raw_weights=torch.linspace(-1.,1.,23,dtype=DT)
    dw=torch.randn(23,generator=generator,dtype=DT)*.2
    ridge=.2
    def arrays(parameter):
        weights=torch.nn.functional.softplus(raw_weights+parameter*dw)
        root=(weights/weights.mean()).sqrt()
        return root[:,None]*(A+parameter*delta),root*(b+parameter*db)
    A0,b0=arrays(torch.tensor(0.,dtype=DT));c0,scale,upper=fit_constants(A0,b0,ridge)
    def exact(parameter):
        matrix,rhs=arrays(parameter);norm=matrix.square().sum(0)
        return torch.linalg.solve(matrix.T@matrix+ridge*torch.diag(norm),matrix.T@rhs)
    parameter=torch.tensor(.02,dtype=DT,requires_grad=True)
    matrix,rhs=arrays(parameter)
    calculated,rms=four_step_pcgls(DenseOperator(matrix,rhs),c0,scale,upper,ridge=ridge,
                                  live_column_scale=True)
    expected=exact(parameter)
    torch.testing.assert_close(calculated,expected,rtol=2e-10,atol=2e-11)
    actual_gradient,=torch.autograd.grad(calculated.square().sum(),parameter)
    h=1e-5
    fd=(exact(parameter.detach()+h).square().sum()-exact(parameter.detach()-h).square().sum())/(2*h)
    torch.testing.assert_close(actual_gradient,fd,rtol=2e-7,atol=2e-9)
    assert float(rms)>0


def test_global_field_scaling_preserves_coefficients_and_scales_training_rms():
    generator=torch.Generator().manual_seed(103)
    A=torch.randn(21,5,generator=generator,dtype=DT);b=torch.randn(21,generator=generator,dtype=DT)
    ridge=.3;c0,scale,upper=fit_constants(A,b,ridge)
    k=torch.tensor(1.3,dtype=DT,requires_grad=True)
    c,rms=four_step_pcgls(DenseOperator(k*A,k*b),c0,scale,upper,ridge=ridge,live_column_scale=True)
    torch.testing.assert_close(c,c0,rtol=2e-13,atol=2e-14)
    invariance,=torch.autograd.grad(c.square().sum(),k,retain_graph=True)
    rms_gradient,=torch.autograd.grad(rms,k)
    assert abs(float(invariance))<1e-12
    torch.testing.assert_close(k*rms_gradient,rms,rtol=2e-12,atol=2e-13)
    old,_=four_step_pcgls(DenseOperator(k*A,k*b),c0,scale,upper,ridge=ridge,live_column_scale=False)
    old_response,=torch.autograd.grad(old.square().sum(),k)
    assert abs(float(old_response))>1e-3


def test_batch_live_ridge_keeps_each_query_scaling_invariant_and_isolated():
    generator=torch.Generator().manual_seed(104)
    A=torch.randn(2,19,5,generator=generator,dtype=DT);b=torch.randn(2,19,generator=generator,dtype=DT)
    ridge=[.1,.3];constants=[fit_constants(A[i],b[i],ridge[i]) for i in range(2)]
    c0,scale,upper=[torch.stack(values) for values in zip(*constants)]
    k=torch.tensor([1.2,.8],dtype=DT,requires_grad=True)
    class Batch:
        def __init__(self):self.A=k[:,None,None]*A;self.rhs=(k[:,None]*b)[...,None];self.valid_rows=torch.tensor([19,19])
        def apply(self,c):return self.A@c[...,None]
        def transpose(self,r):return (self.A.transpose(-1,-2)@r)[...,0]
        def column_norms(self):return self.A.square().sum(1).sqrt()
    calculated,rms=batch_four_step_pcgls(Batch(),c0,scale,upper,ridge=ridge,live_column_scale=True)
    torch.testing.assert_close(calculated,c0,rtol=2e-13,atol=2e-14)
    coefficient_grad,=torch.autograd.grad(calculated.square().sum(),k,retain_graph=True)
    assert float(coefficient_grad.abs().max())<1e-12
    rms_grad,=torch.autograd.grad(rms[0],k)
    torch.testing.assert_close(k[0]*rms_grad[0],rms[0],rtol=2e-12,atol=2e-13)
    assert rms_grad[1]==0


def test_readonly_captured_factors_are_copied_without_numpy_storage_warning():
    import warnings
    A=torch.tensor([[1.,.2],[.3,1.],[-.4,.2]],dtype=DT);b=torch.tensor([.1,.2,-.1],dtype=DT)
    c,scale,upper=fit_constants(A,b,.1)
    arrays=[value.numpy() for value in (c,scale,upper)]
    for value in arrays:value.setflags(write=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result,_=four_step_pcgls(DenseOperator(A,b),*arrays,ridge=.1,live_column_scale=True)
    assert not any('not writable' in str(item.message) for item in caught)
    torch.testing.assert_close(result,c,rtol=1e-12,atol=1e-13)
