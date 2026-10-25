"""Native outer-QS bin and sampled iota range, with live local derivatives."""
from dataclasses import replace
import math

import numpy as np
import pytest
torch=pytest.importorskip('torch')

from coil_eval._autograd.downstream import DownstreamConfig,iota_profile_extrema,qs_metric
from coil_eval._autograd.batch_downstream import batch_qs_metric

DT=torch.float64


def numpy_iota_range(coefficients,rho_min):
    values=[];coordinates=[]
    lower=float(rho_min)**2
    for sample in range(257):
        u=lower+(1-lower)*sample/256.
        value=0.;power=1.
        for coefficient in coefficients:
            value+=float(coefficient)*power
            power*=u
        values.append(value);coordinates.append(u)
    lo,hi=int(np.argmin(values)),int(np.argmax(values))
    return values[lo],values[hi],coordinates[lo],coordinates[hi]


def numpy_qs_outer(B,J,gs,fd,rho,weight,iota,G,c):
    threshold=np.float32(c.volume_rho_min+(1-c.volume_rho_min)*(c.radial_bin_count-1)/c.radial_bin_count)
    numerator=denominator=0.
    for b,j,grad_s,flux_derivative,r,w in zip(B,J,gs,fd,rho,weight):
        if np.float32(r)<threshold:
            continue
        magnitude=np.sqrt(max(float(np.dot(b,b)),1e-30))
        grad_magnitude=np.asarray(j).T@np.asarray(b)/magnitude
        A=np.dot(np.cross(b,float(flux_derivative)*grad_s),grad_magnitude)
        C=np.dot(b,grad_magnitude)
        u=float(r)*float(r);iv=0.;power=1.
        for coefficient in iota:
            iv+=float(coefficient)*power;power*=u
        residual=((c.target_M*iv-c.target_N)*A-c.target_M*float(G)*C)/max(magnitude**3,1e-30)
        numerator+=float(w)*residual**2;denominator+=float(w)
    return math.sqrt(numerator/max(denominator,1e-300))/max(math.hypot(c.target_M,c.target_N),1.)


@pytest.mark.parametrize('coefficients,rho_min',[
    ([1.3],.08),([.7,-.4],.23),([1.1,-.734,1.],.08),
    ([.4,1.1,-2.7,1.4],.13),([0.,0.,0.,0.],0.),
])
def test_iota_extrema_match_native_257_sample_numpy_reference(coefficients,rho_min):
    coeff=torch.tensor(coefficients,dtype=DT,requires_grad=True)
    result=iota_profile_extrema(coeff,rho_min)
    minimum,maximum,umin,umax=numpy_iota_range(coefficients,rho_min)
    torch.testing.assert_close(result['iota_min'],torch.tensor(minimum,dtype=DT),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(result['iota_max'],torch.tensor(maximum,dtype=DT),rtol=1e-14,atol=1e-14)
    gmin,=torch.autograd.grad(result['iota_min'],coeff,retain_graph=True)
    gmax,=torch.autograd.grad(result['iota_max'],coeff)
    powers=np.arange(len(coefficients))
    torch.testing.assert_close(gmin,torch.tensor(umin**powers),rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(gmax,torch.tensor(umax**powers),rtol=1e-13,atol=1e-14)


def test_iota_range_uses_samples_not_true_stationary_points_and_retains_vjp():
    coeff=torch.tensor([1.1,-.734,1.],dtype=DT,requires_grad=True)
    output=iota_profile_extrema(coeff,.08)
    true_minimum=1.1-.734**2/4
    assert float(output['iota_min'].detach())>true_minimum+1e-8
    weights=torch.tensor([.3,-.2,.1],dtype=DT)
    def value(t):
        result=iota_profile_extrema(coeff.detach()+t*weights,.08)
        return result['iota_min']+.17*result['iota_max']
    actual,=torch.autograd.grad(output['iota_min']+.17*output['iota_max'],coeff)
    h=1e-6;expected=(value(h)-value(-h))/(2*h)
    torch.testing.assert_close(actual@weights,expected,rtol=1e-7,atol=1e-9)


def test_batched_iota_ranges_use_independent_lower_radii_and_first_tie_indices():
    coeff=torch.tensor([[1.,.2,-.4,.1],[1.2,0.,0.,0.],[-.2,.8,-.5,.12]],dtype=DT,requires_grad=True)
    lower=torch.tensor([.08,.3,.17],dtype=DT)
    result=iota_profile_extrema(coeff,lower)
    for query in range(3):
        minimum,maximum,umin,umax=numpy_iota_range(coeff.detach()[query].numpy(),float(lower[query]))
        torch.testing.assert_close(result['iota_min'][query],torch.tensor(minimum,dtype=DT),rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(result['iota_max'][query],torch.tensor(maximum,dtype=DT),rtol=1e-14,atol=1e-14)
        gradient,=torch.autograd.grad(result['iota_min'][query],coeff,retain_graph=True)
        torch.testing.assert_close(gradient[query],torch.tensor(umin**np.arange(4)),rtol=1e-13,atol=1e-14)
        assert not torch.count_nonzero(gradient[torch.arange(3)!=query])
    # Every sample is tied for this constant profile; native picks u_min first.
    tie_grad,=torch.autograd.grad(result['iota_max'][1],coeff)
    torch.testing.assert_close(tie_grad[1],lower[1]**(2*torch.arange(4)))


def make_qs_inputs(batch=1,requires_grad=False):
    generator=torch.Generator().manual_seed(511)
    n=9;shape=(batch,n)
    values=[torch.randn((*shape,3),generator=generator,dtype=DT)+2,
            torch.randn((*shape,3,3),generator=generator,dtype=DT),
            torch.randn((*shape,3),generator=generator,dtype=DT),
            torch.rand(shape,generator=generator,dtype=DT)+.2,
            torch.tensor([.2,.5,.87,.907,.911,.93,.95,.975,.999],dtype=DT)[None].expand(batch,-1).clone(),
            torch.rand(shape,generator=generator,dtype=DT)+.3,
            torch.randn((batch,4),generator=generator,dtype=DT)*.1+.3,
            torch.linspace(.2,.4,batch,dtype=DT)]
    return [value.requires_grad_(requires_grad) for value in values]


def test_outer_qs_matches_direct_native_formula_and_float32_bin_boundary():
    c=DownstreamConfig();values=[v[0] for v in make_qs_inputs()]
    edge=np.float32(c.volume_rho_min+(1-c.volume_rho_min)*(c.radial_bin_count-1)/c.radial_bin_count)
    values[4][3]=float(np.nextafter(edge,np.float32(-np.inf)))
    values[4][4]=float(edge)
    B,J,gs,fd,rho,w,iota,G=values
    result=qs_metric(dict(grad_s=gs,flux_derivative=fd,rho=rho,volume_weight=w),B,J,iota,G,c)
    reference=numpy_qs_outer(*[v.detach().numpy() for v in values],c)
    torch.testing.assert_close(result['raw_qs_outer'],torch.tensor(reference,dtype=DT),rtol=2e-13,atol=1e-14)
    # Just taking the full-volume or boundary scalar is observably different.
    assert abs(float(result['raw_qs_outer']-result['raw_qs']))>1e-5


def test_outer_qs_vjp_retains_all_selected_live_inputs():
    c=DownstreamConfig();values=[v[0].detach().requires_grad_() for v in make_qs_inputs()]
    def function(B,J,gs,fd,rho,w,iota,G):
        return qs_metric(dict(grad_s=gs,flux_derivative=fd,rho=rho,volume_weight=w),B,J,iota,G,c)['raw_qs_outer']
    assert torch.autograd.gradcheck(function,tuple(values),fast_mode=True,atol=2e-6,rtol=2e-4)
    gradients=torch.autograd.grad(function(*values),values)
    assert all(bool(torch.linalg.vector_norm(g)>0) for g in gradients)
    assert not torch.count_nonzero(gradients[4][:4])


def test_outer_qs_empty_or_zero_residual_sets_return_zero_with_finite_zero_vjp():
    c=DownstreamConfig()
    for empty in (False,True):
        B,J,gs,fd,rho,w,iota,G=[v[0].detach().requires_grad_() for v in make_qs_inputs()]
        if empty:
            actual_rho=rho*.1
        else:
            J=J*0;actual_rho=rho
        result=qs_metric(dict(grad_s=gs,flux_derivative=fd,rho=actual_rho,volume_weight=w),B,J,iota,G,c)['raw_qs_outer']
        assert result==0
        gradients=torch.autograd.grad(result,(B,J,gs,fd,rho,w,iota,G))
        assert all(bool(torch.isfinite(g).all()) and not torch.count_nonzero(g) for g in gradients)


def test_batch_outer_qs_matches_independent_queries_with_different_rho_min_and_no_cross_gradients():
    c=DownstreamConfig();lower=[.08,.3,.6]
    values=make_qs_inputs(3,requires_grad=True)
    B,J,gs,fd,rho,w,iota,G=values
    result=batch_qs_metric(dict(grad_s=gs,flux_derivative=fd,rho=rho,volume_weight=w),B,J,iota,G,c,volume_rho_min=lower)
    for query in range(3):
        cfg=replace(c,volume_rho_min=lower[query])
        one=[v[query] for v in values]
        reference=numpy_qs_outer(*[v.detach().numpy() for v in one],cfg)
        torch.testing.assert_close(result['raw_qs_outer'][query],torch.tensor(reference,dtype=DT),rtol=2e-13,atol=1e-14)
        single=qs_metric(dict(grad_s=one[2],flux_derivative=one[3],rho=one[4],volume_weight=one[5]),one[0],one[1],one[6],one[7],cfg)
        torch.testing.assert_close(result['raw_qs_outer'][query],single['raw_qs_outer'],rtol=1e-14,atol=1e-14)
    gradient=torch.autograd.grad(result['raw_qs_outer'][0],values)
    assert all(not torch.count_nonzero(g[1:]) for g in gradient)
