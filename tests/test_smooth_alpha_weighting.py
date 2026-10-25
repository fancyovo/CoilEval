"""Smooth radial density, all-point derivatives and formal alpha ridge."""
from dataclasses import replace
import math
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval import EvaluationConfig
from coil_eval._autograd.downstream import DownstreamConfig,smooth_alpha_weights,frozen_pcgls
from coil_eval._autograd.batch_downstream import batch_frozen_pcgls
from test_two_stage_native import factory,Function,configurations
from _oracle import make_coils


DT=torch.float64


def test_smooth_alpha_config_is_opt_in_s2_and_not_added_to_legacy_c_struct():
    old=EvaluationConfig(.03)
    assert old.alpha_radial_weighting=='histogram'
    with pytest.raises(ValueError,match='two_stage'):replace(old,alpha_radial_weighting='smooth')
    smooth=replace(old,psi_fit='two_stage',alpha_radial_weighting='smooth')
    assert smooth.alpha_weight_sigma==.05 and smooth.alpha_count_floor==32.
    assert not any(name.startswith('alpha_') for name in smooth.native_options())
    for kwargs in ({'alpha_radial_weighting':'bad'},{'alpha_weight_sigma':0},
                   {'alpha_count_floor':-1},{'alpha_weight_sigma':math.nan}):
        with pytest.raises(ValueError):replace(smooth,**kwargs)


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_softcounts_and_normalized_weights_match_independent_numpy_formula(dtype):
    rho=torch.linspace(.081,.997,51,dtype=dtype)
    norm2=torch.linspace(.7,1.8,51,dtype=dtype)
    cfg=replace(DownstreamConfig(),alpha_radial_weighting='smooth')
    weights,counts=smooth_alpha_weights(rho,norm2,cfg)
    kernel=np.exp(-.5*((rho.double().numpy()[:,None]-(np.arange(10)+.5)/10)/.05)**2)
    expected_counts=kernel.sum(0)
    raw=(np.sqrt((kernel/(expected_counts+32)).sum(1))/norm2.sqrt().double().numpy()).astype(weights.numpy().dtype)
    normalization=np.sqrt(np.array(len(rho)/(raw.astype(np.float64)**2).sum(),dtype=raw.dtype))
    expected=raw*normalization
    np.testing.assert_allclose(counts.numpy(),expected_counts,rtol=3e-15,atol=1e-14)
    np.testing.assert_allclose(weights.numpy(),expected,rtol=2e-7 if dtype==torch.float32 else 2e-15)
    torch.testing.assert_close(weights.square().mean(),torch.tensor(1.,dtype=dtype),rtol=3e-7,atol=1e-15)


def test_five_to_six_bin_crossing_is_smooth_and_softcount_derivatives_reach_other_rows():
    rho=torch.cat((torch.tensor([.083,.087,.091,.095,.099,.1],dtype=DT),torch.linspace(.111,.999,201,dtype=DT)))
    cfg=replace(DownstreamConfig(),alpha_radial_weighting='smooth')
    direction=torch.zeros_like(rho);direction[5]=1.
    def weights(t):return smooth_alpha_weights(rho+t*direction,torch.ones_like(rho),cfg)[0]
    tiny=1e-8
    assert torch.linalg.vector_norm(weights(torch.tensor(tiny))-weights(torch.tensor(-tiny)))<1e-5
    t=torch.tensor(0.,dtype=DT,requires_grad=True)
    response,=torch.autograd.grad(weights(t)[0],t)
    h=1e-6
    fd=(weights(torch.tensor(h))[0]-weights(torch.tensor(-h))[0])/(2*h)
    torch.testing.assert_close(response,fd,rtol=2e-7,atol=2e-9)
    assert abs(float(response))>1e-4


def test_batched_softcounts_keep_queries_independent_and_match_single():
    rho=torch.stack((torch.linspace(.08,.99,41,dtype=DT),torch.linspace(.12,.97,41,dtype=DT))).requires_grad_()
    norm2=1+rho.square();cfg=replace(DownstreamConfig(),alpha_radial_weighting='smooth')
    weight,count=smooth_alpha_weights(rho,norm2,cfg)
    for i in range(2):
        one,one_count=smooth_alpha_weights(rho[i],norm2[i],cfg)
        torch.testing.assert_close(weight[i],one,atol=1e-15,rtol=1e-15)
        torch.testing.assert_close(count[i],one_count,atol=1e-15,rtol=1e-15)
    gradient,=torch.autograd.grad(weight[0,0],rho)
    assert torch.count_nonzero(gradient[1])==0 and gradient[0].abs().sum()>0


def test_alpha_live_ridge_matches_dense_weighted_refit_and_global_field_scaling():
    generator=torch.Generator().manual_seed(445)
    A=torch.randn(25,5,generator=generator,dtype=DT)
    b=torch.randn(25,generator=generator,dtype=DT)
    da=torch.randn(25,5,generator=generator,dtype=DT)*.1
    ridge=.15;D0=A.square().sum(0).sqrt();A0=A/D0
    upper=torch.linalg.qr(torch.cat((A0,math.sqrt(ridge)*torch.eye(5,dtype=DT))),mode='r').R
    x0=torch.linalg.solve(A0.T@A0+ridge*torch.eye(5,dtype=DT),A0.T@b)
    cache={'upper':upper,'scaled_solution':x0}
    def exact(t):
        matrix=A+t*da;norm=matrix.square().sum(0)
        return torch.linalg.solve(matrix.T@matrix+ridge*torch.diag(norm),matrix.T@b)*D0
    t=torch.tensor(.01,dtype=DT,requires_grad=True)
    calculated,_=frozen_pcgls((A+t*da)/D0,b,cache,ridge,live_column_scale=True)
    torch.testing.assert_close(calculated,exact(t),rtol=2e-10,atol=2e-11)
    actual,=torch.autograd.grad(calculated.square().sum(),t)
    h=1e-5;fd=(exact(t.detach()+h).square().sum()-exact(t.detach()-h).square().sum())/(2*h)
    torch.testing.assert_close(actual,fd,rtol=2e-7,atol=2e-9)
    k=torch.tensor(1.2,dtype=DT,requires_grad=True)
    solution,residual=frozen_pcgls(k*A0,k*b,cache,ridge,live_column_scale=True)
    torch.testing.assert_close(solution,x0,rtol=1e-12,atol=1e-13)
    derivative,=torch.autograd.grad(solution.square().sum(),k,retain_graph=True)
    rms=residual.square().mean().sqrt();rms_derivative,=torch.autograd.grad(rms,k)
    assert abs(float(derivative))<1e-11
    torch.testing.assert_close(k*rms_derivative,rms,rtol=1e-12,atol=1e-13)


def test_batch_alpha_live_ridge_matches_independent_single_values_and_vjps():
    generator=torch.Generator().manual_seed(446)
    A=torch.randn(2,21,5,generator=generator,dtype=DT).requires_grad_()
    b=torch.randn(2,21,generator=generator,dtype=DT).requires_grad_()
    cache={'upper':torch.eye(5,dtype=DT)[None].expand(2,-1,-1),
           'scaled_solution':torch.zeros(2,5,dtype=DT)}
    actual=batch_frozen_pcgls(A,b,cache,.1,live_column_scale=True)
    singles=[frozen_pcgls(A[i],b[i],{k:v[i] for k,v in cache.items()},.1,live_column_scale=True) for i in range(2)]
    for index,value in enumerate(actual):
        torch.testing.assert_close(value,torch.stack([one[index] for one in singles]),rtol=1e-12,atol=1e-13)
    ga=torch.autograd.grad(actual[0].square().sum(),(A,b),retain_graph=True)
    ge=torch.autograd.grad(sum(one[0].square().sum() for one in singles),(A,b))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=2e-11,atol=2e-12)


def test_smooth_context_options_are_explicit_and_snapshot_preserves_them(factory):
    driver,library,_=factory();calls=[]
    library.sgpu_two_stage_set_alpha_radial_weighting_v1=Function(lambda *args:calls.append(args[1:]) or 0)
    config=replace(configurations()[0],alpha_radial_weighting='smooth',alpha_weight_sigma=.04,alpha_count_floor=24.)
    result,snapshot=driver.evaluate(make_coils(),config,capture=True)
    assert calls==[(1,.04,24.)]
    assert snapshot['config']['alpha_radial_weighting']=='smooth'
    assert snapshot['config']['alpha_weight_sigma']==.04 and result['diagnostics']['alpha_count_floor']==24.


def test_old_native_library_histogram_is_usable_but_smooth_requests_rebuild(factory):
    driver,_,_=factory()
    assert driver.evaluate(make_coils(),configurations()[0])[0]['status']=='ok'
    with pytest.raises(RuntimeError,match='rebuild'):
        driver.evaluate(make_coils(),replace(configurations()[0],alpha_radial_weighting='smooth'))
    assert driver._context is None
