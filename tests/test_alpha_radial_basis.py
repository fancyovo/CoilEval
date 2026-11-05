"""Compatible alpha basis selection, native polynomial parity and full feature VJPs."""
from dataclasses import replace
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from coil_eval import EvaluationConfig
from coil_eval._alpha_basis import radial_polynomial
from coil_eval.simsopt.clebsch import AlphaFit,build_modes,radial
from coil_eval.simsopt.geometry import _alpha_dense_theta_derivative


def test_basis_is_explicit_two_stage_option_without_legacy_abi_change():
    original=EvaluationConfig(.05,psi_fit='single_stage',alpha_radial_weighting='histogram',alpha_radial_basis='zernike')
    assert original.alpha_radial_basis=='zernike'
    with pytest.raises(ValueError,match='two_stage'):replace(original,alpha_radial_basis='flux_legendre')
    cfg=replace(original,psi_fit='two_stage',alpha_radial_basis='flux_legendre')
    assert cfg.to_dict()['alpha_radial_basis']=='flux_legendre'
    assert 'alpha_radial_basis' not in cfg.native_options()
    with pytest.raises(ValueError,match='alpha_radial_basis'):replace(cfg,alpha_radial_basis='unknown')


@pytest.mark.parametrize('l,m',[(2,2),(4,2),(12,0),(12,6),(12,12)])
def test_flux_basis_is_legendre_in_flux_radius_squared(l,m):
    rho=np.r_[0.,.08,.33,.71,1.]
    expected=np.polynomial.legendre.Legendre.basis((l-m)//2)(2*rho*rho-1)
    np.testing.assert_allclose(radial(rho,l,m,'flux_legendre'),expected,rtol=2e-12,atol=2e-12)
    assert abs(radial(0.,l,m,'flux_legendre'))==1
    if m>0:assert radial(0.,l,m,'zernike')==0


def test_original_zernike_formula_is_bitwise_unchanged():
    for l in range(13):
        for m in range(l%2,l+1,2):
            old=tuple((l-2*k,(-1)**k*math.factorial(l-k)/
                (math.factorial(k)*math.factorial((l+m)//2-k)*math.factorial((l-m)//2-k)))
                for k in range((l-m)//2+1))
            assert radial_polynomial(l,m)==old


def test_native_flux_polynomials_match_python_exactly(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:pytest.skip('host C++17 compiler unavailable')
    package=Path(__file__).resolve().parents[1]
    cpp=tmp_path/'basis.cpp';exe=tmp_path/('basis.exe' if __import__('os').name=='nt' else 'basis')
    cpp.write_text('''#include "pipeline/alpha_radial_basis.cuh"
#include <iostream>
int main() { for(int l=0;l<=12;++l)for(int m=l%2;m<=l;m+=2) {
  auto c=coil_eval::pipeline::flux_legendre_radial_coefficients(l,m,13);
  std::cout<<l<<' '<<m;for(float v:c)std::cout<<' '<<v;std::cout<<'\\n';
} }
''',encoding='utf-8')
    build=subprocess.run([compiler,'-std=c++17','-I'+str(package/'native/include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);assert run.returncode==0,run.stderr
    for line in run.stdout.splitlines():
        values=line.split();l,m=map(int,values[:2]);expected=np.zeros(13)
        for power,value in radial_polynomial(l,m,'flux_legendre'):expected[power]=value
        np.testing.assert_array_equal(np.array(values[2:],dtype=np.float32),expected.astype(np.float32))


def test_source_capture_basis_and_dense_initializer_evaluation_agree():
    modes=np.asarray(build_modes(6,6,3));rng=np.random.default_rng(922)
    coefficients=rng.normal(0,.01,len(modes))
    snapshot={**{'alpha_mode_'+key:modes[:,i] for i,key in enumerate(('l','m','n','kind'))},
        'alpha_coefficients':coefficients,'alpha_iota_coefficients':np.array([1.1,.02]),
        'nfp':np.asarray(4),'alpha_radial_basis_mode':np.asarray(1),
        'config':{'alpha_radial_basis':'flux_legendre'}}
    fit=AlphaFit.from_snapshot(snapshot);assert fit.radial_basis=='flux_legendre'
    rho=.31;nphi,ntheta=31,67
    fast,_=_alpha_dense_theta_derivative(fit,rho,nphi,ntheta)
    theta,phi=np.broadcast_arrays(-np.arange(ntheta)[None,:]*2*np.pi/ntheta,
        np.arange(nphi)[:,None]*2*np.pi/(4*nphi))
    direct=fit.evaluate(np.full(theta.shape,rho),theta,phi)[1]
    np.testing.assert_allclose(fast,direct,rtol=1e-12,atol=1e-13)
    snapshot['config']['alpha_radial_basis']='zernike'
    with pytest.raises(ValueError,match='captured alpha basis'):AlphaFit.from_snapshot(snapshot)


def test_saved_alpha_old_default_and_explicit_basis_are_distinct(tmp_path):
    values=dict(mode_l=np.array([2]),mode_m=np.array([2]),mode_n=np.array([0]),
        mode_kind=np.array(['sin']),lambda_coeffs=np.array([.1]),iota_coeffs=np.array([1.1]))
    old=tmp_path/'old.npz';new=tmp_path/'new.npz'
    np.savez(old,**values);np.savez(new,**values,radial_basis='flux_legendre')
    a=AlphaFit.read(old,nfp=4);b=AlphaFit.read(new,nfp=4)
    assert a.radial_basis=='zernike' and b.radial_basis=='flux_legendre'
    aa=a.evaluate(np.array([.3]),np.array([.4]),np.array([0.]))[0]
    bb=b.evaluate(np.array([.3]),np.array([.4]),np.array([0.]))[0]
    np.testing.assert_allclose(aa,.3**2*bb,rtol=1e-14)


@pytest.mark.parametrize('basis',['zernike','flux_legendre'])
def test_single_and_batch_design_vjps_include_all_six_moving_features(basis):
    torch=pytest.importorskip('torch')
    from coil_eval._autograd.downstream import clebsch_modes,_AlphaDesign,_design_chunk
    from coil_eval._autograd.batch_downstream import _BatchAlphaDesign
    generator=torch.Generator().manual_seed(910);dtype=torch.float64
    metadata=tuple(torch.as_tensor(v,dtype=dtype) for v in clebsch_modes(4,4,2,basis))
    count=len(metadata[0])+4
    features=[torch.randn(2,7,generator=generator,dtype=dtype) for _ in range(6)]
    features[0]=features[0].sigmoid()*.8+.1;features[5]=features[5].exp()
    features=[v.requires_grad_() for v in features]
    scale=torch.ones(2,count,dtype=dtype)
    answer=_BatchAlphaDesign.apply(*features,scale,metadata,4,3,11,basis)
    upstream=torch.randn(answer.shape,generator=generator,dtype=dtype)
    batch_grad=torch.autograd.grad((answer*upstream).sum(),features)
    for query in range(2):
        single=[v[query].detach().requires_grad_() for v in features]
        value=_AlphaDesign.apply(*single,scale[query],metadata,4,3,11,basis)
        torch.testing.assert_close(value,answer[query],rtol=0,atol=0)
        grads=torch.autograd.grad((value*upstream[query]).sum(),single)
        for k,grad in enumerate(grads):
            torch.testing.assert_close(grad,batch_grad[k][query],rtol=2e-14,atol=2e-13)
            direction=torch.randn(grad.shape,generator=generator,dtype=dtype)
            h=1e-6
            perturbed=[v.detach() for v in single]
            def objective(sign):
                moved=perturbed.copy();moved[k]=moved[k]+sign*h*direction
                return (_design_chunk(*moved,scale[query],metadata,4,0,count,3,basis)*upstream[query]).sum()
            fd=(objective(1)-objective(-1))/(2*h)
            torch.testing.assert_close((grad*direction).sum(),fd,rtol=2e-7,atol=2e-7)


def test_flux_basis_has_identical_column_order_and_count():
    pytest.importorskip('torch')
    from coil_eval._autograd.downstream import clebsch_modes
    old=clebsch_modes(12,12,12);new=clebsch_modes(12,12,12,'flux_legendre')
    assert old[:3]==new[:3] and len(new[0])==2268
    assert all(len(row)==13 for row in new[3])


def test_native_legendre_recurrence_avoids_float32_monomial_cancellation(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:pytest.skip('host C++17 compiler unavailable')
    package=Path(__file__).resolve().parents[1]
    cpp=tmp_path/'stable_basis.cpp';exe=tmp_path/('stable_basis.exe' if __import__('os').name=='nt' else 'stable_basis')
    cpp.write_text('''#include "pipeline/alpha_radial_basis.cuh"
#include <cmath>
#include <iostream>
#include <iomanip>
int main() { std::cout << std::setprecision(17);
  for(int k: {0,1,6,8,10,12}) for(int m: {0,3}) {
    const int degree=2*k+m;
    auto c=coil_eval::pipeline::flux_legendre_radial_coefficients(degree,m,degree+1);
    for(int i=0;i<=10000;++i) {
      const float r=static_cast<float>(i/10000.0);
      float old=c[degree];
      for(int j=degree-1;j>=0;--j) old=std::fma(old,r,c[j]);
      std::cout << k << ' ' << r << ' ' << old << ' '
        << coil_eval::pipeline::flux_legendre_radial_value(c.data(),degree,r) << '\\n';
    }
  }
}
''',encoding='utf-8')
    build=subprocess.run([compiler,'-O2','-std=c++17','-I'+str(package/'native/include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);assert run.returncode==0,run.stderr
    import io
    values=np.loadtxt(io.StringIO(run.stdout))
    for k in (0,1,6,8,10,12):
        rows=values[values[:,0]==k]
        expected=np.polynomial.legendre.Legendre.basis(k)(2*rows[:,1]**2-1)
        np.testing.assert_allclose(rows[:,3],expected,rtol=0,atol=2e-5)
        if k>=6:
            assert np.max(np.abs(rows[:,2]-expected))>1e-4
            assert np.max(np.abs(rows[:,3]-expected))<np.max(np.abs(rows[:,2]-expected))/50


def test_legendre_tensor_values_and_radial_derivatives_match_independent_reference():
    torch=pytest.importorskip('torch')
    from coil_eval._autograd.downstream import _alpha_radial_values
    orders=(0,1,6,8,10,12)
    coefficients=np.zeros((len(orders),25))
    for row,k in enumerate(orders):
        for power,value in radial_polynomial(2*k,0,'flux_legendre'):
            coefficients[row,power]=value
    rho=np.array([0.,.001,.5,.97,.99334,.99806,1.],dtype=np.float32)
    expected=np.stack([np.polynomial.legendre.Legendre.basis(k)(2*rho.astype(float)**2-1)
                       for k in orders],axis=-1)
    values=_alpha_radial_values(torch.from_numpy(rho),torch.tensor(coefficients,dtype=torch.float32),'flux_legendre')
    np.testing.assert_allclose(values.numpy(),expected,rtol=0,atol=2e-5)
    r=torch.tensor(rho.astype(float),requires_grad=True)
    values=_alpha_radial_values(r,torch.tensor(coefficients),'flux_legendre')
    for column,k in enumerate(orders):
        derivative=torch.autograd.grad(values[:,column].sum(),r,retain_graph=True)[0]
        expected_derivative=np.polynomial.legendre.Legendre.basis(k).deriv()(2*rho.astype(float)**2-1)*4*rho
        np.testing.assert_allclose(derivative.numpy(),expected_derivative,rtol=2e-12,atol=2e-12)
