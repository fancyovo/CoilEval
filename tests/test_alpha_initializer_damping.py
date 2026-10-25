"""Coordinate-only alpha seed damping with a strict dense Jacobian check."""
import numpy as np
import pytest

from coil_eval.simsopt import AlphaFit,InitializerConfig
from coil_eval.simsopt.clebsch import build_modes
from coil_eval.simsopt.geometry import alpha_reparameterize,_alpha_dense_theta_derivative


def surface(nphi=9,ntheta=17):
    phi=np.arange(nphi)[:,None]*2*np.pi/(4*nphi)
    theta=np.arange(ntheta)[None,:]*2*np.pi/ntheta
    R=1+.1*np.cos(theta)
    return np.stack(np.broadcast_arrays(R*np.cos(phi),R*np.sin(phi),.07*np.sin(theta)),axis=-1)


class SinAlpha:
    nfp=4
    def __init__(self,amplitude):self.amplitude=amplitude
    def evaluate(self,rho,theta,phi,*,device='cpu'):
        return self.amplitude*np.sin(theta),self.amplitude*np.cos(theta),np.zeros_like(theta)
    def iota(self,rho):return 1.31


@pytest.mark.parametrize('rho',[1.,.61])
def test_separable_dense_derivative_matches_independent_full_alpha_evaluation(rho):
    mode=np.asarray(build_modes(4,4,2))
    fit=AlphaFit(mode,np.random.default_rng(84).normal(0,.03,len(mode)),np.array([1.2,.03]),4)
    nphi,ntheta=97,257
    calculated,method=_alpha_dense_theta_derivative(fit,rho,nphi,ntheta)
    th,ph=np.broadcast_arrays(-np.arange(ntheta)[None,:]*2*np.pi/ntheta,
                              np.arange(nphi)[:,None]*2*np.pi/(4*nphi))
    expected=fit.evaluate(np.full(th.shape,rho),th,ph)[1]
    np.testing.assert_allclose(calculated,expected,rtol=2e-12,atol=3e-13)
    assert method=='separable_fourier'


def test_folded_coordinate_map_is_damped_to_positive_dense_jacobian_without_mutation():
    alpha=SinAlpha(1.3);points=surface()
    with pytest.raises(ValueError,match='noninvertible'):alpha_reparameterize(points,alpha)
    result,diagnostics=alpha_reparameterize(points,alpha,alpha_fold_policy='damp')
    assert np.isfinite(result).all() and diagnostics['original_min_jacobian']<0
    assert 0<diagnostics['damping_factor']<1
    assert diagnostics['damped_min_jacobian']>=.05
    assert diagnostics['damped_dense_min_jacobian']>=.05
    assert diagnostics['noninvertible_fraction']==0 and diagnostics['original_noninvertible_fraction']>0
    assert diagnostics['dense_grid']['nphi']>=97 and diagnostics['dense_grid']['ntheta']>=257
    assert alpha.amplitude==1.3 and alpha.iota(1.)==1.31


def test_zero_lambda_and_legacy_reject_false_path_are_unchanged():
    points=surface();zero=SinAlpha(0.)
    old,old_diagnostics=alpha_reparameterize(points,zero)
    damped,d=alpha_reparameterize(points,zero,alpha_fold_policy='damp')
    np.testing.assert_array_equal(damped,old)
    assert d['damping_factor']==1. and d['damped_min_jacobian']==1.
    assert set(old_diagnostics)=={'one_plus_lambda_theta_min','noninvertible_fraction'}
    folded=SinAlpha(1.3)
    first,_=alpha_reparameterize(points,folded,reject_folds=False)
    second,_=alpha_reparameterize(points,folded,reject_folds=False,alpha_fold_policy='reject',alpha_min_jacobian=.2)
    np.testing.assert_array_equal(first,second)
    # Explicit damp still enforces positive Jacobians even with the old flag off.
    _,d=alpha_reparameterize(points,folded,reject_folds=False,alpha_fold_policy='damp')
    assert d['damped_min_jacobian']>=.05


def test_dense_validation_catches_a_fold_between_extraction_grid_nodes():
    alpha=SinAlpha(1.01);points=surface()
    _,coarse=alpha_reparameterize(points,alpha)
    assert coarse['one_plus_lambda_theta_min']>0
    _,d=alpha_reparameterize(points,alpha,alpha_fold_policy='damp')
    assert d['original_dense_min_jacobian']<0 and d['damping_factor']<1
    assert d['damped_dense_min_jacobian']>=.05


def test_damped_mapping_is_continuous_when_the_floor_becomes_active():
    points=surface();probe=SinAlpha(1.)
    derivative,_=_alpha_dense_theta_derivative(probe,1.,97,257)
    threshold=.95/-derivative.min()
    left,_=alpha_reparameterize(points,SinAlpha(threshold-1e-7),alpha_fold_policy='damp')
    right,_=alpha_reparameterize(points,SinAlpha(threshold+1e-7),alpha_fold_policy='damp')
    assert np.max(abs(left-right))<1e-7


def test_initializer_fold_settings_validate_and_default_to_compatible_reject():
    cfg=InitializerConfig()
    assert cfg.alpha_fold_policy=='reject' and cfg.reject_alpha_folds and cfg.alpha_min_jacobian==.05
    for kwargs in ({'alpha_fold_policy':'off'},{'alpha_min_jacobian':0},{'alpha_min_jacobian':1}):
        with pytest.raises(ValueError):InitializerConfig(**kwargs)
