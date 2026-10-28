"""Coefficient-dependent geometry remains fresh when fixed grids are reused."""
import numpy as np

from coil_eval._two_stage.model import PsiModel,build_modes
from coil_eval._two_stage.polynomial import ray_polynomials
from coil_eval._two_stage.quadrature import sample_polar_quadrature
from coil_eval._two_stage.boundary import max_closed_boundary


def make_model(flat=2.):
    modes=build_modes(4,2);coefficients=np.zeros(len(modes))
    for i,m in enumerate(modes):
        if (m.a,m.b,m.m,m.kind)==(0,2,0,'cos'):coefficients[i]=flat
    axis=np.zeros((24,4));axis[:,0]=1.
    return PsiModel(coefficients,axis,nfp=4,a=.2,degree=4,mtor=2)


def test_grid_reuse_never_reuses_coil_coefficients_or_modified_input_angles():
    phi=np.arange(4)*2*np.pi/(4*4);theta=np.arange(32)*2*np.pi/32
    first=ray_polynomials(make_model(2.),phi,theta)
    second=ray_polynomials(make_model(8.),phi,theta)
    expected=np.broadcast_to(6*np.sin(theta)**2,first.shape[:2])
    np.testing.assert_allclose(second[:,:,2]-first[:,:,2],expected,rtol=1e-14,atol=1e-14)
    shifted=theta+.03
    changed=ray_polynomials(make_model(2.),phi,shifted)
    assert not np.array_equal(changed,first)
    np.testing.assert_array_equal(first,ray_polynomials(make_model(2.),phi,theta))


def test_cached_radial_rule_keeps_public_sample_arrays_independent():
    phi=np.arange(4)*2*np.pi/(4*4)
    boundary=max_closed_boundary(make_model(),phi,ntheta=32)
    first=sample_polar_quadrature(boundary,512,theta_count=16)
    expected=first['radial_u'].copy()
    first['radial_u'].setflags(write=True);first['radial_u'][:]=0
    first['radial_weights'].setflags(write=True);first['radial_weights'][:]=0
    second=sample_polar_quadrature(boundary,512,theta_count=16)
    np.testing.assert_array_equal(second['radial_u'],expected)
    assert np.all(second['radial_weights']>0)
    assert not second['radial_u'].flags.writeable
    assert not np.shares_memory(first['radial_u'],second['radial_u'])
