"""Formal CPU sampling preserves the first branch and per-ray provenance."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval._autograd.front import PsiGeometry,native_modes,_RadialRoot
from coil_eval._autograd.downstream import DownstreamConfig,volume_points
from coil_eval._cpu.geometry import _formal_boundary_radii,formal_volume_points

DT=torch.float64


def radial_geometry(terms,kind=PsiGeometry):
    degree=2*max(terms)
    coeff=[]
    for a,b,m,trig in native_modes(degree,0):
        coeff.append(terms.get((a+b)//2,0.)*math.comb((a+b)//2,a//2)
                     if a%2==0 and b%2==0 else 0.)
    axis=torch.zeros((8,4),dtype=DT);axis[:,0]=3.
    return kind(axis,torch.tensor(coeff,dtype=DT),nfp=1,a=1.,degree=degree,mtor=0)


def sampling(geometry,level):
    c=DownstreamConfig(nfp=1,volume_phi_count=4,volume_theta_count=16,
                       volume_point_count=128,alpha_fit_point_count=32)
    level=torch.tensor(level,dtype=DT)
    flux={'coefficients':torch.ones(1,dtype=DT),'edge_flux':level}
    cfg=SimpleNamespace(flux_boundary_tolerance=1e-8)
    return formal_volume_points(geometry,level,flux,c,cfg),c,level,flux


@pytest.mark.parametrize('terms,level,expected',[
    ({1:1.,2:4.,3:-8.},.5,2**(-.75)),
    ({1:1.,2:8.,3:-16.,4:8.},1.,math.sqrt(.5)),
])
def test_formal_points_match_replay_on_first_root_after_capacity_guard(terms,level,expected):
    geometry=radial_geometry(terms)
    (actual,selection,info),c,level,flux=sampling(geometry,level)
    expected_points=volume_points(geometry,level,flux,c,
        source_indices=selection['volume_source_indices'],
        sampling_shape=selection['volume_sampling_shape'])
    for name in ('xyz','grad_s','grad_theta','rho','volume_weight'):
        torch.testing.assert_close(actual[name].double(),expected_points[name].double(),
                                   rtol=2e-5,atol=2e-6)
    source=selection['volume_source_indices'];nr=selection['volume_sampling_shape'][2]
    radius=torch.linalg.vector_norm(torch.stack((torch.linalg.vector_norm(actual['xyz'][:,:2],dim=-1)-3.,actual['xyz'][:,2]),-1),dim=-1)
    fraction=np.sqrt(c.volume_rho_min**2+(1-c.volume_rho_min**2)*((source%nr)+.371)/nr)
    np.testing.assert_allclose(radius.numpy()/fraction,expected,rtol=2e-6,atol=2e-6)
    assert info['volume_valid_fraction']==1.
    np.testing.assert_array_equal(source,np.arange(c.volume_point_count)*info['volume_candidate_count']//c.volume_point_count)


def test_unavailable_ray_is_filtered_without_reordering_other_candidates():
    class MissingRay(PsiGeometry):
        def rays(self,phi,theta):
            p=super().rays(phi,theta).clone()
            p.reshape(-1,p.shape[-1])[0,2]=-1.
            return p
    (points,selection,info),c,_,_=sampling(radial_geometry({1:1.},MissingRay),.2)
    nr=selection['volume_sampling_shape'][2]
    available=np.arange(nr,info['volume_candidate_count'])
    expected=available[np.arange(c.volume_point_count)*len(available)//c.volume_point_count]
    np.testing.assert_array_equal(selection['volume_source_indices'],expected)
    assert info['volume_available_count']==info['volume_candidate_count']-nr
    assert torch.isfinite(points['xyz']).all()


def test_all_unavailable_rays_retain_the_formal_minimum_point_failure():
    class MissingRays(PsiGeometry):
        def rays(self,phi,theta):
            p=super().rays(phi,theta).clone();p[...,2]=-1.
            return p
    with pytest.raises(ValueError,match='volume sampler found 0 valid points; minimum is'):
        sampling(radial_geometry({1:1.},MissingRays),.2)


def test_twelve_frozen_rays_and_an_unavailable_ray_keep_their_own_results():
    fixture=json.loads((Path(__file__).parent/'fixtures/radial_first_branch.json').read_text())
    polynomial=torch.tensor([r['polynomial'] for r in fixture['rays']],dtype=DT)
    invalid=polynomial[:1].clone();invalid[:,2]=-1.
    polynomial=torch.cat((polynomial[:4],invalid,polynomial[4:]))
    geometry=SimpleNamespace(solve_rays=lambda p,l:_RadialRoot.apply(p,l,1.,1.,20,1e-12))
    radii,error=_formal_boundary_radii(geometry,polynomial,torch.tensor(fixture['level'],dtype=DT))
    mask=torch.arange(len(radii))!=4
    expected=torch.tensor([r['expected_first_root'] for r in fixture['rays']],dtype=DT)
    torch.testing.assert_close(radii[mask],expected,rtol=1e-10,atol=1e-12)
    assert torch.isinf(error[4]) and bool((error[mask]<1e-10).all())


def test_repaired_branch_response_matches_centered_difference():
    geometry=SimpleNamespace(solve_rays=lambda p,l:_RadialRoot.apply(p,l,1.,1.,20,1e-12))
    def value(parameter):
        p=torch.tensor([0,0,1,0,8,0,-16,0,8],dtype=DT)
        p=p+parameter*torch.tensor([0,0,0,0,1,0,0,0,0],dtype=DT)
        root,error=_formal_boundary_radii(geometry,p[None,:],torch.tensor(1.,dtype=DT))
        return root.sum()
    parameter=torch.tensor(0.,dtype=DT,requires_grad=True)
    derivative,=torch.autograd.grad(value(parameter),parameter)
    h=1e-5
    expected=(value(parameter.detach()+h)-value(parameter.detach()-h))/(2*h)
    torch.testing.assert_close(derivative,expected,rtol=2e-6,atol=1e-8)
