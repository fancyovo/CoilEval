import numpy as np
import pytest

from coil_eval._two_stage import (PsiModel, build_modes, first_connected_branch,
                                 max_closed_boundary, sample_frozen_sections,
                                 shifted_holdout)
from coil_eval._two_stage.polynomial import ray_polynomials
from coil_eval._two_stage.branches import branch_limits
from coil_eval._two_stage.branches import resolve_workers
import _two_stage_reference as reference


def model(*,flat=4.,fold=False,degree=6,mtor=2):
    modes=build_modes(degree,mtor)
    coefficients=np.zeros(len(modes))
    for i,m in enumerate(modes):
        if (m.a,m.b,m.m,m.kind)==(0,2,0,'cos'):coefficients[i]=flat
        if fold and (m.a,m.b,m.m,m.kind)==(4,0,0,'cos'):coefficients[i]=-.8
    phi=np.arange(32)*2*np.pi/(4*32)
    axis=np.column_stack((1.+.01*np.cos(4*phi),.02*np.sin(4*phi),-.04*np.sin(4*phi),.08*np.cos(4*phi)))
    return PsiModel(coefficients,axis,nfp=4,a=.3,degree=degree,mtor=mtor)


@pytest.mark.parametrize('flat,fold',[(1.,False),(16.,False),(4.,True)])
def test_geometry_matches_scalar_reference(flat,fold):
    current=model(flat=flat,fold=fold)
    phi=np.arange(12)*2*np.pi/(4*12)
    theta=np.arange(64)*2*np.pi/64
    np.testing.assert_allclose(ray_polynomials(current,phi,theta),reference._ray_polynomials(current,phi,theta),rtol=2e-13,atol=2e-14)
    actual=max_closed_boundary(current,phi,ntheta=64)
    expected=reference.max_closed_boundary(current,phi,ntheta=64)
    np.testing.assert_allclose(actual['radii'],expected['radii'],rtol=2e-12,atol=2e-14)
    assert actual['diagnostics']['limiting_ray']['reason']==expected['diagnostics']['limiting_ray']['reason']
    assert actual['volume_m3']==pytest.approx(expected['volume_m3'],rel=2e-12)


def test_flat_stationary_point_is_not_a_fold():
    result=first_connected_branch([0.,0.,.125,-1/3,.25],1.)
    assert result['reason']=='radius_cap'
    assert result['flat_nonfold_count']==1


def test_quadratic_basis_has_no_critical_roots():
    current=model(degree=2,mtor=0)
    phi=np.arange(4)*2*np.pi/(4*4)
    actual=max_closed_boundary(current,phi,ntheta=16)
    expected=reference.max_closed_boundary(current,phi,ntheta=16)
    np.testing.assert_allclose(actual['radii'],expected['radii'],rtol=1e-12)


def test_batched_critical_roots_match_scalar_policy():
    rng=np.random.default_rng(1107)
    p=np.zeros((64,11));p[:,2]=rng.uniform(.2,2.,len(p));p[:,3:]=rng.normal(0,.3,(len(p),8))
    caps=np.ones(len(p))
    actual,folds,flat,_=branch_limits(p,caps,first_connected_branch)
    expected=[reference.first_connected_branch(row,1.) for row in p]
    np.testing.assert_allclose(actual,[x['maximum_u'] for x in expected],rtol=5e-12,atol=1e-13)
    np.testing.assert_array_equal(folds,[x['first_fold_u'] is not None for x in expected])
    np.testing.assert_array_equal(flat,[x['flat_nonfold_count'] for x in expected])


def test_batched_repeated_roots_keep_flat_nonfold_policy():
    polynomial=np.asarray([[0.,0.,.125,-1/3,.25]])
    actual,folds,flat,count=branch_limits(polynomial,np.ones(1),first_connected_branch)
    np.testing.assert_allclose(actual,1.)
    assert not folds[0] and flat[0]==1
    assert count==1


def test_geometry_workers_respect_slurm_cpu_allowance(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','2')
    monkeypatch.setenv('COIL_EVAL_GEOMETRY_WORKERS','4')
    assert resolve_workers()==(4,2)
    assert resolve_workers(1)==(1,1)
    for bad in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='positive integer'):resolve_workers(bad)


def test_parallel_eigenvalue_chunks_preserve_boundary_bitwise(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    current=model(degree=10,mtor=3)
    rng=np.random.default_rng(107)
    current.coeffs+=rng.normal(0,1e-4,current.coeffs.shape)
    phi=np.arange(12)*2*np.pi/(4*12)
    expected=max_closed_boundary(current,phi,ntheta=64,workers=1)
    for workers in (2,4):
        actual=max_closed_boundary(current,phi,ntheta=64,workers=workers)
        np.testing.assert_array_equal(actual['radii'],expected['radii'])
        np.testing.assert_array_equal(actual['normalized_branch_caps'],expected['normalized_branch_caps'])
        assert actual['diagnostics']['geometry_workers_effective']==workers


def test_scanline_strict_edges_and_concave_polygons():
    rng=np.random.default_rng(710)
    polygons=[np.asarray([[.5,-.5],[1.5,-.5],[1.5,.5],[1.,.5],[1.,0],[.5,0]])]
    for _ in range(12):
        angles=np.arange(32)*2*np.pi/32;radii=rng.uniform(.1,.4,32)
        polygons.append(np.column_stack((1+radii*np.cos(angles),radii*np.sin(angles))))
    phi=np.arange(len(polygons))*2*np.pi/(4*len(polygons))
    for shift in ((0.,0.),(.5,.5),(.12,.72)):
        actual=sample_frozen_sections(polygons,phi,nfp=4,spacing=.0625,lattice_shift=shift)
        expected=reference.sample_frozen_sections(polygons,phi,nfp=4,spacing=.0625,lattice_shift=shift)
        np.testing.assert_array_equal(actual['rz'],expected['rz'])
        np.testing.assert_array_equal(actual['section_index'],expected['section_index'])
        np.testing.assert_allclose(actual['spacing']*(actual['lattice_indices']+actual['lattice_shift']),actual['rz'],rtol=0,atol=1e-15)


@pytest.mark.parametrize('shift',[(.5,.5),(0.,0.),(.12,.72)])
def test_sampler_preserves_reference_order_and_points(shift):
    current=model(flat=16.,fold=False)
    phi=np.arange(8)*2*np.pi/(4*8)
    boundary=max_closed_boundary(current,phi,ntheta=64)
    kwargs=dict(target_count=4000,nfp=4,lattice_shift=shift)
    actual=sample_frozen_sections(boundary['polygons_rz'],phi,**kwargs)
    expected=reference.sample_frozen_sections(boundary['polygons_rz'],phi,**kwargs)
    np.testing.assert_array_equal(actual['rz'],expected['rz'])
    np.testing.assert_array_equal(actual['section_index'],expected['section_index'])
    assert actual['spacing']==expected['spacing']
    held=shifted_holdout(boundary['polygons_rz'],phi,actual,nfp=4)
    assert held['spacing']==actual['spacing']
    assert held['metadata']['holdout']['training_points_sha256']==actual['metadata']['points_sha256']
