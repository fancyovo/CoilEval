"""CPU numerical contracts that do not need a GPU or production-size solve."""
import math
import os
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('scipy')
pytest.importorskip('threadpoolctl')
from coil_eval._cpu.threads import available_threads
from coil_eval._cpu.fitting import regularized_qr


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','F','sliced'])
def test_column_copy_matches_original_strided_assignment_bitwise(dtype,layout):
    from coil_eval._cpu.fitting import _copy_columns
    rng=np.random.default_rng(238)
    raw=np.array(rng.normal(size=(513,18)),dtype=dtype,order='F' if layout=='F' else 'C')
    source=raw[:,1:] if layout=='sliced' else raw
    rows,cols=source.shape
    expected=np.zeros((rows+cols,cols),dtype=dtype,order='F')
    actual=expected.copy(order='F')
    expected[:rows]=source;_copy_columns(actual[:rows],source)
    np.testing.assert_array_equal(actual,expected)
    scale=np.linspace(.1,8,cols,dtype=dtype)
    expected[:rows]=source/scale;_copy_columns(actual[:rows],source,scale)
    np.testing.assert_array_equal(actual,expected)


def test_threads_respect_process_allocation(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','3')
    monkeypatch.setattr(os,'cpu_count',lambda:16)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:set(range(6)),raising=False)
    assert available_threads()==3
    assert available_threads(2)==2
    for invalid in (0,-1,True,1.5,'2'):
        with pytest.raises(ValueError):available_threads(invalid)


def test_cpu_axis_grid_validates_before_library_loading(monkeypatch):
    from coil_eval._cpu.backend import CpuBackend
    monkeypatch.setattr('coil_eval._cpu.backend.CpuLibrary',lambda *a:object())
    for value in (None,24,32,48):assert CpuBackend(axis_grid=value).axis_grid==value
    for value in (True,3,0,-5,24.,'24'):
        with pytest.raises(ValueError,match='axis_grid'):CpuBackend(axis_grid=value)


def test_thread_scope_caps_openblas_growth_and_restores_providers():
    from threadpoolctl import threadpool_info
    from coil_eval._cpu.threads import thread_scope
    before=threadpool_info();torch_before=torch.get_num_threads()
    def sizes(pools):return {p['filepath']:p['num_threads'] for p in pools}
    with thread_scope(2) as state:
        assert torch.get_num_threads()==2
        original=sizes(before)
        for pool in state['active_pools']:
            if pool['internal_api']=='openblas':assert pool['num_threads']<=original[pool['filepath']]
        outer=sizes(threadpool_info())
        with thread_scope(4):
            assert torch.get_num_threads()==4
            for pool in threadpool_info():
                if pool['internal_api']=='openblas':assert pool['num_threads']<=outer[pool['filepath']]
        assert sizes(threadpool_info())==outer
    assert sizes(threadpool_info())==sizes(before)
    assert torch.get_num_threads()==torch_before


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('driver',['scipy','torch'])
def test_scaled_householder_qr_matches_augmented_problem(dtype,driver):
    rng=np.random.default_rng(19)
    matrix=(rng.normal(size=(130,13))*np.logspace(-2,2,13)).astype(dtype)
    rhs=rng.normal(size=130).astype(dtype)
    actual,cache=regularized_qr(matrix,rhs,1e-3,driver=driver)
    scale=np.sqrt((matrix.astype(float)**2).sum(0))
    augmented=np.vstack((matrix/scale,np.sqrt(1e-3)*np.eye(13)))
    expected=np.linalg.lstsq(augmented,np.r_[rhs,np.zeros(13)],rcond=None)[0]/scale
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-6)
    np.testing.assert_allclose(cache['upper'].T@cache['upper'],augmented.T@augmented,rtol=4e-5,atol=1e-6)


def test_target_volume_and_adjoint_on_circular_torus():
    from coil_eval._cpu.native import CpuLibrary
    from coil_eval._cpu.geometry import target_surface
    from coil_eval._autograd.front import PsiGeometry,predict_target_level
    try:library=CpuLibrary()
    except RuntimeError:pytest.skip('CPU native library not installed')
    a=.2;major=1.1;target=.05
    axis=torch.tensor([[major,0.,0.,0.]]*16,dtype=torch.float64)
    # degree2 modes are X^2(fixed), XY and Y^2.
    coeff=torch.tensor([0.,1.],dtype=torch.float64)
    geom=PsiGeometry(axis,coeff,nfp=4,a=a,degree=2,mtor=0)
    cfg=SimpleNamespace(flux_phi_count=8,psi_n_phi=12,psi_m_tor=0,flux_theta_count=64,
        surface_max_radius_scale=1.,target_volume=target,target_volume_relative_tolerance=1e-10)
    level,snapshot,info=target_surface(geom,cfg,library)
    expected=target/(2*math.pi**2*major*a*a)
    assert float(level)==pytest.approx(expected,rel=1e-8)
    assert info['target_surface_volume']==pytest.approx(target,rel=1e-9)
    snapshot['psi_coefficients']=coeff
    live_coeff=coeff.clone().requires_grad_()
    live_axis=axis.clone().requires_grad_()
    predicted=predict_target_level(PsiGeometry(live_axis,live_coeff,nfp=4,a=a,degree=2,mtor=0),snapshot)
    dc,da=torch.autograd.grad(predicted,(live_coeff,live_axis))
    assert float(dc[1])==pytest.approx(expected/2,rel=1e-8)
    assert float(da[:,0].sum())==pytest.approx(-expected/major,rel=1e-8)


def test_cpu_dispatch_does_not_resolve_cuda(monkeypatch):
    from coil_eval import Evaluator,EvaluationConfig
    from coil_eval import _native_loader
    monkeypatch.setattr(_native_loader,'resolve_library',lambda *a,**kw:pytest.fail('CUDA loader invoked'))
    class Backend:
        def __init__(self,*a,**kw):self.threads=kw['threads']
        def close(self):pass
    monkeypatch.setattr('coil_eval._cpu.backend.CpuBackend',Backend)
    with Evaluator(config=EvaluationConfig(.05),device='cpu',threads=2) as evaluator:
        assert evaluator.device=='cpu'
        assert evaluator._backend.threads==2


def test_tensor_and_surface_cpu_dispatch_do_not_resolve_cuda(monkeypatch):
    from coil_eval import EvaluationConfig
    from coil_eval import _native_loader
    from coil_eval.torch import TensorEvaluator
    from coil_eval.qs import SurfaceQSEvaluator
    monkeypatch.setattr(_native_loader,'resolve_library',lambda *a,**kw:pytest.fail('CUDA loader invoked'))
    calls=[]
    class Engine:
        def __init__(self,*a,**kw):calls.append(kw)
    monkeypatch.setattr('coil_eval._cpu.tensor.CpuTensorEvaluator',Engine)
    monkeypatch.setattr('coil_eval._cpu.qs.CpuSurfaceQSEvaluator',Engine)
    config=EvaluationConfig(.05,fast=True,flux_method='stokes')
    assert isinstance(TensorEvaluator(config=config,device='cpu',threads=3),Engine)
    assert isinstance(SurfaceQSEvaluator(config=config,device='cpu',threads=4),Engine)
    assert [c['threads'] for c in calls]==[3,4]
    for factory in (TensorEvaluator,SurfaceQSEvaluator):
        with pytest.raises(ValueError,match='device must'):factory(config=config,device='tpu')
        with pytest.raises(ValueError,match='threads'):factory(config=config,threads=2)


@pytest.mark.parametrize('two_stage',[False,True])
@pytest.mark.parametrize('nfp,ncoils,helicity',[(4,4,(1,4)),(2,1,(1,0)),(3,2,(1,-3)),(5,6,(1,5))])
def test_complete_cpu_pipeline_on_exact_nested_field(monkeypatch,two_stage,nfp,ncoils,helicity):
    """Exercise all stage interfaces against a known circular flux foliation."""
    from pathlib import Path
    from coil_eval import CoilSet,EvaluationConfig
    from coil_eval._cpu.backend import CpuBackend
    try:backend=CpuBackend(threads=1)
    except RuntimeError:pytest.skip('CPU native library not installed')
    original=backend.library.configuration
    def small_config(config,nfp):
        cfg=original(config,nfp)
        settings=dict(psi_poly_degree=2,psi_m_tor=0,psi_n_r=12,psi_n_z=12,psi_n_phi=8,
            axis_sample_count=24,flux_phi_count=8,flux_theta_count=32,flux_radial_quadrature=8,
            flux_level_count=5,volume_point_count=2048,volume_phi_count=8,volume_theta_count=16,
            alpha_fit_point_count=1024,alpha_radial_order=2,alpha_poloidal_order=2,
            alpha_toroidal_order=1,surface_theta_count=32,radial_bin_count=4)
        for k,v in settings.items():setattr(cfg,k,v)
        return cfg
    monkeypatch.setattr(backend.library,'configuration',small_config)
    class Field:
        def __init__(self,*a,**kw):pass
        def close(self):pass
        def axis(self,cfg):return np.array([[1.1,0.,0.,0.]]*cfg.axis_sample_count),{},None
        @staticmethod
        def value(q):
            x,y,z=q.unbind(-1);R=(x*x+y*y).sqrt();c,s=x/R,y/R
            br=-.3*z/R;bz=.3*(R-1.1)/R
            return torch.stack((br*c-s,br*s+c,bz),-1)
        def __call__(self,q,gradient=False):
            b=self.value(q).float()
            if gradient:return b,torch.vmap(torch.func.jacrev(self.value))(q).float()
            return b
        def potential(self,q):raise AssertionError('formal pipeline uses area flux')
        def trace(self,rz,p0,p1,steps,return_samples=False):
            def at(t):
                angle=.3*t;x=rz[:,0]-1.1;z=rz[:,1]
                return torch.stack((1.1+x*math.cos(angle)-z*math.sin(angle),
                    x*math.sin(angle)+z*math.cos(angle)),-1)
            return torch.stack([at((p1-p0)*k/steps) for k in range(steps+1)]) if return_samples else at(p1-p0)
    monkeypatch.setattr('coil_eval._cpu.backend.CpuField',Field)
    example=CoilSet.read_json(Path(__file__).parents[1]/'examples/coils.json')
    selection=np.arange(ncoils)%example.n_base_coils
    coils=CoilSet(*(a[selection] for a in (example.coeffs_x,example.coeffs_y,
        example.coeffs_z,example.currents_a)),nfp=nfp)
    config=EvaluationConfig(.01,fast=True,helicity=helicity,psi_fit='two_stage' if two_stage else 'single_stage',
        psi_refit_points=1024,psi_refit_sampling='polar_quadrature')
    raw,snapshot=backend.capture(coils,config)
    assert raw['status']=='ok',raw
    assert raw['diagnostics']['iota_target_surface']==pytest.approx(-.3,abs=2e-4)
    assert raw['diagnostics']['target_quality_error']<1e-4
    assert snapshot['volume_source_indices'].shape==(2048,)
    assert ('two_stage' in snapshot)==two_stage
    assert snapshot['config']['nfp']==nfp
    assert (snapshot['config']['target_M'],snapshot['config']['target_N'])==helicity
    assert raw['diagnostics']['surface_volume']==pytest.approx(config.target_volume,rel=2e-5)
