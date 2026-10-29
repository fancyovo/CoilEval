"""CPU interfaces preserve field periods, coil layout and requested helicity."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('scipy')
pytest.importorskip('threadpoolctl')

from coil_eval import CoilSet, EvaluationConfig
from coil_eval._cpu.tensor import CpuTensorEvaluator
from coil_eval._cpu.qs import CpuSurfaceQSEvaluator
from coil_eval.qs import SurfaceQSConfig


def coils(nfp, count):
    x=np.zeros((count,3));x[:,0]=1.;x[:,2]=.2
    y=np.zeros_like(x);y[:,1]=.2
    return CoilSet(x,y,np.zeros_like(x),np.full(count,1e5),nfp)


@pytest.mark.parametrize('nfp',[0,-1,True,np.bool_(True),2.,'2'])
def test_cpu_tensor_rejects_invalid_periods_before_loading(nfp,monkeypatch):
    monkeypatch.setattr('coil_eval._cpu.tensor.CpuBackend',lambda *a,**k:pytest.fail('backend loaded'))
    config=EvaluationConfig(.05,fast=True,flux_method='stokes')
    with pytest.raises(ValueError,match='positive integer'):
        CpuTensorEvaluator(config=config,nfp=nfp)


@pytest.mark.parametrize('nfp,count,batch',[(1,1,False),(2,3,True),(5,6,False)])
def test_cpu_tensor_capture_receives_requested_geometry(nfp,count,batch,monkeypatch):
    class Captured(Exception):pass
    seen=[]
    class Backend:
        threads=1
        library=SimpleNamespace(path='unused')
        def __init__(self,*a,**kw):pass
        def _capture(self,coil,config):
            seen.append((coil.nfp,coil.n_base_coils,config.helicity))
            raise Captured
    monkeypatch.setattr('coil_eval._cpu.tensor.CpuBackend',Backend)
    monkeypatch.setattr('coil_eval._cpu.tensor.CpuAutogradLibrary',lambda *a:None)
    config=EvaluationConfig(.05,fast=True,flux_method='stokes',helicity=(1,0))
    engine=CpuTensorEvaluator(config=config,nfp=nfp)
    parameters=torch.from_numpy(np.column_stack((coils(nfp,count).coeffs_x,
        coils(nfp,count).coeffs_y,coils(nfp,count).coeffs_z,coils(nfp,count).currents_a)))
    with pytest.raises(Captured):
        engine.evaluate_batch(parameters[None]) if batch else engine.evaluate(parameters)
    assert seen==[(nfp,count,(1,0))]


def test_cpu_surface_value_batch_preserves_mixed_periods_and_coil_counts(monkeypatch):
    import coil_eval._cpu.qs as module
    config=EvaluationConfig(.05,fast=True,flux_method='stokes',helicity=(1,0))
    inputs=[coils(1,1),coils(2,3),coils(5,2)]
    captured=[];sampled=[]
    def capture(coil,cfg):
        captured.append((coil.nfp,coil.n_base_coils))
        return {'timing':{}},{'config':{'nfp':coil.nfp,'psi_a':.2,'psi_poly_degree':2,'psi_m_tor':0},
            'axis_samples':np.tile([1.,0.,0.,0.],(8,1)),'psi_coefficients':np.array([0.,1.]),
            'alpha_coefficients':np.array([0.]),'iota_coefficients':np.array([.3]),'target_level':.1}
    engine=SimpleNamespace(config=config,threads=1,surface_qs=SurfaceQSConfig(),
        field_library=None,backend=SimpleNamespace(_capture=capture),_check=lambda:None)
    evaluator=object.__new__(CpuSurfaceQSEvaluator);evaluator._evaluator=engine
    monkeypatch.setattr(module,'CpuSegmentField',lambda p,w,library:SimpleNamespace(positions=p))
    def sample(geometry,level,physics,field,cfg,**kwargs):
        sampled.append((geometry.nfp,cfg.nfp,len(field.positions)))
        theta=torch.arange(17,dtype=torch.float64)*2*torch.pi/17
        # A purely poloidal B spectrum is exactly quasi-axisymmetric.
        magnitude=(1+.1*theta.cos())[None].expand(9,-1)
        return dict(B_magnitude=magnitude,iota=torch.tensor(.3),G=torch.tensor(1.),
                    major_radius=torch.tensor(1.),diagnostics={})
    monkeypatch.setattr(module,'sample_boozer_surface',sample)
    results=evaluator.evaluate_many(inputs)
    assert captured==[(1,1),(2,3),(5,2)]
    assert sampled==[(c.nfp,c.nfp,c.n_base_coils*2*c.nfp*256) for c in inputs]
    assert [r.diagnostics['nfp'] for r in results]==[1,2,5]
    assert all(r.f_B==pytest.approx(0.,abs=1e-25) for r in results)
