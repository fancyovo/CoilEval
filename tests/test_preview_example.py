"""Guard the registered perturbation and exact scalar/tensor objective agreement."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def example(monkeypatch):
    pytest.importorskip('torch')
    path=Path(__file__).resolve().parents[1]/'examples/preview'
    monkeypatch.syspath_prepend(str(path))
    spec=importlib.util.spec_from_file_location('preview_example_contract',path/'perturb_optimize.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fixed32_noise_and_amplitude_retry_keep_draws_and_currents(example):
    first,sha=example.candidates(.01);half,sha_half=example.candidates(.005)
    base,_=example.candidates(0.)
    assert first.shape==(32,4,28) and sha==sha_half
    np.testing.assert_array_equal(first[:,:,-1],base[:,:,-1])
    np.testing.assert_array_equal(np.sign(first[:,:,:27]),np.sign(base[:,:,:27]))
    z=np.random.default_rng(2026110502).standard_normal((32,4,27))
    np.testing.assert_allclose(first[:,:,:27],base[:,:,:27]*np.exp(math.log(1.01)*z),rtol=1e-14,atol=0)
    np.testing.assert_allclose(np.log(first[:,:,:27]/base[:,:,:27])/math.log1p(.01),
        np.log(half[:,:,:27]/base[:,:,:27])/math.log1p(.005),rtol=1e-10,atol=1e-12)
    assert example.CONFIG.psi_fit=='two_stage' and not example.CONFIG.psi_target_refit


def test_complete_formal_loss_matches_single_reverse_tensor_definition(example):
    import torch
    values=dict(qs_volume=.017,iota_boundary=.451,coil_gap_min=.093,
        coil_curvature_max=6.2,coil_length_mean=3.54,surface_volume=.3,
        relative_normal_drift_p95=.013)
    m=SimpleNamespace(**{k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in values.items()})
    r=torch.tensor(1.021,dtype=torch.float64,requires_grad=True)
    quality=10000*(.0025*np.logaddexp(0.,(.013-.01)/.0025))**2
    d=dict(qs_target_global_error_per_helicity=.017,iota_target_surface=.451,
        coil_min_intercoil_distance=.093,coil_curvature_max=6.2,coil_length_mean=3.54,
        surface_volume=.3,surface_effective_minor_radius=.1,surface_inverse_aspect_ratio=.1/1.021,
        target_quality_penalty=quality)
    scalar=example.formal_objective(dict(status='ok',diagnostics=d))
    total=example.tensor_objective(m,r)
    assert float(total.detach())==pytest.approx(scalar['loss'],rel=2e-14,abs=2e-14)
    grad,=torch.autograd.grad(total,r)
    assert float(grad)>0
    assert not example.formal_objective(dict(status='no_surface'))['valid']
