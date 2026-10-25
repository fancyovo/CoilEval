"""Flux validation changes quadrature without dropping moving-input VJPs."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from coil_eval import EvaluationConfig
from coil_eval._autograd.flux_validation import stokes_check
from coil_eval._autograd.downstream import (DownstreamConfig, downstream,
                                            flux_calibration, clebsch_modes)
from coil_eval._autograd.batch_downstream import batch_downstream
from test_autograd_downstream import CircleGeometry
from test_autograd_batch_downstream import ToyBatchField


def test_derivative_check_detects_error_hidden_by_equal_edge_flux():
    ref = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    level = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    # Both polynomials give F(1)=1, but their derivatives differ substantially.
    coeff = torch.tensor([.8, .2], dtype=torch.float64, requires_grad=True)
    bad, info = stokes_check(dict(coefficients=coeff, edge_flux=coeff.sum()), level,
        dict(reference_flux_coefficients=ref, reference_flux_level=level), 1e-3)
    assert bad and info['method']=='area' and info['fallback']
    assert info['edge_relative_error']==0
    assert info['derivative_relative_error']==pytest.approx(.2)
    assert ref.grad is None and level.grad is None and coeff.grad is None


def test_validation_disabled_and_legacy_reference_are_explicit():
    flux=dict(coefficients=torch.tensor([1.]), edge_flux=torch.tensor(1.))
    bad, info=stokes_check(flux, torch.tensor(1.), {}, None)
    assert not bad and info['validation']=='disabled'
    bad, info=stokes_check(flux, torch.tensor(1.), {}, 1e-3)
    assert not bad and info['validation']=='reference_unavailable'


@pytest.mark.parametrize('value',[0.,-1.,float('inf'),float('nan')])
def test_public_validation_tolerance_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='stokes_validation_rtol'):
        EvaluationConfig(target_volume=.05,stokes_validation_rtol=value)


def test_mixed_batch_fallback_matches_scalar_area_and_keeps_independent_vjps():
    config=replace(DownstreamConfig(),axis_sample_count=12,surface_theta_count=20,
        flux_phi_count=3,flux_theta_count=24,flux_level_count=5,flux_polynomial_degree=3,
        volume_point_count=50,volume_phi_count=3,volume_theta_count=5,alpha_fit_point_count=40,
        alpha_radial_order=2,alpha_poloidal_order=2,alpha_toroidal_order=1,iota_degree=1,
        radial_bin_count=4,alpha_chunk_columns=5)
    values=torch.tensor([[1.2,.03,.8],[1.3,.04,1.1]],dtype=torch.float64,requires_grad=True)
    front=SimpleNamespace(levels=values[:,1],geometry=lambda i:CircleGeometry(values[i,0]))
    class SelectedToyField(ToyBatchField):
        selected=[]
        def select_queries(self, indices):
            self.selected.append(indices.tolist())
            return type(self)(self.amplitude.index_select(0,indices))
    field=SelectedToyField(values[:,2])
    columns=len(clebsch_modes(2,2,1)[0])+2
    caches=[]
    for i in range(2):
        one=field.one(i)
        stokes=flux_calibration(front.geometry(i),front.levels[i],one,one.potential,config)
        reference=stokes['coefficients'].detach()*(1 if i==0 else 2)
        caches.append(dict(upper=torch.eye(columns,dtype=torch.float64),
            scale=torch.ones(columns,dtype=torch.float64),
            scaled_solution=torch.zeros(columns,dtype=torch.float64),
            reference_flux_coefficients=reference,
            reference_flux_level=front.levels[i].detach()))
    actual=batch_downstream(front,caches,field,[config,config],return_arrays=True)
    assert field.selected==[[1]]
    assert [v['method'] for v in actual['flux_validation']]==['stokes','area']
    expected=[]
    for i in range(2):
        one=field.one(i)
        cfg=replace(config,stokes_flux=(i==0),stokes_validation_rtol=None)
        expected.append(downstream(front.geometry(i),front.levels[i],caches[i],
                                   one,one.trace,one.potential,cfg,return_arrays=True))
    for name in ('flux_coefficients','flux_edge','surface_volume','raw_qs','raw_qs_outer',
                 'iota_boundary','target_quality_error'):
        scalar=torch.stack([e[name] for e in expected])
        torch.testing.assert_close(actual[name],scalar,rtol=2e-9,atol=2e-10)
        got=torch.autograd.grad(actual[name].sum(),values,retain_graph=True)[0]
        wanted=torch.autograd.grad(scalar.sum(),values,retain_graph=True)[0]
        torch.testing.assert_close(got,wanted,rtol=2e-8,atol=2e-9)
    selected=torch.autograd.grad(actual['flux_edge'][1],values,retain_graph=True)[0]
    assert torch.count_nonzero(selected[0])==0
    assert bool(torch.all(selected[1].abs()>0))
    one=field.one(1)
    scalar_fallback=downstream(front.geometry(1),front.levels[1],caches[1],
                               one,one.trace,one.potential,config)
    assert scalar_fallback['flux_validation']['method']=='area'
    torch.testing.assert_close(scalar_fallback['raw_qs'],actual['raw_qs'][1])
