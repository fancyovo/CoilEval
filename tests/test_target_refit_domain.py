"""Target-interior selection and implicit moving-domain derivatives."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from coil_eval import EvaluationConfig
from coil_eval._autograd.front import PsiGeometry, native_modes
from coil_eval._autograd.target_domain import target_sampling_geometry
from coil_eval._two_stage.model import PsiModel
from coil_eval._two_stage.boundary import max_closed_boundary, MaxClosedBoundaryError
from coil_eval._two_stage.target_domain import target_training_domain, polygon_volume


def setup():
    nfp, a, degree, mtor = 3, .15, 4, 1
    modes = native_modes(degree=degree, mtor=mtor)
    coeff = np.zeros(len(modes))
    coeff[modes.index((0, 2, 0, 0))] = 1.3
    coeff[modes.index((3, 0, 0, 0))] = .04
    coeff[modes.index((4, 0, 0, 0))] = -.8
    coeff[modes.index((4, 0, 1, 0))] = -.08
    angle = np.arange(12)*2*np.pi/(nfp*12)
    axis = np.column_stack((1+.012*np.cos(nfp*angle), .008*np.sin(nfp*angle),
                            -.012*nfp*np.sin(nfp*angle), .008*nfp*np.cos(nfp*angle)))
    phis = np.arange(8)*2*np.pi/(nfp*8)
    def model(c=coeff, ax=axis):
        return PsiModel(c, ax, nfp=nfp, a=a, degree=degree, mtor=mtor)
    return coeff, axis, phis, model


@pytest.mark.parametrize('fraction,clipped',[(.35, False), (.94, True)])
def test_target_volume_domain_and_live_derivative_match_independent_rebuild(fraction, clipped):
    coeff, axis, phis, model = setup()
    cap = max_closed_boundary(model(), phis, ntheta=32, workers=1)
    target = fraction*cap['volume_m3']
    def rebuild(c=coeff, ax=axis):
        return target_training_domain(model(c, ax), phis, target,
            ntheta=32, theta_count=16, target_count=1024, workers=1)
    metadata = rebuild()
    boundary = metadata['boundary']
    assert boundary['diagnostics']['capacity_clipped'] == clipped
    desired = min(1.2*target, .5*(target+cap['volume_m3']))
    assert boundary['volume_m3'] == pytest.approx(desired, rel=2e-11)
    assert boundary['volume_m3'] > target
    assert np.min(metadata['sampling']['radial_u']) < .1
    assert np.max(metadata['sampling']['radial_u']) > .9
    metadata['stage2_coefficients'] = coeff
    rng = np.random.default_rng(631)
    dc = rng.normal(size=len(coeff))*.003
    dax = rng.normal(size=axis.shape)*.0002
    weights = rng.normal(size=metadata['sampling']['rz'].shape)
    row_weights = rng.normal(size=metadata['sampling']['row_weights'].shape)
    parameter = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    def live(parameter, local=False):
        geom = PsiGeometry(torch.tensor(axis)+parameter*torch.tensor(dax),
            torch.tensor(coeff)+parameter*torch.tensor(dc), nfp=3, a=.15, degree=4, mtor=1)
        return target_sampling_geometry(geom, metadata, reference_axis=torch.tensor(axis), local_value=local)
    def value(domain):
        return ((domain['rz']*torch.tensor(weights)).sum()
                + .03*(domain['row_weights']*torch.tensor(row_weights)).sum()+domain['volume'])
    actual = live(parameter)
    torch.testing.assert_close(actual['rz'], torch.tensor(metadata['sampling']['rz']), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual['row_weights'], torch.tensor(metadata['sampling']['row_weights']), rtol=1e-13, atol=1e-13)
    gradient, = torch.autograd.grad(value(actual), parameter)
    def independent(t):
        domain = rebuild(coeff+t*dc, axis+t*dax)
        return ((domain['sampling']['rz']*weights).sum()
                + .03*(domain['sampling']['row_weights']*row_weights).sum()
                + polygon_volume(domain['boundary']['polygons_rz']))
    h = 2e-5
    expected = (independent(h)-independent(-h))/(2*h)
    assert float(gradient) == pytest.approx(expected, rel=3e-4, abs=2e-6)
    local_fd = (value(live(torch.tensor(h), True))-value(live(torch.tensor(-h), True)))/(2*h)
    torch.testing.assert_close(gradient, local_fd, rtol=3e-5, atol=2e-7)
    assert abs(float(gradient)) > 1e-5


def test_capacity_failure_does_not_reduce_target_or_return_outer_root():
    _, _, phis, model = setup()
    cap = max_closed_boundary(model(), phis, ntheta=32, workers=1)
    with pytest.raises(MaxClosedBoundaryError, match='capacity'):
        target_training_domain(model(), phis, 1.01*cap['volume_m3'],
            ntheta=32, theta_count=16, workers=1)


@pytest.mark.parametrize('options',[
    {'psi_target_refit': 1}, {'psi_target_refit': True, 'psi_fit': 'single_stage'},
    {'psi_target_refit_factor': 1.}, {'psi_target_refit_factor': float('nan')},
    {'psi_target_refit_ridge': -1}, {'psi_target_refit_ridge': True}])
def test_bad_target_refit_config_is_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        EvaluationConfig(target_volume=.02, **options)
