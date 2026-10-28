"""Independent Boozer batches preserve scalar branches and parameter VJPs."""
import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd.batch_surface_qs import sample_boozer_surfaces
from coil_eval._autograd.boozer import sample_boozer_surface
from coil_eval._autograd.downstream import DownstreamConfig, clebsch_modes
from coil_eval._autograd.front import PsiGeometry
from coil_eval.qs_metrics import boozer_qs_metrics_field, boozer_spectrum_metrics


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(parameters):
    config = DownstreamConfig(alpha_radial_order=2, alpha_poloidal_order=2,
                             alpha_toroidal_order=1, alpha_radial_basis='flux_legendre')
    count = len(clebsch_modes(2, 2, 1, 'flux_legendre')[0])
    direction = torch.sin(torch.arange(count, dtype=parameters.dtype) + .4)
    geometries, levels, physics = [], [], []
    for major, ellipticity, level, amplitude, iota, scale in parameters.unbind():
        zero = major * 0
        axis = torch.stack((major.expand(8), zero.expand(8),
                            zero.expand(8), zero.expand(8)), -1)
        geometries.append(PsiGeometry(axis, torch.stack((zero, ellipticity)),
                                      nfp=4, a=.5, degree=2, mtor=0))
        levels.append(level)
        physics.append(dict(alpha_coefficients=amplitude*direction,
                            iota_coefficients=iota[None], major_radius=major))
    return geometries, levels, physics, [config] * len(parameters)


def magnetic(points, parameters, gradient=False):
    """Tangent field with nonzero, nonresonant rotational transform.

    B=(scale/R**2)*(x_phi+iota*x_theta) on each manufactured elliptical
    psi surface. A purely toroidal field would make the target-alpha transport
    fit rank deficient in every poloidal-only mode, so it is unsuitable for
    testing refinement. The analytic field below keeps that physical guard.
    """
    major, ellipticity, _, _, iota, scale = parameters.unbind(-1)
    x, y, z = points.unbind(-1)
    r2 = x*x+y*y
    R = r2.sqrt()
    offset = R-major
    polar_speed = (offset.square()+z.square())/(offset.square()+ellipticity*z.square())
    br = scale*iota*ellipticity*z*polar_speed/r2
    bz = -scale*iota*offset*polar_speed/r2
    B = torch.stack((br*x/R-scale*y/r2, br*y/R+scale*x/r2, bz), -1)
    if not gradient:
        return B
    # Each point depends only on its own coordinates. Three exact forward AD
    # sweeps give the Cartesian Jacobian while leaving its parameter VJP live.
    basis = torch.eye(3, dtype=points.dtype, device=points.device)
    J = torch.stack([torch.func.jvp(lambda xyz: magnetic(xyz, parameters),
        (points,), (direction.expand_as(points),))[1] for direction in basis], -1)
    return B, J


def objective(sample, derivatives):
    values = (boozer_qs_metrics_field(sample['B'], sample['dBdx'],
        sample['x_theta'], sample['x_phi'], nfp=4, helicity=(1, 4),
        iota=sample['iota'], G=sample['G'], major_radius=sample['major_radius'])
        if derivatives else boozer_spectrum_metrics(sample['B_magnitude'], nfp=4,
                                                     helicity=(1, 4)))
    value = values['f_B'] + .03*sample['G'] + .07*sample['iota']
    return value + values['f_T'] if derivatives else value


@pytest.mark.parametrize('refinement', [None, 'geometric', 'straight'])
@pytest.mark.parametrize('nu_method', ['transport', 'potential'])
@pytest.mark.parametrize('derivatives', [False, True])
def test_batch_field_sweeps_match_independent_values_and_all_parameter_vjps(
        refinement, nu_method, derivatives):
    parameters = torch.tensor([
        [2., 1.1, .16, .003, .71, 1.3],
        [2.2, .95, .18, -.002, .83, 1.5],
        [1.9, 1.02, .13, .001, .57, 1.1]], dtype=torch.float64, requires_grad=True)
    geometries, levels, physics, configs = inputs(parameters)
    options = dict(nphi=9, ntheta=11, nu_order=2, nu_method=nu_method,
        field_derivatives=derivatives, alpha_order=2 if refinement else 0,
        alpha_coordinates=refinement or 'straight')
    calls = []

    def batch_field(points, gradient=False):
        calls.append((points.shape, gradient))
        return magnetic(points, parameters[:, None, :], gradient)

    batched = sample_boozer_surfaces(geometries, levels, physics, batch_field,
                                    configs, **options)
    references = [sample_boozer_surface(geometry, level, phys,
        lambda points, gradient=False, i=i: magnetic(points, parameters[i], gradient),
        config, **options) for i, (geometry, level, phys, config) in
        enumerate(zip(geometries, levels, physics, configs))]
    assert len(calls) == (3 if refinement else 2)
    assert all(shape[0] == len(parameters) for shape, _ in calls)
    assert [gradient for _, gradient in calls] == [False]*(len(calls)-1)+[derivatives]
    for actual, expected in zip(batched, references):
        for key in ('B_magnitude', 'xyz', 'x_theta', 'x_phi', 'G', 'iota'):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-13)
        if derivatives:
            torch.testing.assert_close(actual['dBdx'], expected['dBdx'], rtol=1e-12, atol=1e-13)
        for key, value in actual['diagnostics'].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected['diagnostics'][key], rtol=1e-12, atol=1e-13)
            elif key != 'surface_alpha_solver':
                assert value == expected['diagnostics'][key]
    values = torch.stack([objective(sample, derivatives) for sample in batched])
    reference_values = torch.stack([objective(sample, derivatives) for sample in references])
    isolated, = torch.autograd.grad(values[0], parameters, retain_graph=True)
    torch.testing.assert_close(isolated[1:], torch.zeros_like(isolated[1:]), rtol=0, atol=0)
    weights = parameters.new_tensor([.7, -1.1, 1.3])
    actual_gradient, = torch.autograd.grad(values@weights, parameters, retain_graph=True)
    expected_gradient, = torch.autograd.grad(reference_values@weights, parameters)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12)


def test_rejected_query_preserves_alpha_failure_without_evaluating_a_batch_field():
    parameters = torch.tensor([[2., 1., .16, 0., .71, 1.3],
                               [2.2, 1., .16, 0., .83, 1.5]], dtype=torch.float64)
    geometries, levels, physics, configs = inputs(parameters)
    physics[1]['alpha_angular_data'] = tuple(parameters.new_tensor([x]) for x in [1., 0., 0., -2.])
    calls = []
    with pytest.raises(ValueError, match='noninvertible alpha'):
        sample_boozer_surfaces(geometries, levels, physics,
            lambda points: calls.append(points), configs, nphi=9, ntheta=11, nu_order=2)
    assert not calls


def test_batch_rejects_empty_or_incomplete_input():
    with pytest.raises(ValueError, match='matching nonempty'):
        sample_boozer_surfaces([], [], [], None, [])
    with pytest.raises(ValueError, match='matching nonempty'):
        sample_boozer_surfaces([object()], [], [], None, [])
