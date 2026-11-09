"""Manufactured coordinate checks, independent of CUDA and captured centers."""
import math

import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd.boozer import (
    alpha_angular_coefficients, fit_nu, fourier_values, invert_alpha,
    invert_nu, sample_boozer_surface, surface_geometry, ComposedAngularCorrection,
)
from coil_eval._autograd.downstream import DownstreamConfig, clebsch_modes
from coil_eval._autograd.front import PsiGeometry


@pytest.fixture(autouse=True)
def single_thread_torch():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def circle_geometry(major, ellipticity=1., *, a=.5):
    major = torch.as_tensor(major, dtype=torch.float64)
    zero = major * 0
    axis = torch.stack((major.expand(8), zero.expand(8), zero.expand(8), zero.expand(8)), -1)
    coefficients = torch.stack((zero, torch.as_tensor(ellipticity).to(major)))
    return PsiGeometry(axis, coefficients, nfp=4, a=a, degree=2, mtor=0)


def angular_grid(np=17, nt=19):
    return torch.meshgrid(torch.arange(np, dtype=torch.float64) * (2 * math.pi / (4 * np)),
                          torch.arange(nt, dtype=torch.float64) * (2 * math.pi / nt), indexing='ij')


def finite_difference(function, parameters, step=1e-6):
    derivatives = []
    for i in range(len(parameters)):
        direction = torch.zeros_like(parameters)
        direction[i] = step
        derivatives.append((function(parameters.detach() + direction) -
                            function(parameters.detach() - direction)) / (2 * step))
    return torch.stack(derivatives)


def test_implicit_alpha_gradient_includes_coefficients_and_both_angles():
    p = torch.tensor([.12, .08, .7, .4], dtype=torch.float64, requires_grad=True)

    def objective(parameters):
        c, s, theta, phi = parameters.unbind()
        data = (c.new_tensor([2.]), c.new_tensor([-4.]), c[None], s[None])
        root, info = invert_alpha(data, theta, phi)
        assert info['alpha_inversion_max_rad'] < 1e-10
        return root

    value = objective(p)
    gradient, = torch.autograd.grad(value, p)
    torch.testing.assert_close(gradient, finite_difference(objective, p), atol=2e-8, rtol=2e-7)


def test_composed_alpha_inversion_keeps_both_corrections_and_chain_derivatives():
    parameters = torch.tensor([.17, .06, .8, .3], dtype=torch.float64, requires_grad=True)

    def objective(p):
        a, b, theta, phi = p.unbind()
        base = (a.new_tensor([1.]), a.new_tensor([-4.]), a[None]*0, a[None])
        correction = (a.new_tensor([2.]), a.new_tensor([4.]), b[None], b[None]*0)
        data = ComposedAngularCorrection(base, correction)
        root, _ = invert_alpha(data, theta, phi)
        val, dt, dp = fourier_values(data, root, phi)
        torch.testing.assert_close(root+val, theta, atol=1e-11, rtol=1e-11)
        return root+.1*dt+.2*dp

    gradient, = torch.autograd.grad(objective(parameters), parameters)
    torch.testing.assert_close(gradient, finite_difference(objective, parameters), atol=3e-8, rtol=3e-7)


def test_implicit_nu_gradient_includes_iota_and_poloidal_map_motion():
    p = torch.tensor([.08, -.03, .71, .2, 1.31], dtype=torch.float64, requires_grad=True)

    def objective(parameters):
        c, s, theta, phi, iota = parameters.unbind()
        data = (c.new_tensor([2.]), c.new_tensor([-4.]), c[None], s[None])
        th, ph, info = invert_nu(data, theta, phi, iota)
        assert info['nu_inversion_max_rad'] < 1e-10
        return ph + .3 * th

    gradient, = torch.autograd.grad(objective(p), p)
    torch.testing.assert_close(gradient, finite_difference(objective, p), atol=2e-8, rtol=2e-7)


def test_nu_fft_matches_known_solution_and_resonance_is_reported():
    phi, theta = angular_grid()
    iota = torch.tensor(1.31, dtype=torch.float64)
    expected = .04 * torch.cos(2 * theta - 4 * phi) + .03 * torch.sin(theta + 8 * phi)
    target = -.04 * (2 * iota - 4) * torch.sin(2 * theta - 4 * phi)
    target = target + .03 * (iota + 8) * torch.cos(theta + 8 * phi)
    data, diagnostics = fit_nu(target, iota, 4, 3)
    value, _, _ = fourier_values(data, theta, phi)
    torch.testing.assert_close(value, expected, atol=2e-15, rtol=2e-14)
    assert diagnostics['nu_transport_residual_rms'] < 1e-15
    data, diagnostics = fit_nu(torch.cos(2 * theta - 4 * phi), iota * 0 + 2, 4, 3)
    assert diagnostics['nu_resonant_mode_count'] >= 2
    assert diagnostics['nu_resonant_spectrum_l2'] > .7
    assert diagnostics['nu_transport_residual_rms'] > .7


def test_nu_half_spectrum_preserves_values_derivatives_and_parameter_gradient():
    phi, theta = angular_grid()
    amplitude = torch.linspace(-.03, .04, 17 * 19, dtype=torch.float64).reshape(17, 19)
    amplitude = (amplitude + .07 * torch.sin(3 * theta + 4 * phi)).requires_grad_()
    iota = torch.tensor(1.31, dtype=torch.float64, requires_grad=True)
    half, _ = fit_nu(amplitude, iota, 4, 3)
    assert half[0].numel() == ((2 * 3 + 1)**2 - 1) // 2
    kp = torch.fft.fftfreq(17, d=1 / 17, dtype=torch.float64)
    kt = torch.fft.fftfreq(19, d=1 / 19, dtype=torch.float64)
    k, m = torch.meshgrid(4 * kp, kt, indexing='ij')
    mask = (kp[:, None].abs() <= 3) & (kt[None, :].abs() <= 3) & ((m != 0) | (k != 0))
    coefficient = torch.fft.fft2(amplitude, norm='forward')[mask] / (1j * (k + iota * m)[mask])
    full = (m[mask], k[mask], coefficient.real, -coefficient.imag)
    query_theta, query_phi = theta + .17, phi - .09
    actual = fourier_values(half, query_theta, query_phi)
    expected = fourier_values(full, query_theta, query_phi)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=1e-14, rtol=1e-13)
    value = sum((a.square()).mean() for a in actual)
    reference = sum((e.square()).mean() for e in expected)
    gradient = torch.autograd.grad(value, (amplitude, iota), retain_graph=True)
    reference_gradient = torch.autograd.grad(reference, (amplitude, iota))
    for a, e in zip(gradient, reference_gradient):
        torch.testing.assert_close(a, e, atol=1e-14, rtol=1e-12)


def test_direct_surface_tangents_follow_negative_geometric_angle():
    geometry = circle_geometry(2., 1.2)
    phi, theta = angular_grid(9, 11)
    level = torch.tensor(.16, dtype=torch.float64)
    xyz, xt, xp = surface_geometry(geometry, level, theta, phi)
    step = 1e-6
    tp = surface_geometry(geometry, level, theta + step, phi)[0]
    tm = surface_geometry(geometry, level, theta - step, phi)[0]
    pp = surface_geometry(geometry, level, theta, phi + step)[0]
    pm = surface_geometry(geometry, level, theta, phi - step)[0]
    torch.testing.assert_close(xt, (tp - tm) / (2 * step), atol=5e-10, rtol=5e-8)
    torch.testing.assert_close(xp, (pp - pm) / (2 * step), atol=5e-10, rtol=5e-8)
    assert xyz[0, 1, 2] < 0


def test_axisymmetric_vacuum_field_is_already_boozer():
    geometry = circle_geometry(2.)
    config = DownstreamConfig(alpha_radial_order=2, alpha_poloidal_order=2,
                              alpha_toroidal_order=1, alpha_radial_basis='flux_legendre')
    count = len(clebsch_modes(2, 2, 1, 'flux_legendre')[0])
    physics = dict(alpha_coefficients=torch.zeros(count, dtype=torch.float64),
                   iota_coefficients=torch.zeros(1, dtype=torch.float64),
                   major_radius=torch.tensor(2., dtype=torch.float64))

    def field(xyz):
        x, y, z = xyz.unbind(-1)
        return torch.stack((-1.3 * y / (x*x+y*y), 1.3 * x / (x*x+y*y), z*0), -1)

    result = sample_boozer_surface(geometry, torch.tensor(.16), physics, field, config,
                                   nphi=17, ntheta=19, nu_order=3)
    expected = 1.3 / (2 + .2 * result['theta_b'].cos())
    torch.testing.assert_close(result['B_magnitude'], expected, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(result['G'], result['G'].new_tensor(1.3), atol=2e-14, rtol=2e-14)
    assert result['diagnostics']['relative_boozer_equation_rms'] < 1e-14
    assert result['diagnostics']['relative_normal_B_rms'] < 1e-14


def test_field_derivative_sample_keeps_jacobian_gradient_and_boozer_tangents_live():
    config = DownstreamConfig(alpha_radial_order=2, alpha_poloidal_order=2,
                              alpha_toroidal_order=1, alpha_radial_basis='flux_legendre')
    count = len(clebsch_modes(2, 2, 1, 'flux_legendre')[0])

    def objective(parameters):
        major, scale = parameters.unbind()
        geometry = circle_geometry(major)
        physics = dict(alpha_coefficients=major.new_zeros(count),
                       iota_coefficients=major.new_zeros(1), major_radius=major)

        def field(xyz, gradient=False):
            x, y, z = xyz.unbind(-1)
            r2 = x*x+y*y
            B = torch.stack((-scale*y/r2, scale*x/r2, .2*x), -1)
            if not gradient:
                return B
            zero = z*0
            J = torch.stack((
                torch.stack((2*scale*x*y/r2**2, -scale/r2+2*scale*y*y/r2**2, zero), -1),
                torch.stack((scale/r2-2*scale*x*x/r2**2, -2*scale*x*y/r2**2, zero), -1),
                torch.stack((zero+.2, zero, zero), -1)), -2)
            return B, J

        sampled = sample_boozer_surface(geometry, major*0+.16, physics, field, config,
            nphi=17, ntheta=19, nu_order=3, field_derivatives=True)
        xyz, xt, xp = surface_geometry(geometry, major*0+.16,
                                       sampled['theta_b'], sampled['phi_b'])
        torch.testing.assert_close(sampled['xyz'], xyz, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(sampled['x_theta'], xt, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(sampled['x_phi'], xp, atol=1e-14, rtol=1e-14)
        return sampled['parallel_derivative'].square().mean()

    parameters = torch.tensor([2., 1.3], dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(objective(parameters), parameters)
    assert bool((gradient.abs() > 1e-9).all())
    torch.testing.assert_close(gradient, finite_difference(objective, parameters), rtol=2e-5, atol=1e-12)


def test_complete_sampler_gradient_keeps_axis_psi_level_lambda_iota_and_field_live():
    config = DownstreamConfig(alpha_radial_order=2, alpha_poloidal_order=2,
                              alpha_toroidal_order=1, alpha_radial_basis='flux_legendre')
    count = len(clebsch_modes(2, 2, 1, 'flux_legendre')[0])
    direction = torch.sin(torch.arange(count, dtype=torch.float64) + .4)
    initial = torch.tensor([2., 1.1, .16, .003, .71, 1.3], dtype=torch.float64, requires_grad=True)

    def objective(parameters):
        major, ellipticity, level, amplitude, iota, scale = parameters.unbind()
        geometry = circle_geometry(major, ellipticity)
        physics = dict(alpha_coefficients=amplitude * direction,
                       iota_coefficients=iota[None], major_radius=major)

        def field(xyz):
            x, y, z = xyz.unbind(-1)
            R = torch.sqrt(x*x+y*y)
            phi = torch.atan2(y, x)
            theta = -torch.atan2(z, R - major)
            ripple = 1 + .015 * torch.cos(theta - 4 * phi)
            br = -.01 * z
            bz = -.01 * (R - major)
            bp = scale * ripple / R
            return torch.stack((br*x/R-bp*y/R, br*y/R+bp*x/R, bz), -1)

        result = sample_boozer_surface(geometry, level, physics, field, config,
                                       nphi=17, ntheta=19, nu_order=3)
        weight = 1 + .17 * torch.sin(result['theta_b'] - 4 * result['phi_b'])
        return (weight * result['B_magnitude'].square()).mean() + .03 * result['G']

    gradient, = torch.autograd.grad(objective(initial), initial)
    expected = finite_difference(objective, initial)
    assert bool((gradient.abs() > 1e-7).all())
    torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=2e-5)


def test_folded_alpha_is_rejected_instead_of_damped():
    data = tuple(torch.tensor([x], dtype=torch.float64) for x in [1., 0., 0., -2.])
    with pytest.raises(ValueError, match='noninvertible alpha'):
        invert_alpha(data, torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64))


def test_alpha_radial_contraction_preserves_coefficient_gradients():
    config = DownstreamConfig(alpha_radial_order=4, alpha_poloidal_order=2,
                              alpha_toroidal_order=1, alpha_radial_basis='flux_legendre')
    count = len(clebsch_modes(4, 2, 1, 'flux_legendre')[0])
    coefficient = torch.linspace(-.01, .01, count, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    data = alpha_angular_coefficients(coefficient, config, rho=rho)
    value = fourier_values(data, rho*0+.3, rho*0+.2)[0]
    dc, dr = torch.autograd.grad(value, (coefficient, rho))
    assert bool(torch.isfinite(dc).all()) and dc.abs().sum() > 0
    assert torch.isfinite(dr) and dr.abs() > 0


@pytest.mark.parametrize('coordinates', ['geometric', 'straight'])
def test_target_alpha_source_recovers_values_and_derivatives(coordinates):
    from coil_eval._autograd.surface_alpha import refine_surface_alpha
    geometry = circle_geometry(2.)
    phi, theta = angular_grid()
    p = torch.tensor([.17, .04, 1.31], dtype=torch.float64, requires_grad=True)
    base_data = (p.new_tensor([1.]), p.new_tensor([-4.]), p[:1]*0, p[:1])

    def field(points):
        x, y, z = points.unbind(-1)
        R = torch.sqrt(x*x+y*y)
        ph, th = torch.atan2(y, x), -torch.atan2(z, R-2)
        radius = torch.sqrt((R-2)**2+z*z)
        base, bt = p[0]*torch.sin(th-4*ph), p[0]*torch.cos(th-4*ph)
        bp = -4*bt
        phase = 2*(th+base if coordinates == 'straight' else th)-4*ph
        dt, dp = 2*p[1]*phase.cos(), -4*p[1]*phase.cos()
        lt = bt+dt*(1+bt) if coordinates == 'straight' else bt+dt
        lp = bp*(1+dt)+dp if coordinates == 'straight' else bp+dp
        beta = (p[2]-lp)/(1+lt)
        xp = torch.stack((-y, x, z*0), -1)
        xt = torch.stack((-radius*th.sin()*ph.cos(), -radius*th.sin()*ph.sin(), -radius*th.cos()), -1)
        return xp+beta[:, None]*xt

    data, iota, diagnostic = refine_surface_alpha(geometry, p.new_tensor(.16), base_data,
        p[2]*0+1.2, field, nfp=4, order=3, nphi=17, ntheta=19, ridge=1e-13,
        coordinates=coordinates)
    value = fourier_values(data, theta, phi)[0]
    base = p[0]*torch.sin(theta-4*phi)
    expected = base+p[1]*torch.sin(2*(theta+base if coordinates == 'straight' else theta)-4*phi)
    torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(iota, p[2], atol=1e-11, rtol=1e-10)
    assert diagnostic['surface_alpha_transport_residual_rms'] < 1e-10
    solver = diagnostic['surface_alpha_solver']
    assert solver['method'] == 'dense' and solver['forward']['gate_passed']
    assert solver['backward'] == []
    actual_grad, = torch.autograd.grad(value.square().mean()+iota**2, p, retain_graph=True)
    assert len(solver['backward']) == 1 and solver['backward'][0]['gate_passed']
    expected_grad, = torch.autograd.grad(expected.square().mean()+p[2]**2, p)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-9, rtol=1e-8)
    if coordinates == 'geometric':
        no_volume, no_volume_iota, no_volume_diagnostic = refine_surface_alpha(
            geometry, p.new_tensor(.16), None, None, field, nfp=4, order=3,
            nphi=17, ntheta=19, ridge=1e-13, coordinates=coordinates)
        no_volume_value = fourier_values(no_volume, theta, phi)[0]
        torch.testing.assert_close(no_volume_value, value, atol=0., rtol=0.)
        torch.testing.assert_close(no_volume_iota, iota, atol=0., rtol=0.)
        assert 'surface_alpha_previous_iota' not in no_volume_diagnostic
        no_volume_grad, = torch.autograd.grad(no_volume_value.square().mean()+no_volume_iota**2, p)
        torch.testing.assert_close(no_volume_grad, actual_grad, atol=0., rtol=0.)


def test_ridge_least_squares_rank_two_reverse_matches_live_scaled_reference_and_fd():
    from coil_eval._autograd.surface_alpha import _RidgeLeastSquares
    generator = torch.Generator().manual_seed(27)
    A = torch.randn(17, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(17, generator=generator, dtype=torch.float64, requires_grad=True)
    ridge = 1e-8
    weight = torch.randn(5, generator=generator, dtype=torch.float64)
    scales = A.norm(dim=0)
    scaled = A/scales
    actual = _RidgeLeastSquares.apply(scaled, b, ridge)/scales
    reference = torch.linalg.solve(scaled.T@scaled+ridge*torch.eye(5, dtype=A.dtype), scaled.T@b)/scales
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-13)
    gradients = torch.autograd.grad((actual*weight).sum(), (A, b), retain_graph=True)
    expected = torch.autograd.grad((reference*weight).sum(), (A, b))
    for a, e in zip(gradients, expected):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=1e-12)
    assert torch.autograd.gradcheck(lambda x, y: _RidgeLeastSquares.apply(x, y, ridge),
                                   (A, b), atol=2e-6, rtol=2e-5)


def test_bounded_angular_fourier_values_and_all_first_derivatives_match_dense(monkeypatch):
    import coil_eval._autograd.angular_fourier as module
    monkeypatch.setattr(module, '_MAX_FEATURE_ELEMENTS', 12)
    monkeypatch.setattr(module, '_MAX_MODE_TILE', 3)
    gen = torch.Generator().manual_seed(83)
    m = torch.tensor([0., 1., 2., 3., 4.], dtype=torch.float64)
    k = torch.tensor([4., -4., 8., -12., 4.], dtype=torch.float64)
    c = torch.randn(5, generator=gen, dtype=torch.float64, requires_grad=True)
    s = torch.randn(5, generator=gen, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(3, 1, generator=gen, dtype=torch.float64, requires_grad=True)
    phi = torch.randn(1, 4, generator=gen, dtype=torch.float64, requires_grad=True)

    def operation(c, s, th, ph):
        return module.angular_fourier_values((m, k, c, s), th, ph)

    actual = operation(c, s, theta, phi)
    phase = theta[..., None]*m+phi[..., None]*k
    co, si = phase.cos(), phase.sin()
    derivative = -si*c+co*s
    expected = (co@c+si@s, derivative@m, derivative@k)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=2e-13)
    weights = [torch.randn(3, 4, generator=gen, dtype=torch.float64) for _ in range(3)]
    objective = sum((a*w).sum() for a, w in zip(actual, weights))
    reference = sum((e*w).sum() for e, w in zip(expected, weights))
    gradients = torch.autograd.grad(objective, (c, s, theta, phi), retain_graph=True)
    references = torch.autograd.grad(reference, (c, s, theta, phi))
    for a, e in zip(gradients, references):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=2e-12)
    assert torch.autograd.gradcheck(operation, (c, s, theta, phi), atol=2e-6, rtol=2e-5)


def test_angular_fourier_saves_only_compact_inputs():
    from coil_eval._autograd.angular_fourier import angular_fourier_values
    saved = []
    def pack(value):
        saved.append(value.shape)
        return value
    m = torch.arange(41, dtype=torch.float64)
    c = torch.ones(41, dtype=torch.float64, requires_grad=True)
    theta = torch.linspace(0, 1, 103, dtype=torch.float64, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        out = angular_fourier_values((m, 4*m, c, c*.2), theta, theta*.3)
    assert (103, 41) not in saved
    assert all(len(shape) <= 1 for shape in saved)
    gradient, = torch.autograd.grad(sum(v.sum() for v in out), c)
    assert torch.isfinite(gradient).all()


def test_transport_design_compact_beta_vjp_matches_dense_and_fd(monkeypatch):
    import coil_eval._autograd.angular_fourier as module
    monkeypatch.setattr(module, '_MAX_FEATURE_ELEMENTS', 12)
    monkeypatch.setattr(module, '_MAX_MODE_TILE', 3)
    gen = torch.Generator().manual_seed(89)
    beta = torch.randn(11, generator=gen, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(11, generator=gen, dtype=torch.float64)
    phi = torch.randn(11, generator=gen, dtype=torch.float64)
    m = torch.tensor([0., 1., 2., 3., 4.], dtype=torch.float64)
    k = torch.tensor([4., -4., 8., -12., 4.], dtype=torch.float64)
    actual = module.transport_fourier_design(beta, m, k, theta, phi)
    phase = theta[:, None]*m+phi[:, None]*k
    frequency = k+beta[:, None]*m
    expected = torch.cat((-frequency*phase.sin(), frequency*phase.cos(),
                          -torch.ones_like(beta[:, None])), -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weight = torch.randn(actual.shape, generator=gen, dtype=torch.float64)
    gradient, = torch.autograd.grad((actual*weight).sum(), beta, retain_graph=True)
    reference, = torch.autograd.grad((expected*weight).sum(), beta)
    torch.testing.assert_close(gradient, reference, rtol=1e-13, atol=1e-13)
    assert torch.autograd.gradcheck(lambda b: module.transport_fourier_design(b, m, k, theta, phi),
                                   (beta,), atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='fixed angular grid'):
        module.transport_fourier_design(beta, m, k, theta.requires_grad_(), phi)
