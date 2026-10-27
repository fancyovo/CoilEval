"""Analytic vacuum potential modes, including transport-resonant corrections."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from coil_eval._autograd.boozer import fit_nu, fourier_values
from coil_eval._autograd.boozer_potential import fit_nu_potential


def fixture():
    phi, theta = torch.meshgrid(torch.arange(33, dtype=torch.float64)*2*np.pi/(4*33),
        torch.arange(35, dtype=torch.float64)*2*np.pi/35, indexing='ij')
    phase1, phase2 = theta-4*phi+.2, 2*theta+8*phi-.3
    nu = .12*phase1.sin()+.04*phase2.cos()
    nt = .12*phase1.cos()-.08*phase2.sin()
    np_ = -.48*phase1.cos()-.32*phase2.sin()
    return phi, theta, nu, nt, np_


def test_known_potential_and_resonant_mode_are_reconstructed():
    phi, theta, nu, nt, np_ = fixture()
    data, diagnostics = fit_nu_potential(2.7*nt, 2.7*(1+np_), 2.7, 4, 4)
    actual, actual_t, actual_p = fourier_values(data, theta, phi)
    torch.testing.assert_close(actual, nu, atol=3e-15, rtol=3e-14)
    torch.testing.assert_close(actual_t, nt, atol=3e-15, rtol=3e-14)
    torch.testing.assert_close(actual_p, np_, atol=1e-14, rtol=3e-14)
    assert diagnostics['nu_potential_derivative_residual_rms'] < 2e-15
    assert diagnostics['nu_potential_curl_rms'] < 5e-14
    # iota=4 annihilates the (m,k)=(1,-4) component in transport. It is still
    # fixed by the two covariant components, independent of rotational transform.
    transport, report = fit_nu(np_+4*nt, torch.tensor(4.), 4, 4)
    transport_nu = fourier_values(transport, theta, phi)[0]
    assert torch.linalg.vector_norm(actual-transport_nu) > 1
    assert report['nu_transport_residual_rms'] < 1e-14


@pytest.mark.parametrize('scale', [.02, 2.3, -3.1])
def test_signed_field_strength_and_length_scaling_invariance(scale):
    phi, theta, nu, nt, np_ = fixture()
    data, _ = fit_nu_potential(2.7*scale*nt, 2.7*scale*(1+np_), 2.7*scale, 4, 4)
    torch.testing.assert_close(fourier_values(data, theta, phi)[0], nu,
                              atol=3e-15, rtol=3e-14)


def test_nonclosed_and_nonzero_poloidal_mean_are_reported():
    phi, theta, _, nt, np_ = fixture()
    _, diagnostic = fit_nu_potential(2.7*(nt+.02+.03*torch.sin(4*phi)),
        2.7*(1+np_+.01), 2.7, 4, 4)
    assert float(diagnostic['nu_potential_I_over_G']) == pytest.approx(.02)
    assert float(diagnostic['nu_potential_toroidal_mean_error']) == pytest.approx(.01)
    assert diagnostic['nu_potential_curl_rms'] > .08
    assert diagnostic['nu_potential_derivative_residual_rms'] > .02


def test_constant_gauge_is_zero_and_omitted_modes_remain_diagnostic():
    phi, theta, _, nt, np_ = fixture()
    data, diagnostic = fit_nu_potential(2.7*(nt+.05*6*torch.cos(6*theta)),
        2.7*(1+np_), 2.7, 4, 4)
    value = fourier_values(data, theta, phi)[0]
    assert abs(float(value.mean())) < 1e-16
    assert float(diagnostic['nu_potential_omitted_derivative_l2']) == pytest.approx(.3/np.sqrt(2))


def test_live_parameter_gradient_matches_centered_differences():
    phi, theta, *_ = fixture()
    parameters = torch.tensor([.12, .04, 2.7, .2], dtype=torch.float64, requires_grad=True)
    def value(p):
        phase1, phase2 = theta-4*phi+p[3], 2*theta+8*phi-.3
        nt = p[0]*phase1.cos()-2*p[1]*phase2.sin()
        np_ = -4*p[0]*phase1.cos()-8*p[1]*phase2.sin()
        data, _ = fit_nu_potential(2.7*nt, 2.7*(1+np_), p[2], 4, 4)
        v = fourier_values(data, theta, phi)[0]
        return v[3, 4]+.2*v.square().mean()
    gradient, = torch.autograd.grad(value(parameters), parameters)
    numerical = []
    for i in range(len(parameters)):
        step = torch.zeros_like(parameters); step[i] = 1e-6
        numerical.append(float((value(parameters+step)-value(parameters-step))/2e-6))
    np.testing.assert_allclose(gradient.numpy(), numerical, rtol=3e-8, atol=2e-10)
    assert torch.count_nonzero(gradient) == len(parameters)


@pytest.mark.parametrize('change', [dict(nfp=True), dict(nfp=0), dict(order=0),
    dict(order=17), dict(G=0.), dict(G=float('nan')), dict(G=torch.ones(2))])
def test_invalid_options(change):
    _, _, _, nt, np_ = fixture()
    settings = dict(G=2.7, nfp=4, order=4); settings.update(change)
    with pytest.raises(ValueError):
        fit_nu_potential(2.7*nt, 2.7*(1+np_), **settings)
