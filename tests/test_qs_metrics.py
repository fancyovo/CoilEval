"""Independent analytic mode, normalization and gradient checks for QS metrics."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from coil_eval.qs_metrics import boozer_qs_metrics, boozer_qs_metrics_field, boozer_spectrum_metrics


def grid(nphi=33, ntheta=35, nfp=4):
    return np.meshgrid(np.arange(nphi)*2*np.pi/(nfp*nphi),
                       np.arange(ntheta)*2*np.pi/ntheta, indexing="ij")


def field_and_derivatives(phi, theta):
    B = np.full(phi.shape, 2.1)
    bt, bp, btt, btp, bpp = (np.zeros(phi.shape) for _ in range(5))
    for amplitude, m, k, phase in ((.32, 1, -4, .2), (.14, 3, 8, -.3), (.09, 2, 0, .7)):
        angle = m*theta + k*phi + phase
        co, si = np.cos(angle), np.sin(angle)
        B += amplitude*co
        bt -= amplitude*m*si
        bp -= amplitude*k*si
        btt -= amplitude*m*m*co
        btp -= amplitude*m*k*co
        bpp -= amplitude*k*k*co
    return B, bt, bp, btt, btp, bpp


def call(B, **kwargs):
    settings = dict(nfp=4, helicity=(1, 4), iota=1.23, G=2.7, I=.13, major_radius=1.6)
    settings.update(kwargs)
    return boozer_qs_metrics(B, **settings)


def test_analytic_modes_match_direct_triple_product_and_fourier_energy():
    phi, theta = grid()
    B, bt, bp, btt, btp, bpp = field_and_derivatives(phi, theta)
    iota, G, I, R = 1.23, 2.7, .13, 1.6
    J = (G+iota*I)/B**2
    q = (bp+iota*bt)/J
    # Independently differentiate q, including derivatives of J, before taking
    # the triple product. The implementation analytically cancels those terms.
    qt = (btp+iota*btt)/J + q*2*bt/B
    qp = (bpp+iota*btp)/J + q*2*bp/B
    raw = (bt*qp-bp*qt)/J
    normed = R**2*raw/B**4
    expected_ft = np.sqrt(np.sum(abs(J)*normed**2)/np.sum(abs(J)))
    result = call(B)
    expected_fb = np.sqrt((.14**2+.09**2)/(2*2.1**2+.32**2+.14**2+.09**2))
    np.testing.assert_allclose(result["f_B"], expected_fb, rtol=2e-14)
    np.testing.assert_allclose(result["f_T_raw"], raw, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(result["f_T"], expected_ft, rtol=2e-13)
    np.testing.assert_allclose(result["f_T_normalized"], normed, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(result["f_B_squared"], expected_fb**2, rtol=2e-14)
    np.testing.assert_allclose(result["f_T_squared"], expected_ft**2, rtol=2e-13)


@pytest.mark.parametrize("helicity", [(1, 0), (1, 4), (2, 4), (0, 4)])
def test_exact_symmetry_and_full_torus_nfp_convention(helicity):
    phi, theta = grid()
    M, N = helicity
    B = 2+.3*np.cos(M*theta-N*phi)+.1*np.sin(2*(M*theta-N*phi))
    result = call(B, helicity=helicity)
    assert result["f_B"] < 3e-15
    assert result["f_T"] < 2e-12
    if helicity == (1, 4):
        assert call(B, helicity=(1, 1))["f_B"] > .1


def test_scaling_and_angle_origin_invariance():
    phi, theta = grid()
    B = field_and_derivatives(phi, theta)[0]
    baseline = call(B)
    shifted = call(field_and_derivatives(phi+.032, theta+.23)[0])
    strength = call(B*3.2, G=2.7*3.2, I=.13*3.2)
    length = call(B, G=2.7*2.3, I=.13*2.3, major_radius=1.6*2.3)
    orientation = call(B, G=-2.7, I=-.13)
    for result in (shifted, strength, length, orientation):
        for key in ("f_B", "f_T"):
            np.testing.assert_allclose(result[key], baseline[key], rtol=5e-13, atol=2e-14)
    np.testing.assert_allclose(strength["f_T_raw"], baseline["f_T_raw"]*3.2**4, atol=1e-9)
    np.testing.assert_allclose(length["f_T_raw"], baseline["f_T_raw"]/2.3**2, atol=1e-11)


def test_broadcast_surface_parameters_and_zero_residual():
    phi, theta = grid()
    B = field_and_derivatives(phi, theta)[0]
    batch = np.stack((B, B*2))
    result = call(batch, G=np.array([2.7, 5.4]), I=np.array([.13, .26]),
                  iota=np.array([1.23, 1.23]))
    assert result["f_B"].shape == result["f_T"].shape == (2,)
    for k in ("f_B", "f_T"):
        np.testing.assert_allclose(result[k], np.repeat(call(B)[k], 2), rtol=2e-14)
    uniform = call(np.ones((5, 7))*2)
    assert uniform["f_B"] < 1e-15
    assert uniform["f_T"] == 0


@pytest.mark.parametrize("shape", [(4, 7), (7, 8), (3, 7), (7,)])
def test_ambiguous_nyquist_and_inadequate_grids_rejected(shape):
    with pytest.raises(ValueError, match="angular"):
        call(np.ones(shape))


@pytest.mark.parametrize("override", [dict(nfp=0), dict(nfp=True), dict(nfp=2.5),
    dict(helicity=(0, 0)), dict(helicity=(1,)), dict(helicity=(1, 1.5)),
    dict(major_radius=0), dict(major_radius=np.inf), dict(G=0, I=0),
    dict(iota=np.nan), dict(G=[1, 2])])
def test_invalid_parameters_rejected(override):
    with pytest.raises(ValueError):
        call(np.ones((7, 9)), **override)


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, 1j])
def test_invalid_field_rejected(bad):
    B = np.ones((7, 9), dtype=complex if np.iscomplexobj(bad) else float)
    B[2, 3] = bad
    with pytest.raises(ValueError):
        call(B)


def test_tail_reports_resolved_high_modes_without_rejecting_or_filtering():
    phi, theta = grid(15, 17)
    B = 2+.2*np.cos(7*theta-20*phi)
    result = call(B)
    expected = .2**2/(2*2**2+.2**2)
    np.testing.assert_allclose(result["spectral_tail_fraction"], expected, rtol=2e-14)
    np.testing.assert_allclose(result["f_B_squared"], expected, rtol=2e-14)
    assert result["f_T"] < 2e-11


def test_import_does_not_load_torch():
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-B", "-c",
        f"import sys; sys.path.insert(0, {source!r}); import coil_eval.qs_metrics; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_torch_forward_and_parameter_gradients_match_independent_differences():
    torch = pytest.importorskip("torch")
    phi, theta = grid(13, 15)
    phi, theta = torch.tensor(phi), torch.tensor(theta)
    parameters = torch.tensor([.32, .14, .09, 1.23, 2.7, .13, 1.6], dtype=torch.float64, requires_grad=True)

    def metrics(p):
        B = 2.1+p[0]*torch.cos(theta-4*phi+.2)+p[1]*torch.cos(3*theta+8*phi-.3)+p[2]*torch.cos(2*theta+.7)
        return call(B, iota=p[3], G=p[4], I=p[5], major_radius=p[6])

    result = metrics(parameters)
    expected = call(field_and_derivatives(phi.numpy(), theta.numpy())[0])
    for key in ("f_B", "f_T", "f_T_raw", "f_T_normalized", "spectral_tail_fraction"):
        np.testing.assert_allclose(result[key].detach().numpy(), expected[key], atol=2e-10, rtol=2e-12)
    for key in ("f_B", "f_T"):
        gradient, = torch.autograd.grad(result[key], parameters, retain_graph=True)
        numerical = []
        for k in range(len(parameters)):
            step = torch.zeros_like(parameters); step[k] = 2e-6
            numerical.append(float((metrics(parameters+step)[key]-metrics(parameters-step)[key])/(4e-6)))
        np.testing.assert_allclose(gradient.numpy(), numerical, rtol=2e-7, atol=3e-9)


def test_torch_independent_batches_and_zero_subgradient():
    torch = pytest.importorskip("torch")
    phi, theta = grid(9, 11)
    B = torch.tensor(np.stack((field_and_derivatives(phi, theta)[0], np.ones(phi.shape)*2)), requires_grad=True)
    result = call(B)
    first, = torch.autograd.grad(result["f_B"][0]+result["f_T"][0], B, retain_graph=True)
    assert torch.count_nonzero(first[1]) == 0
    zero, = torch.autograd.grad(result["f_B"][1]+result["f_T"][1], B)
    assert torch.isfinite(zero).all()
    assert torch.count_nonzero(zero[0]) == 0
    assert torch.count_nonzero(zero[1]) == 0


def field_jets(phi, theta, *, G=2.7, iota=1.23):
    """Consistent local field/tangent jets; no global embedding is claimed."""
    b, bt, bp, *_ = field_and_derivatives(phi, theta)
    field = np.zeros((*b.shape, 3)); field[..., 0] = b
    xt = np.zeros_like(field); xt[..., 1] = 1
    xp = -iota*xt; xp[..., 0] = G/b
    jac = np.zeros((*b.shape, 3, 3))
    jac[..., 0, 0] = b/G*(bp+iota*bt)
    jac[..., 0, 1] = bt
    return field, jac, xt, xp


def test_field_jet_kernel_matches_analytic_triple_and_independent_reference():
    from coil_eval.simsopt.qs_reference import analyze_qs_arrays
    phi, theta = grid(65, 67)
    jets = field_jets(phi, theta)
    settings = dict(nfp=4, helicity=(1, 4), iota=1.23, G=2.7, major_radius=1.6)
    actual = boozer_qs_metrics_field(*jets, **settings)
    expected = call(field_and_derivatives(phi, theta)[0], I=0)
    reference = analyze_qs_arrays(xyz=np.zeros_like(jets[0]),
        B=jets[0], dBdx=jets[1], gtheta=jets[2]*2*np.pi,
        gphi=jets[3]*2*np.pi, G_turn=2*np.pi*settings['G'],
        **{k: v for k, v in settings.items() if k != 'G'})
    for key in ('f_B', 'f_T', 'f_T_raw', 'f_T_normalized'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-9, atol=3e-10)
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-10, atol=1e-10)


def test_field_jet_physical_q_is_not_silently_replaced_by_coordinate_q():
    phi, theta = grid(33, 35)
    field, jac, xt, xp = field_jets(phi, theta)
    settings = dict(nfp=4, helicity=(1, 4), iota=1.23, G=2.7, major_radius=1.6)
    base = boozer_qs_metrics_field(field, jac, xt, xp, **settings)
    changed = jac.copy(); changed[..., 0, 0] += .3*np.sin(theta+4*phi)
    perturbed = boozer_qs_metrics_field(field, changed, xt, xp, **settings)
    assert perturbed['f_B'] == base['f_B']
    assert abs(perturbed['f_T']/base['f_T']-1) > .01
    np.testing.assert_allclose(perturbed['parallel_derivative']-base['parallel_derivative'],
        field[..., 0]*.3*np.sin(theta+4*phi), atol=1e-15)


def test_field_jet_batch_scaling_and_spatial_rotation():
    phi, theta = grid(33, 35)
    field, jac, xt, xp = field_jets(phi, theta)
    settings = dict(nfp=4, helicity=(1, 4), iota=1.23, G=2.7, major_radius=1.6)
    base = boozer_qs_metrics_field(field, jac, xt, xp, **settings)
    # A proper Cartesian rotation checks every component of the Jacobian rule.
    rotation = np.array([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
    transformed = boozer_qs_metrics_field(field@rotation.T,
        np.einsum('ij,...jk,lk->...il', rotation, jac, rotation),
        xt@rotation.T, xp@rotation.T, **settings)
    batch = boozer_qs_metrics_field(np.stack([field, 3*field]),
        np.stack([jac, 3*jac/2]), np.stack([xt, 2*xt]), np.stack([xp, 2*xp]),
        **dict(settings, G=np.array([2.7, 2.7*3*2]), major_radius=np.array([1.6, 3.2])))
    for name in ('f_B', 'f_T'):
        np.testing.assert_allclose(transformed[name], base[name], rtol=5e-13)
        np.testing.assert_allclose(batch[name], np.repeat(base[name], 2), rtol=5e-13)


@pytest.mark.parametrize('bad_index', [0, 1, 2, 3])
def test_field_jet_nonfinite_inputs_rejected(bad_index):
    jets = list(field_jets(*grid(9, 11)))
    jets[bad_index] = jets[bad_index]*np.nan
    with pytest.raises(ValueError, match='finite'):
        boozer_qs_metrics_field(*jets, nfp=4, helicity=(1, 4),
            iota=1.23, G=2.7, major_radius=1.6)


def test_field_jet_torch_gradient_includes_field_derivatives_and_tangents():
    torch = pytest.importorskip('torch')
    jets = [torch.tensor(v) for v in field_jets(*grid(17, 19))]
    parameters = torch.tensor([1., 1., 1., 1., 2.7, .13, 1.23, 1.6],
        dtype=torch.float64, requires_grad=True)

    def metric(p):
        # Independent perturbations test all four live array inputs and scalar
        # normalization paths, even away from an exactly consistent Boozer jet.
        return boozer_qs_metrics_field(*(p[i]*v for i, v in enumerate(jets)),
            nfp=4, helicity=(1, 4), G=p[4], I=p[5], iota=p[6], major_radius=p[7])

    actual = metric(parameters)
    gradient, = torch.autograd.grad(actual['f_T'], parameters)
    differences = []
    for index in range(len(parameters)):
        step = torch.zeros_like(parameters); step[index] = 1e-6
        differences.append(float((metric(parameters+step)['f_T']-
                                  metric(parameters-step)['f_T'])/2e-6))
    np.testing.assert_allclose(gradient.numpy(), differences, atol=2e-9, rtol=3e-8)
    assert torch.count_nonzero(gradient) == len(parameters)


def test_spectrum_only_matches_full_metrics_without_inverse_fft(monkeypatch):
    B=field_and_derivatives(*grid())[0]
    expected=call(B)
    def forbidden(*args,**kwargs):
        raise AssertionError('spectrum-only f_B must not compute angular derivatives')
    monkeypatch.setattr(np.fft,'ifft2',forbidden)
    actual=boozer_spectrum_metrics(B,nfp=4,helicity=(1,4))
    assert 'f_T' not in actual
    for name in ('f_B','f_B_squared','spectrum','spectral_tail_fraction'):
        np.testing.assert_array_equal(actual[name],expected[name])


def test_spectrum_only_torch_batch_gradient_and_uniform_zero():
    torch=pytest.importorskip('torch')
    B=torch.tensor(np.stack([field_and_derivatives(*grid(13,15))[0],np.full((13,15),2.)]),
                   requires_grad=True)
    actual=boozer_spectrum_metrics(B,nfp=4,helicity=(1,4))
    expected=call(B)
    g,=torch.autograd.grad(actual['f_B'].sum(),B,retain_graph=True)
    ge,=torch.autograd.grad(expected['f_B'].sum(),B)
    torch.testing.assert_close(g,ge,rtol=0,atol=0)
    assert torch.count_nonzero(g[1])==0


@pytest.mark.parametrize('B',[np.ones((6,7)),np.ones(7),np.zeros((7,9)),
    np.full((7,9),np.nan),np.ones((7,9),dtype=complex)])
def test_spectrum_only_validates_grids_and_positive_real_values(B):
    with pytest.raises(ValueError):boozer_spectrum_metrics(B,nfp=4,helicity=(1,4))
