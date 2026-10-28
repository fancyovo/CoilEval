"""Shared features retain the scalar-basis values and coordinate derivatives."""
import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd.front import (
    PsiGeometry, _monomial_features, _monomial_layout, _ray_denominators,
    _trig, monomials,
)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def scalar_features(X, Y, degree, scale):
    modes = monomials(degree)
    px = torch.stack([X**k for k in range(degree+1)], -1)
    py = torch.stack([Y**k for k in range(degree+1)], -1)
    return (
        torch.stack([px[..., i]*py[..., j] for i, j in modes], -1),
        torch.stack([i*px[..., max(i-1, 0)]*py[..., j]/scale for i, j in modes], -1),
        torch.stack([j*px[..., i]*py[..., max(j-1, 0)]/scale for i, j in modes], -1),
    )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('degree', [2, 6, 10])
def test_basis_values_vjp_and_independent_scale_match_scalar_basis(device, dtype, degree):
    generator = torch.Generator(device=device).manual_seed(680)
    data = torch.randn((2, 3, 7), generator=generator, device=device, dtype=dtype)*.4
    data[:, :, 0] = 0
    X, Y = data.clone().requires_grad_(), data.flip(-1).clone().requires_grad_()
    scale = torch.tensor([.15, .24], device=device, dtype=dtype).reshape(2, 1, 1).requires_grad_()
    expected = scalar_features(X, Y, degree, scale)
    actual = _monomial_features(X, Y, degree, scale)
    weights = [torch.randn(v.shape, generator=generator, device=device, dtype=dtype) for v in actual]
    # Replacing independent scalar arithmetic by vector arithmetic preserves
    # feature values exactly, while gradient accumulation order can differ.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    old = torch.autograd.grad(sum((w*v).sum() for w, v in zip(weights, expected)), (X, Y, scale))
    new = torch.autograd.grad(sum((w*v).sum() for w, v in zip(weights, actual)), (X, Y, scale))
    # Grouped scatter/reduction changes FP32 cancellation at roundoff scale;
    # use an absolute floor as well as relative tolerance near zero entries.
    tolerance = 3e-5 if dtype == torch.float32 else 2e-13
    for a, b in zip(new, old):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    independent = _monomial_features(X, Y, degree, scale)[1][0].square().sum()
    dx, dy, ds = torch.autograd.grad(independent, (X, Y, scale))
    assert not torch.count_nonzero(dx[1]) and not torch.count_nonzero(dy[1])
    assert not torch.count_nonzero(ds[1])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_physical_ray_values_and_vjps_match_scalar_basis(device, dtype):
    generator = torch.Generator(device=device).manual_seed(419)
    axis = torch.zeros((8, 4), device=device, dtype=dtype)
    axis[:, 0] = 1
    coeff = (torch.randn(1574, generator=generator, device=device, dtype=dtype)*.0001).requires_grad_()
    phi = torch.linspace(.01, .9, 6, device=device, dtype=dtype)[:, None].requires_grad_()
    theta = torch.linspace(-1., 4., 9, device=device, dtype=dtype)[None, :].requires_grad_()
    geometry = PsiGeometry(axis, coeff)
    pp, tt = torch.broadcast_tensors(phi, theta)
    amplitude = _trig(pp, 12, 4)@geometry._matrix.T
    modes = monomials(10)
    angular = torch.stack([tt.cos()**a*tt.sin()**b/.15**(a+b) for a, b in modes], -1)
    terms = amplitude*angular
    expected = torch.stack([terms[..., [j for j, (a, b) in enumerate(modes) if a+b == k]].sum(-1)
                            if k >= 2 else torch.zeros_like(pp) for k in range(11)], -1)
    actual = geometry.rays(phi, theta)
    # CUDA scalar division and vector division can select reciprocal-multiply
    # versus divide instructions. Bound their propagated rounding; the CPU
    # scalar and vector paths are also recorded bitwise by the frozen runner.
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=128*eps,
                               atol=128*eps*max(1., float(expected.abs().max())))
    # Radius powers make all degrees contribute on their natural physical scale.
    weight = torch.randn(actual.shape, generator=generator, device=device, dtype=dtype)
    weight = weight*(.13**torch.arange(11, device=device, dtype=dtype))
    old = torch.autograd.grad((weight*expected).sum(), (coeff, phi, theta), retain_graph=True)
    new = torch.autograd.grad((weight*actual).sum(), (coeff, phi, theta))
    tolerance = 8e-6 if dtype == torch.float32 else 3e-13
    for a, b in zip(new, old):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)


def test_value_only_cache_warmup_does_not_capture_inference_tensors():
    _monomial_layout.cache_clear()
    _ray_denominators.cache_clear()
    with torch.inference_mode():
        _monomial_features(torch.ones(3), torch.ones(3), 3)
        _ray_denominators(3, .15, torch.device('cpu'), torch.float32)
    X = torch.ones(3, requires_grad=True)
    features = _monomial_features(X, X, 3)
    torch.autograd.grad(sum(v.sum() for v in features), X)
    divisor = _ray_denominators(3, .15, torch.device('cpu'), torch.float32)
    torch.autograd.grad((X[:, None]/divisor).sum(), X)
