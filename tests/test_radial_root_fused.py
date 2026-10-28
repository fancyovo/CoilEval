"""Optional CUDA ray solve: forward branches, explicit VJP and old-library fallback."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from coil_eval._autograd.field import NativeFieldLibrary
from coil_eval._autograd.front import _RadialRoot, PsiGeometry


def solve(polynomial, level, library=None, *, iterations=20, tolerance=1e-12):
    return _RadialRoot.apply(polynomial, level, .15, 1., iterations, tolerance, library)


def inputs(device='cpu', dtype=torch.float64, degree=10):
    # A non-contiguous source and broadcast level exercise shape reduction in VJP.
    generator = torch.Generator(device=device).manual_seed(91)
    p = torch.zeros((3, 4, degree+1), device=device, dtype=dtype)
    p[..., 2] = torch.linspace(20, 80, 12, device=device, dtype=dtype).reshape(3, 4)
    for power in range(3, degree+1):
        p[..., power] = torch.rand((3, 4), generator=generator, device=device, dtype=dtype)*.04/.15**power
    return p.transpose(0, 1).detach().requires_grad_(), torch.tensor(
        [.02, .07, .12], device=device, dtype=dtype, requires_grad=True)


def value_vjp(p, level, library=None):
    root = solve(p, level, library)
    adjoint = torch.linspace(-.7, 1.2, root.numel(), device=root.device, dtype=root.dtype).reshape(root.shape)
    return (root, *torch.autograd.grad((root*adjoint).sum(), (p, level)))


def test_cpu_keeps_existing_math_even_when_library_advertises_cuda_capability():
    def forbidden(*args):
        raise AssertionError('CPU must not call a CUDA helper')
    p, level = inputs()
    actual = value_vjp(p, level, SimpleNamespace(has_radial_root=True, radial_root=forbidden))
    expected = value_vjp(p, level)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    direction = torch.linspace(-.1, .2, p.numel(), dtype=p.dtype).reshape(p.shape)
    delta = 1e-5
    weight = torch.linspace(-.7, 1.2, actual[0].numel(), dtype=p.dtype).reshape(actual[0].shape)
    fd = ((solve(p.detach()+delta*direction, level.detach())-
           solve(p.detach()-delta*direction, level.detach()))*weight).sum()/(2*delta)
    torch.testing.assert_close((actual[1]*direction).sum(), fd, rtol=2e-7, atol=2e-9)


def test_older_library_loads_without_new_export(monkeypatch):
    exported = {f'fc_{name}_{suffix}': SimpleNamespace() for suffix in ('f32', 'f64')
                for name in ('field', 'potential', 'field_vjp', 'potential_vjp')}
    exported['fc_error_string'] = SimpleNamespace()
    monkeypatch.setattr('ctypes.CDLL', lambda path: SimpleNamespace(**exported))
    library = NativeFieldLibrary('not-loaded-old-library.so')
    assert not library.has_radial_root
    p, level = inputs()
    for actual, expected in zip(value_vjp(p, level, library), value_vjp(p, level)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_geometry_retains_explicit_library_and_original_six_argument_apply():
    library = SimpleNamespace(has_radial_root=False)
    axis = torch.zeros((8, 4), dtype=torch.float64)
    geometry = PsiGeometry(axis, torch.zeros(2, dtype=torch.float64), degree=2, mtor=0,
                           root_library=library)
    assert geometry.root_library is library
    p, level = inputs()
    original = _RadialRoot.apply(p, level, .15, 1., 20, 1e-12)
    torch.testing.assert_close(original, solve(p, level), rtol=0, atol=0)
    # Autograd still accepts the original six-input custom Function call.
    torch.autograd.grad(original.sum(), (p, level))


@pytest.fixture
def cuda_library():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    explicit = os.environ.get('COIL_EVAL_TEST_AUTOGRAD_LIBRARY')
    native = os.environ.get('COIL_EVAL_TEST_LIBRARY')
    if not explicit and not native:
        pytest.skip('set COIL_EVAL_TEST_AUTOGRAD_LIBRARY or COIL_EVAL_TEST_LIBRARY')
    path = explicit or str(Path(native).with_name('libcoil_eval_autograd.so'))
    library = NativeFieldLibrary(path)
    assert library.has_radial_root, 'test library does not contain the new radial-root kernel'
    return library


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('degree', [2, 10, 16, 32, 34])
def test_cuda_values_and_vjp_match_torch(cuda_library, dtype, degree):
    p, level = inputs('cuda', dtype, degree)
    expected = value_vjp(p, level)
    actual = value_vjp(p, level, cuda_library)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-6 if dtype == torch.float32 else 2e-11,
                                   atol=2e-8 if dtype == torch.float32 else 2e-12)


@pytest.mark.gpu
def test_cuda_scalar_level_empty_input_and_nondefault_stream(cuda_library):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p, _ = inputs('cuda')
        level = p.new_tensor(.08, requires_grad=True)
        actual = value_vjp(p, level, cuda_library)
        expected = value_vjp(p, level)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-12)
        empty = p.new_empty((0, 11), requires_grad=True)
        root = solve(empty, level, cuda_library)
        dp, dl = torch.autograd.grad(root.sum(), (empty, level))
        assert dp.shape == empty.shape and float(dl) == 0


@pytest.mark.gpu
def test_cuda_nonlinear_known_roots_and_independent_directional_difference(cuda_library):
    radius = torch.linspace(.02, .1, 7, dtype=torch.float64, device='cuda')
    p = torch.zeros((7, 5), dtype=radius.dtype, device=radius.device)
    p[:, 2], p[:, 4] = 40., 2000.
    p.requires_grad_()
    level = (40*radius.square()+2000*radius.pow(4)).requires_grad_()
    root = solve(p, level, cuda_library)
    dp, dl = torch.autograd.grad(root.sum(), (p, level))
    derivative = 80*radius+8000*radius.pow(3)
    torch.testing.assert_close(root, radius, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(dl, 1/derivative, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(dp[:, 4], -radius.pow(4)/derivative, rtol=1e-10, atol=1e-12)
    direction = torch.linspace(-.1, .2, p.numel(), device=p.device, dtype=p.dtype).reshape(p.shape)
    delta = 1e-5
    fd = (solve(p.detach()+delta*direction, level.detach(), cuda_library).sum()-
          solve(p.detach()-delta*direction, level.detach(), cuda_library).sum())/(2*delta)
    torch.testing.assert_close((dp*direction).sum(), fd, rtol=2e-6, atol=2e-9)


@pytest.mark.gpu
@pytest.mark.parametrize('case', ['quadratic_zero', 'quadratic_nan', 'unavailable',
                                 'negative_level', 'nan_level', 'singular', 'iteration_limit'])
def test_cuda_preserves_failure_categories(cuda_library, case):
    p = torch.zeros((2, 11), dtype=torch.float64, device='cuda')
    p[:, 2] = 40
    level = p.new_tensor(.1)
    iterations = 20
    if case == 'quadratic_zero': p[0, 2] = 0
    elif case == 'quadratic_nan': p[0, 2] = float('nan')
    elif case == 'unavailable': level.fill_(10)
    elif case == 'negative_level': level.fill_(-.1)
    elif case == 'nan_level': level.fill_(float('nan'))
    elif case == 'singular': p[:, 2] = 1e-16; level.fill_(1e-20)
    elif case == 'iteration_limit': p[:, 4] = 1e5; iterations = 0
    errors = []
    for library in (None, cuda_library):
        with pytest.raises(ValueError) as error:
            solve(p, level, library, iterations=iterations)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


@pytest.mark.gpu
def test_cuda_explicit_old_library_and_tensor_scale_fallback(cuda_library):
    p, level = inputs('cuda')
    old = SimpleNamespace(has_radial_root=False)
    for actual, expected in zip(value_vjp(p, level, old), value_vjp(p, level)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = _RadialRoot.apply(p, level, p.new_tensor(.15), 1., 20, 1e-12, cuda_library)
    torch.testing.assert_close(actual, solve(p, level), rtol=0, atol=0)
