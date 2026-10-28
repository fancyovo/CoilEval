"""Native CPU derivatives against an independent differentiable Torch oracle."""
import os

import pytest

torch = pytest.importorskip('torch')

from coil_eval._cpu.autograd import CpuAutogradLibrary, CpuSegmentField
from coil_eval._autograd.field import coil_segments


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("COIL_EVAL_CPU_LIBRARY")
    try:
        return CpuAutogradLibrary(path, threads=2)
    except RuntimeError:
        if path:
            raise
        pytest.skip("CPU native library not installed")


def oracle(p, w, q, potential=False):
    r = q[:, None, :] - p[None, :, :]
    r2 = r.square().sum(-1).clamp_min(1e-24)
    inv = r2.rsqrt()
    if potential:
        return (w[None] * inv[..., None]).sum(1) * 1e-7
    cross = torch.linalg.cross(w[None].expand_as(r), r)
    b = (cross * inv[..., None] ** 3).sum(1) * 1e-7
    axes = torch.eye(3, dtype=p.dtype)
    cross_gradient = torch.linalg.cross(w[:, None, :].expand(-1, 3, -1),
                                        axes[None].expand(len(w), -1, -1)).transpose(-1, -2)
    g = (cross_gradient[None] * inv[..., None, None] ** 3
         - 3 * cross[..., :, None] * r[..., None, :] * inv[..., None, None] ** 5).sum(1) * 1e-7
    return b, g


def geometry(dtype=torch.float64):
    generator = torch.Generator().manual_seed(482)
    p = torch.randn(9, 3, dtype=dtype, generator=generator) * .1
    w = torch.randn(9, 3, dtype=dtype, generator=generator) * 1e5
    q = torch.randn(5, 3, dtype=dtype, generator=generator) * .2 + 1.3
    return tuple(t.requires_grad_() for t in (p, w, q))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("component", ["field", "gradient", "both", "potential"])
def test_field_and_moving_point_reverse(library, dtype, component):
    p, w, q = geometry(dtype)
    field = CpuSegmentField(p, w, library)
    potential = component == "potential"
    actual = field.potential(q) if potential else field(q, gradient=True)
    expected = oracle(p, w, q, potential)
    if potential:
        actual, expected = (actual,), (expected,)
    elif component == "field":
        actual, expected = actual[:1], expected[:1]
    elif component == "gradient":
        actual, expected = actual[1:], expected[1:]
    tolerance = dict(rtol=3e-5, atol=3e-8) if dtype == torch.float32 else dict(rtol=2e-11, atol=2e-14)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, **tolerance)
    weights = [torch.linspace(.2, 1.1, a.numel(), dtype=dtype).reshape(a.shape) for a in actual]
    a_grad = torch.autograd.grad(sum((a*v).sum() for a, v in zip(actual, weights)), (p, w, q))
    e_grad = torch.autograd.grad(sum((a*v).sum() for a, v in zip(expected, weights)), (p, w, q))
    for a, e in zip(a_grad, e_grad):
        torch.testing.assert_close(a, e, **tolerance)


def torch_trace(p, w, starts, phi0, phi1, steps, samples):
    def rhs(y, phi):
        c, s = phi.cos(), phi.sin()
        R, Z = y.unbind(-1)
        q = torch.stack((R*c, R*s, Z), -1).to(p.dtype)
        b = oracle(p, w, q)[0].to(y.dtype)
        br, bp = b[:, 0]*c+b[:, 1]*s, -b[:, 0]*s+b[:, 1]*c
        floor = torch.where(bp < 0, -1., 1.) * 1e-14
        denominator = torch.where(bp.abs() > 1e-14, bp, floor)
        return torch.stack((R*br/denominator, R*b[:, 2]/denominator), -1)

    h = (phi1 - phi0) / steps
    y, history = starts, [starts]
    for step in range(steps):
        phi = phi0 + step*h
        k1 = rhs(y, phi)
        k2 = rhs(y + h[:, None]/2*k1, phi + h/2)
        k3 = rhs(y + h[:, None]/2*k2, phi + h/2)
        k4 = rhs(y + h[:, None]*k3, phi + h)
        y = y + h[:, None]*(k1+2*k2+2*k3+k4)/6
        history.append(y)
    return torch.stack(history) if samples else y


@pytest.mark.parametrize("field_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("samples", [False, True])
def test_rk4_reverse_matches_torch_with_moving_states(library, field_dtype, state_dtype, samples):
    p, w, _ = geometry(field_dtype)
    # A strong toroidal component keeps the short audit trajectories regular.
    with torch.no_grad():
        w[:, 2] += 8e5
    starts = torch.tensor([[1.2, .2], [1.4, -.1]], dtype=state_dtype, requires_grad=True)
    phi0 = torch.tensor([.03, .04], dtype=state_dtype)
    phi1 = torch.tensor([.16, .18], dtype=state_dtype)
    actual = CpuSegmentField(p, w, library).trace(starts, phi0, phi1, 4, return_samples=samples)
    expected = torch_trace(p, w, starts, phi0, phi1, 4, samples)
    tolerance = dict(rtol=8e-5, atol=3e-7) if torch.float32 in (field_dtype, state_dtype) else dict(rtol=2e-10, atol=3e-13)
    torch.testing.assert_close(actual, expected, **tolerance)
    weight = torch.linspace(.2, 1.1, actual.numel(), dtype=state_dtype).reshape(actual.shape)
    actual_grad = torch.autograd.grad((actual*weight).sum(), (p, w, starts))
    expected_grad = torch.autograd.grad((expected*weight).sum(), (p, w, starts))
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, **tolerance)


@pytest.mark.parametrize("potential", [False, True])
def test_cpu_token_and_current_derivatives(library, potential):
    tokens = torch.tensor([[1., .04, .3, 0., .2, .03, 0., .4, .02, 1e5]],
                          dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.1, .3, .2], [.8, .1, -.3]], dtype=torch.float64, requires_grad=True)
    field = CpuSegmentField.from_tokens(tokens, library, nfp=2, count=16)
    expected_field = coil_segments(tokens, nfp=2, count=16)
    actual = field.potential(q) if potential else field(q)
    expected = oracle(*expected_field, q, potential)
    if not potential:
        expected = expected[0]
    a_grad = torch.autograd.grad(actual.square().sum(), (tokens, q))
    e_grad = torch.autograd.grad(expected.square().sum(), (tokens, q))
    for a, e in zip(a_grad, e_grad):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-14)


def test_empty_queries_and_cpu_only_operation(library, monkeypatch):
    def cuda_forbidden(*args, **kwargs):
        raise AssertionError("CPU operators attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", cuda_forbidden)
    p, w, q = geometry()
    field = CpuSegmentField(p, w, library)
    empty = torch.empty(0, 3, dtype=p.dtype, requires_grad=True)
    for result in (field(empty), field.potential(empty)):
        grads = torch.autograd.grad(result.sum(), (p, w, empty))
        assert all(torch.count_nonzero(g) == 0 for g in grads)
    field(q).sum().backward()
    assert p.grad is not None


def test_independent_fields_do_not_mix_cotangents(library):
    p, w, q = geometry()
    p2, w2 = p.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    one = CpuSegmentField(p, w, library)(q)
    two = CpuSegmentField(p2, w2, library)(q.detach())
    dp, dp2 = torch.autograd.grad(one.sum(), (p, p2), allow_unused=True)
    assert torch.isfinite(dp).all() and dp2 is None
    torch.testing.assert_close(one, two)


def test_rk4_reverse_against_finite_differences_by_parameter_block(library):
    p, w, _ = geometry()
    with torch.no_grad():
        w[:, 2] += 8e5
    y = torch.tensor([[1.2, .2], [1.4, -.1]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[.2, -.3], [.7, .4]], dtype=y.dtype)

    def objective(pp, ww, yy):
        return (CpuSegmentField(pp, ww, library).trace(yy, .03, .18, 4)*weights).sum()

    actual = torch.autograd.grad(objective(p, w, y), (p, w, y))
    generator = torch.Generator().manual_seed(776)
    args = [p.detach(), w.detach(), y.detach()]
    for block, (base, derivative) in enumerate(zip(args, actual)):
        direction = torch.randn(base.shape, dtype=base.dtype, generator=generator)
        direction /= direction.norm()
        if block == 1:
            direction *= 1e5
        plus, minus = list(args), list(args)
        h = 1e-5
        plus[block], minus[block] = base+h*direction, base-h*direction
        expected = (objective(*plus)-objective(*minus))/(2*h)
        torch.testing.assert_close((derivative*direction).sum(), expected, rtol=3e-6, atol=1e-10)


@pytest.mark.parametrize("bad_steps", [0, -1, True, 1.5])
def test_invalid_step_counts(library, bad_steps):
    p, w, _ = geometry()
    with pytest.raises(ValueError, match="positive integer"):
        CpuSegmentField(p, w, library).trace(torch.ones(2, 2), 0., .2, bad_steps)
