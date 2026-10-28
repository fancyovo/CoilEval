"""CPU geometry adapter preserves exact active-branch analytical derivatives."""
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from coil_eval._cpu.engineering import engineering_metrics, metric_gradient
from coil_eval._cpu.native import CpuLibrary


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("COIL_EVAL_CPU_LIBRARY")
    try:
        return CpuLibrary(path)
    except RuntimeError:
        if path:
            raise
        pytest.skip("CPU native library not installed")


def simple_tokens():
    return np.array([
        [1.3, .08, .25, .01, .02, .18, .25, .04, .003, -.01,
         .1, .04, .3, .01, -.002, 1e5],
        [2., .07, .2, .009, -.03, .35, .24, .01, -.01, .003,
         .32, .1, .22, .005, .015, 8e4],
    ])


@pytest.mark.parametrize("nfp", [1, 4])
def test_cpu_geometry_analytic_directions(library, nfp):
    tokens = simple_tokens()
    center = metric_gradient(tokens, library, nfp)
    jac = center["derivatives"].transpose(0, 2, 1, 3).reshape(6, 2, 15)
    names = list(center["values"])[:6]
    rng = np.random.default_rng(918)
    for _ in range(3):
        direction = rng.normal(size=(2, 15))
        direction /= np.linalg.norm(direction)
        full = np.column_stack((direction, np.zeros(2)))
        h = 1e-7
        plus = metric_gradient(tokens + h*full, library, nfp)
        minus = metric_gradient(tokens - h*full, library, nfp)
        expected = [(plus["values"][name] - minus["values"][name]) / (2*h) for name in names]
        actual = np.einsum("mci,ci->m", jac, direction)
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_tensor_geometry_gradient_matches_native(library, dtype):
    tokens = torch.tensor(simple_tokens(), dtype=dtype, requires_grad=True)
    out = engineering_metrics(tokens, library, 1)
    ref = metric_gradient(tokens.detach().double().numpy(), library, 1)
    torch.testing.assert_close(out, tokens.new_tensor([ref["values"][key] for key in
        ("min_intercoil_distance", "curvature_max", "length_mean")]))
    weights = tokens.new_tensor([.8, .3, -.5])
    actual, = torch.autograd.grad(out@weights, (tokens,))
    blocks = ref["derivatives"][[3, 2, 0]].transpose(0, 2, 1, 3).reshape(3, 2, 15)
    expected = np.einsum("m,mci->ci", weights.double().numpy(), blocks)
    torch.testing.assert_close(actual[:, :-1], tokens.new_tensor(expected), rtol=2e-6, atol=2e-7)
    assert torch.count_nonzero(actual[:, -1]) == 0


def test_cpu_duplicate_coil_gap_derivative(library):
    tokens = np.repeat(simple_tokens()[:1], 2, axis=0)
    result = metric_gradient(tokens, library, 1)
    assert result["values"]["min_intercoil_distance"] == 0
    np.testing.assert_array_equal(result["derivatives"][3], 0)
