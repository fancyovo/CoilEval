"""SIMD VJP prefixes match the retained scalar loop in the same binary."""
import os
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from coil_eval._cpu.autograd import CpuAutogradLibrary


@pytest.fixture(params=[1, 3])
def library(request):
    path = os.environ.get('COIL_EVAL_CPU_LIBRARY')
    if not path:
        pytest.skip('set COIL_EVAL_CPU_LIBRARY')
    return CpuAutogradLibrary(path, threads=request.param)


def data(dtype, ns, nq):
    rng = torch.Generator().manual_seed(87)
    p = torch.randn(ns, 3, generator=rng, dtype=dtype)*.1
    w = torch.randn(ns, 3, generator=rng, dtype=dtype)*2
    q = torch.randn(nq, 3, generator=rng, dtype=dtype)*.2+1.1
    a = torch.randn(nq, 3, generator=rng, dtype=dtype)*.03
    h = torch.randn(nq, 3, 3, generator=rng, dtype=dtype)*.02
    return p, w, q, a, h


def call(library, arrays, kind, outputs=(True, True, True)):
    p, w, q, a, h = arrays
    dp, dw, dq = [torch.empty_like(v) if requested else None
                  for v, requested in zip((p, w, q), outputs)]
    if kind == 'potential':
        library.call('potential_vjp', (p, w, q, a, dp, dw, dq), len(p), len(q), q)
    else:
        library.call('field_vjp', (p, w, q, None if kind == 'gradient' else a,
            None if kind == 'field' else h, dp, dw, dq), len(p), len(q), q)
    return dp, dw, dq


def scalar_reference(library, arrays, kind):
    p, w, q, a, h = arrays
    # One outer item cannot fill a SIMD group. The reduction dimension stays
    # intact, so this checks its exact order independently of the vector kernel.
    dp, dw = torch.empty_like(p), torch.empty_like(w)
    for s in range(len(p)):
        results = call(library, (p[s:s+1], w[s:s+1], q, a, h), kind, (True, True, False))
        dp[s], dw[s] = results[0][0], results[1][0]
    dq = torch.empty_like(q)
    for i in range(len(q)):
        dq[i] = call(library, (p, w, q[i:i+1], a[i:i+1], h[i:i+1]),
                     kind, (False, False, True))[2][0]
    return dp, dw, dq


def assert_same(actual, expected):
    for a, e in zip(actual, expected):
        if a is None:
            continue
        av, ev = a.numpy(), e.numpy()
        np.testing.assert_array_equal(av, ev)
        integer = np.uint32 if av.dtype == np.float32 else np.uint64
        finite = np.isfinite(ev)
        np.testing.assert_array_equal(av[finite].view(integer), ev[finite].view(integer))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('kind', ['field', 'gradient', 'both', 'potential'])
@pytest.mark.parametrize('shape', [(1, 1), (3, 7), (8, 8), (9, 17), (17, 9), (17, 0)])
def test_vjp_lane_reduction_order_and_tails(library, dtype, kind, shape):
    arrays = data(dtype, *shape)
    assert_same(call(library, arrays, kind), scalar_reference(library, arrays, kind))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('outputs', [(True, False, False), (False, True, False),
                                    (True, True, False), (False, False, True)])
def test_partial_output_buffers(library, dtype, outputs):
    arrays = data(dtype, 17, 19)
    assert_same(call(library, arrays, 'both', outputs), scalar_reference(library, arrays, 'both'))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('kind', ['field', 'gradient', 'both', 'potential'])
def test_clamp_and_nonfinite_branches(library, dtype, kind):
    arrays = data(dtype, 9, 17)
    p, w, q, a, h = arrays
    p[0].zero_()
    q[0].zero_(); q[1].fill_(1e-13); q[2].zero_(); q[2, 0] = 1e-12
    q[3, 0] = torch.nan; q[16, 1] = torch.inf
    assert_same(call(library, arrays, kind), scalar_reference(library, arrays, kind))
