"""CUDA field/RK4 values, moving-point VJPs, batch isolation and stream ordering."""
import os
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd.batch_field import BatchSegmentField
from coil_eval._autograd.field import NativeFieldLibrary, SegmentField
from test_cpu_autograd import oracle, torch_trace


@pytest.fixture(scope='module')
def cuda_library():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    explicit = os.environ.get('COIL_EVAL_TEST_AUTOGRAD_LIBRARY')
    native = os.environ.get('COIL_EVAL_TEST_LIBRARY')
    if not explicit and not native:
        pytest.skip('set COIL_EVAL_TEST_AUTOGRAD_LIBRARY or COIL_EVAL_TEST_LIBRARY')
    return NativeFieldLibrary(explicit or str(Path(native).with_name('libcoil_eval_autograd.so')))


def inputs(dtype, queries=3):
    generator = torch.Generator(device='cuda').manual_seed(3748)
    p = torch.randn((queries, 257, 3), generator=generator, device='cuda', dtype=dtype)*.1
    w = torch.randn(p.shape, generator=generator, device='cuda', dtype=dtype)*1e5
    w[..., 2] += 8e5
    q = torch.randn((queries, 7, 3), generator=generator, device='cuda', dtype=dtype)*.2 + 1.3
    return tuple(t.requires_grad_() for t in (p, w, q))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('component', ['field', 'gradient', 'both', 'potential'])
def test_batched_field_and_reverse_match_independent_oracle(cuda_library, dtype, component):
    # All input generation, native kernels and dependent Torch kernels run on a
    # nondefault stream; a default-stream launch would race the queued inputs.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p, w, q = inputs(dtype)
        field = BatchSegmentField(p, w, cuda_library)
        potential = component == 'potential'
        actual = field.potential(q) if potential else field(q, gradient=True)
        expected_rows = [oracle(p[i], w[i], q[i], potential) for i in range(len(p))]
        if potential:
            actual, expected = (actual,), (torch.stack(expected_rows),)
        else:
            expected = tuple(torch.stack([row[j] for row in expected_rows]) for j in range(2))
            if component == 'field':
                actual, expected = actual[:1], expected[:1]
            elif component == 'gradient':
                actual, expected = actual[1:], expected[1:]
        weights = [torch.linspace(.2, 1.1, t.numel(), device='cuda', dtype=dtype).reshape(t.shape)
                   for t in actual]
        actual_grad = torch.autograd.grad(sum((a*v).sum() for a, v in zip(actual, weights)), (p, w, q))
        expected_grad = torch.autograd.grad(sum((e*v).sum() for e, v in zip(expected, weights)), (p, w, q))
    stream.synchronize()
    tolerance = dict(rtol=5e-5, atol=5e-8) if dtype == torch.float32 else dict(rtol=3e-11, atol=3e-13)
    for a, e in zip((*actual, *actual_grad), (*expected, *expected_grad)):
        torch.testing.assert_close(a, e, **tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize('field_dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('samples', [False, True])
def test_batched_rk4_matches_single_and_torch(cuda_library, field_dtype, state_dtype, samples):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p, w, _ = inputs(field_dtype, queries=2)
        starts = torch.tensor([[[1.2, .2], [1.4, -.1]], [[1.1, .1], [1.3, -.2]]],
                              device='cuda', dtype=state_dtype, requires_grad=True)
        phi0 = starts.new_tensor([[.03], [.04]])
        phi1 = starts.new_tensor([[.16], [.18]])
        actual = BatchSegmentField(p, w, cuda_library).trace(starts, phi0, phi1, 5, return_samples=samples)
        independent, expected = [], []
        for i in range(len(p)):
            independent.append(SegmentField(p[i], w[i], cuda_library).trace(
                starts[i], phi0[i], phi1[i], 5, return_samples=samples))
            expected.append(torch_trace(p[i], w[i], starts[i], phi0[i].expand(2),
                                        phi1[i].expand(2), 5, samples))
        independent = torch.stack(independent, dim=1 if samples else 0)
        expected = torch.stack(expected, dim=1 if samples else 0)
        weight = torch.linspace(.2, 1.1, actual.numel(), device='cuda', dtype=state_dtype).reshape(actual.shape)
        actual_grad = torch.autograd.grad((actual*weight).sum(), (p, w, starts), retain_graph=True)
        independent_grad = torch.autograd.grad((independent*weight).sum(), (p, w, starts))
        expected_grad = torch.autograd.grad((expected*weight).sum(), (p, w, starts))
        # A single-query objective must leave every other query exactly zero.
        selected = actual[:, 0] if samples else actual[0]
        isolated, = torch.autograd.grad(selected.sum(), p)
    stream.synchronize()
    for a, e in zip((actual, *actual_grad), (independent, *independent_grad)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    tolerance = dict(rtol=1e-4, atol=5e-7) if torch.float32 in (field_dtype, state_dtype) else dict(rtol=3e-10, atol=5e-13)
    for a, e in zip((actual, *actual_grad), (expected, *expected_grad)):
        torch.testing.assert_close(a, e, **tolerance)
    assert torch.count_nonzero(isolated[1]).item() == 0


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scalar_field_matches_batch_values_and_vjp(cuda_library, dtype):
    p, w, q = inputs(dtype, queries=1)
    batch = BatchSegmentField(p, w, cuda_library)(q, gradient=True)
    single = SegmentField(p[0], w[0], cuda_library)(q[0], gradient=True)
    batch_grad = torch.autograd.grad(sum(t.square().sum() for t in batch), (p, w, q))
    single_grad = torch.autograd.grad(sum(t.square().sum() for t in single), (p, w, q))
    for a, e in zip(batch, single):
        torch.testing.assert_close(a[0], e, rtol=0, atol=0)
    for a, e in zip(batch_grad, single_grad):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
