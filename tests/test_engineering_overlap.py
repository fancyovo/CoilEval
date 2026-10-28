"""CPU engineering overlap preserves scalar values, VJPs and worker ownership."""
import threading

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd import engineering


def fake_native(library, x, y, z, currents, nfp):
    derivatives = np.zeros((6, 3, *x.shape))
    derivatives[0, 0] = 2*x
    derivatives[2, 1] = 3*y*y
    derivatives[3, 2] = 2*z
    return dict(values=dict(length_mean=float((x*x).sum()),
                            curvature_max=float((y*y*y).sum()),
                            min_intercoil_distance=float((z*z).sum())),
                derivatives=derivatives)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_prepared_batch_matches_scalar_values_and_complete_vjp(monkeypatch, dtype):
    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', fake_native)
    tokens = torch.linspace(-.7, 1.3, 3*4*10, dtype=dtype).reshape(3, 4, 10).requires_grad_()
    with engineering.prepare_engineering_batch(tokens.detach().numpy(), 'fake.so', 4) as work:
        actual = work.tensor(tokens)
    expected = torch.stack([engineering._EngineeringMetrics.apply(row, 'fake.so', 4) for row in tokens])
    weight = torch.linspace(.2, 1.1, 9, dtype=dtype).reshape(3, 3)
    actual_grad, = torch.autograd.grad((actual*weight).sum(), tokens, retain_graph=True)
    expected_grad, = torch.autograd.grad((expected*weight).sum(), tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    isolated, = torch.autograd.grad(actual[1].sum(), tokens)
    assert torch.count_nonzero(isolated[[0, 2]]).item() == 0
    assert torch.count_nonzero(actual_grad[..., -1]).item() == 0


def test_worker_owns_snapshot_and_upload_occurs_once_on_caller(monkeypatch):
    caller = threading.get_ident()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    native_threads, upload_threads = [], []

    def native(*args):
        native_threads.append(threading.get_ident())
        started.set()
        assert release.wait(5), 'caller did not release engineering worker'
        result = fake_native(*args)
        finished.set()
        return result

    original_as_tensor = torch.as_tensor

    def as_tensor(*args, **kwargs):
        upload_threads.append(threading.get_ident())
        return original_as_tensor(*args, **kwargs)

    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', native)
    monkeypatch.setattr(engineering.torch, 'as_tensor', as_tensor)
    host = np.linspace(.1, 1.2, 2*4*10).reshape(2, 4, 10)
    original = host.copy()
    with engineering.prepare_engineering_batch(host, 'fake.so') as work:
        assert started.wait(5), 'engineering worker did not start'
        # The caller remains free to enqueue its independent GPU work here.
        assert not finished.is_set()
        host.fill(0)
        release.set()
        actual = work.tensor(torch.tensor(original))
    expected = [[fake_native('fake.so', *engineering.split_tokens(row), 4)['values'][name]
                 for name in ('min_intercoil_distance', 'curvature_max', 'length_mean')]
                for row in original]
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert len(set(native_threads)) == 1 and native_threads[0] != caller
    assert upload_threads == [caller]
    assert finished.is_set()


def test_worker_exception_propagates_when_joined(monkeypatch):
    def native(*args):
        raise RuntimeError('native engineering failed')
    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', native)
    tokens = torch.ones((2, 4, 10))
    with pytest.raises(RuntimeError, match='native engineering failed'):
        with engineering.prepare_engineering_batch(tokens.numpy(), 'fake.so') as work:
            work.tensor(tokens)


def test_context_joins_worker_when_downstream_raises(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def native(*args):
        started.set()
        assert release.wait(5)
        try:
            return fake_native(*args)
        finally:
            finished.set()

    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', native)
    with pytest.raises(ValueError, match='downstream failed'):
        with engineering.prepare_engineering_batch(np.ones((1, 4, 10)), 'fake.so'):
            assert started.wait(5)
            release.set()
            raise ValueError('downstream failed')
    assert finished.is_set()


@pytest.mark.parametrize('shape', [(4, 10), (0, 4, 10), (2, 0, 10), (2, 4, 8)])
def test_bad_host_shape_rejected_before_native_work(monkeypatch, shape):
    def forbidden(*args):
        pytest.fail('invalid engineering input reached native worker')
    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', forbidden)
    with pytest.raises(ValueError):
        with engineering.prepare_engineering_batch(np.zeros(shape), 'fake.so'):
            pass


def test_prepared_work_rejects_mismatched_tensor_shape(monkeypatch):
    monkeypatch.setattr(engineering.native, 'coil_metric_gradient_native', fake_native)
    with engineering.prepare_engineering_batch(np.zeros((2, 4, 10)), 'fake.so') as work:
        with pytest.raises(ValueError, match='batch shape'):
            work.tensor(torch.zeros((1, 4, 10)))
