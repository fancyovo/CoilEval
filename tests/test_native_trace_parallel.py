"""Exact native trace regression across block widths, precision and query batches."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pytest

from coil_eval import CoilSet
from coil_eval._binding import BatchCoilFieldGpu, CoilFieldGpu


LIBRARY = os.environ.get('COIL_EVAL_TEST_LIBRARY')
REFERENCE = os.environ.get('COIL_EVAL_REFERENCE_LIBRARY')
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not LIBRARY or not REFERENCE, reason='set candidate and reference native libraries')]


@pytest.fixture(scope='module')
def coils():
    coils = CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    return coils, [coils.coeffs_x, coils.coeffs_y, coils.coeffs_z, coils.currents_a]


@pytest.mark.parametrize('threads', [128, 256, 512, 1024])
@pytest.mark.parametrize('precision', ['fp64', 'mixed64', 'fp32'])
@pytest.mark.parametrize('segments', [17, 256])
def test_blockline_endpoints_preserve_reference_bits(coils, threads, precision, segments):
    config, arrays = coils
    outputs = []
    for path in (REFERENCE, LIBRARY):
        field = CoilFieldGpu(path, *arrays, config.nfp, segments_per_coil=segments)
        try:
            try:
                value = field.trace_period_blockline_precision(
                    [.98, 1., 1.02], [-.01, 0., .01], steps=12,
                    precision=precision, threads_per_line=threads)
            except RuntimeError as exc:
                # Unsupported launch configurations remain part of the
                # contract: candidate and frozen reference must fail with the
                # same native error rather than silently dropping the case.
                outputs.append(('error', str(exc)))
            else:
                outputs.append(('ok', value))
        finally:
            field.close()
    assert outputs[0][0] == outputs[1][0]
    if outputs[0][0] == 'error':
        assert outputs[1][1] == outputs[0][1]
    else:
        for expected, actual in zip(outputs[0][1], outputs[1][1]):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('batch', [False, True])
def test_axis_samples_and_batch_tracks_preserve_reference_bits(coils, batch):
    config, arrays = coils
    if batch:
        arrays = [np.repeat(v[None], 3, axis=0) for v in arrays]
        arrays[0][1, 0, 0] += 1e-4
        arrays[3][2] *= 1.001
    outputs = []
    for path in (REFERENCE, LIBRARY):
        field = (BatchCoilFieldGpu if batch else CoilFieldGpu)(path, *arrays, config.nfp)
        try:
            if batch:
                tracks = field.trace_period(np.full((3, 5), 1.), np.zeros((3, 5)), steps=16)
                samples = field.trace_axis_samples(np.ones(3), np.zeros(3),
                                                  integration_steps=16, sample_count=8)
            else:
                tracks = field.trace_period_blockline_precision([1.]*5, [0.]*5, 16)
                samples = field.trace_axis_samples(1., 0., integration_steps=16, sample_count=8)
            outputs.append((*tracks, *samples))
        finally:
            field.close()
    for expected, actual in zip(*outputs):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('batch', [False, True])
def test_reused_buffers_grow_shrink_and_interleave_independent_handles(coils, batch):
    config, original = coils
    handles = []
    try:
        pairs = []
        for shift in (0., 2e-4):
            arrays = [v.copy() for v in original]
            arrays[0][0, 0] += shift
            if batch:
                arrays = [np.repeat(v[None], 2, axis=0) for v in arrays]
                arrays[0][1, 0, 0] += 1e-4
            pair = []
            for path in (REFERENCE, LIBRARY):
                field = (BatchCoilFieldGpu if batch else CoilFieldGpu)(
                    path, *arrays, config.nfp, segments_per_coil=17)
                handles.append(field)
                pair.append(field)
            pairs.append(pair)
        for kind, count in [('period', 1), ('period', 33), ('axis', 8), ('period', 257),
                            ('axis', 16), ('period', 5), ('period', 257)]:
            for index, pair in enumerate(pairs):
                outputs = []
                for field in pair:
                    if kind == 'axis':
                        outputs.append(field.trace_axis_samples(
                            np.ones(2) if batch else 1., np.zeros(2) if batch else 0.,
                            integration_steps=16, sample_count=count))
                    else:
                        R = np.linspace(.98+index*1e-3, 1.02, count)
                        Z = np.linspace(-.01, .01, count)
                        outputs.append(field.trace_period(np.tile(R, (2, 1)), np.tile(Z, (2, 1)), steps=8)
                                       if batch else field.trace_period_blockline_precision(
                                           R, Z, 8, precision='mixed64' if index else 'fp64'))
                for expected, actual in zip(*outputs):
                    np.testing.assert_array_equal(actual, expected)
        # A rejected call must leave the cached workspace available to the next
        # valid call, with no stale outputs or retained mutex ownership.
        candidate = pairs[0][1]
        with pytest.raises(RuntimeError):
            if batch:
                candidate.trace_period(np.ones((2, 1)), np.zeros((2, 1)), steps=0)
            else:
                candidate.trace_period_blockline_precision([1.], [0.], 0)
        if batch:
            actual = candidate.trace_period(np.ones((2, 1)), np.zeros((2, 1)), steps=8)
            expected = pairs[0][0].trace_period(np.ones((2, 1)), np.zeros((2, 1)), steps=8)
        else:
            actual = candidate.trace_period_blockline_precision([1.], [0.], 8)
            expected = pairs[0][0].trace_period_blockline_precision([1.], [0.], 8)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
    finally:
        for field in reversed(handles):
            field.close()


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('same_handle', [False, True])
def test_worker_threads_preserve_trace_buffers_and_host_results(coils, batch, same_handle):
    config, arrays = coils
    if batch:
        arrays = [np.repeat(v[None], 2, axis=0) for v in arrays]
    constructor = BatchCoilFieldGpu if batch else CoilFieldGpu
    reference = constructor(REFERENCE, *arrays, config.nfp, segments_per_coil=17)
    candidates = [constructor(LIBRARY, *arrays, config.nfp, segments_per_coil=17)
                  for _ in range(1 if same_handle else 2)]
    barrier = threading.Barrier(2)

    def operation(field, count):
        R, Z = np.linspace(.98, 1.02, count), np.linspace(-.01, .01, count)
        return (field.trace_period(np.tile(R, (2, 1)), np.tile(Z, (2, 1)), steps=8) if batch
                else field.trace_period_blockline_precision(R, Z, 8))

    def worker(index):
        barrier.wait(timeout=10)
        return [operation(candidates[0 if same_handle else index], count)
                for count in ([1, 129, 5] if index == 0 else [33, 3, 257])]

    try:
        expected = [[operation(reference, count) for count in counts]
                    for counts in ([1, 129, 5], [33, 3, 257])]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            actual = [future.result(timeout=60) for future in futures]
        for sequence, truth in zip(actual, expected):
            for outputs, wanted in zip(sequence, truth):
                for a, e in zip(outputs, wanted):
                    np.testing.assert_array_equal(a, e)
    finally:
        reference.close()
        for field in candidates:
            field.close()
