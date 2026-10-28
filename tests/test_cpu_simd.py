"""Eight-query AVX2 lanes agree bit-for-bit with the retained scalar path."""
import ctypes as ct
import os

import numpy as np
import pytest

from coil_eval._cpu.native import CpuLibrary, pointer


@pytest.fixture
def library():
    path = os.environ.get('COIL_EVAL_CPU_LIBRARY')
    if not path:
        pytest.skip('set COIL_EVAL_CPU_LIBRARY')
    lib = CpuLibrary(path)
    enabled = getattr(lib.lib, 'ce_cpu_avx2_enabled', None)
    assert enabled is not None, 'CPU test library lacks SIMD feature detection'
    enabled.argtypes = []; enabled.restype = ct.c_int
    assert lib.avx2_enabled == bool(enabled())
    return lib


@pytest.fixture(params=[1, 3])
def field(library, request):
    rng = np.random.default_rng(113)
    positions = np.ascontiguousarray(rng.normal(size=(37, 3))*.3)
    elements = np.ascontiguousarray(rng.normal(size=(37, 3))*2e4)
    handle = ct.c_void_p()
    library.check(library.lib.ce_cpu_open(pointer(positions), pointer(elements),
                  len(positions), request.param, ct.byref(handle)))
    try:
        yield library, handle
    finally:
        library.lib.ce_cpu_close(handle)


def evaluate(field, xyz, gradient, potential, mixed=1):
    library, handle = field
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    B = np.empty_like(xyz)
    G = np.empty((len(xyz), 3, 3)) if gradient else None
    A = np.empty_like(xyz) if potential else None
    library.check(library.lib.ce_cpu_field(handle, pointer(xyz), len(xyz), mixed,
                  pointer(B), pointer(G), pointer(A)))
    return [a for a in (B, G, A) if a is not None]


@pytest.mark.parametrize('count', [0, 1, 7, 8, 9, 17, 63])
@pytest.mark.parametrize('gradient,potential', [(False, False), (True, False),
                                              (False, True), (True, True)])
def test_field_lane_order_tails_and_optional_outputs(field, count, gradient, potential):
    rng = np.random.default_rng(9)
    xyz = rng.normal(size=(count, 3))*.15+np.array([1.1, .02, -.03])
    actual = evaluate(field, xyz, gradient, potential)
    if not count:
        assert all(len(value) == 0 for value in actual)
        return
    # count=1 is always the original scalar segment loop, in the same binary.
    reference = [evaluate(field, point[None], gradient, potential) for point in xyz]
    for column, value in enumerate(actual):
        expected = np.concatenate([row[column] for row in reference])
        np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))


def trace(field, R, Z, mixed=1):
    library, handle = field
    R, Z = np.ascontiguousarray(R), np.ascontiguousarray(Z)
    endR, endZ = np.empty_like(R), np.empty_like(Z)
    library.check(library.lib.ce_cpu_trace(handle, pointer(R), pointer(Z),
        pointer(endR), pointer(endZ), len(R), .03, .04, 7, mixed))
    return np.column_stack((endR, endZ))


@pytest.mark.parametrize('count', [0, 1, 7, 8, 9, 17, 33])
@pytest.mark.parametrize('mixed', [0, 1])
def test_trace_grouping_preserves_every_scalar_rk4_step(field, count, mixed):
    R = np.linspace(1.02, 1.15, count)
    Z = np.linspace(-.04, .04, count)
    actual = trace(field, R, Z, mixed)
    expected = np.concatenate([trace(field, R[k:k+1], Z[k:k+1], mixed)
                               for k in range(count)]) if count else np.empty((0, 2))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_double_field_does_not_change_precision(field):
    xyz = np.random.default_rng(412).normal(size=(17, 3))*.2+np.array([1.1, 0., 0.])
    actual = evaluate(field, xyz, True, True, mixed=0)
    reference = [evaluate(field, point[None], True, True, mixed=0) for point in xyz]
    for column, value in enumerate(actual):
        expected = np.concatenate([row[column] for row in reference])
        np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))


def test_nonfinite_query_lane_cannot_contaminate_other_queries(field):
    xyz = np.random.default_rng(7).normal(size=(17, 3))*.2+np.array([1.1, 0., 0.])
    xyz[2, 0], xyz[5, 1], xyz[16, 2] = np.nan, np.inf, -np.inf
    actual = evaluate(field, xyz, True, True)
    reference = [evaluate(field, point[None], True, True) for point in xyz]
    for column, value in enumerate(actual):
        expected = np.concatenate([row[column] for row in reference])
        # NaN payloads are not a contract; finite lanes still agree bit-for-bit.
        np.testing.assert_array_equal(value, expected)
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(value[finite].view(np.uint64), expected[finite].view(np.uint64))


def test_nonfinite_trace_lane_cannot_contaminate_other_trajectories(field):
    R, Z = np.linspace(1.02, 1.15, 17), np.linspace(-.04, .04, 17)
    R[2], Z[5], Z[16] = np.nan, np.inf, -np.inf
    actual = trace(field, R, Z)
    expected = np.concatenate([trace(field, R[k:k+1], Z[k:k+1]) for k in range(len(R))])
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))
