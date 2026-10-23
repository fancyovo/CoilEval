"""Owned native center arrays for local reverse evaluation."""
from __future__ import annotations

import ctypes as ct
import numpy as np


def _bind(lib):
    if getattr(lib, "_full_chain_capture_bound", False):
        return
    lib.sgpu_set_reverse_capture.argtypes = [ct.c_int]
    lib.sgpu_set_reverse_capture.restype = ct.c_int
    lib.sgpu_reverse_capture_array_count.argtypes = []
    lib.sgpu_reverse_capture_array_count.restype = ct.c_int
    lib.sgpu_reverse_capture_array_info.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_int, ct.POINTER(ct.c_int64), ct.c_int, ct.POINTER(ct.c_int)]
    lib.sgpu_reverse_capture_array_info.restype = ct.c_int
    lib.sgpu_reverse_capture_copy.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.c_int64]
    lib.sgpu_reverse_capture_copy.restype = ct.c_int
    lib._full_chain_capture_bound = True


def copy_reverse_snapshot(field):
    """Copy the calling thread's latest native snapshot; no pointers are retained."""
    _bind(field.lib)
    count = field.lib.sgpu_reverse_capture_array_count()
    if count <= 0:
        raise RuntimeError("No successful reverse center snapshot on this host thread")
    result = {}
    for index in range(count):
        name = ct.create_string_buffer(128)
        shape = (ct.c_int64 * 4)()
        ndim = ct.c_int()
        field._check(field.lib.sgpu_reverse_capture_array_info(
            index, name, len(name), shape, len(shape), ct.byref(ndim)))
        dims = tuple(shape[i] for i in range(ndim.value))
        values = np.empty(dims, dtype=np.float64)
        field._check(field.lib.sgpu_reverse_capture_copy(
            index, values.ctypes.data_as(ct.POINTER(ct.c_double)), values.size))
        key = name.value.decode("ascii")
        if "_mode_" in key:
            values = values.astype(np.int64)
        result[key] = values
    return result


def _config_dict(config):
    result = {}
    for name, _ in config._fields_:
        value = getattr(config, name)
        result[name] = list(value) if isinstance(value, ct.Array) else value
    return result


def capture_center(field, coeffs_x, coeffs_y, coeffs_z, currents_a, *,
                   target_helicity=(1, 4), config_overrides=None, fast=True,
                   diagnostics=False):
    """Capture a complete native center and return an independently owned dict.

    ``field`` is BatchCoilFieldGpu, including a one-query handle. Multiple dicts
    may coexist safely; capture operations themselves must remain sequential
    within a host thread because the existing native warm cache is thread-local.
    ``diagnostics`` additionally copies native volume/alpha point fields.
    """
    _bind(field.lib)
    field._check(field.lib.sgpu_set_reverse_capture(3 if diagnostics else 1))
    try:
        captured = field.capture_psi_center(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            target_helicity=target_helicity, config_overrides=config_overrides)
        snapshot = copy_reverse_snapshot(field)
        snapshot["config"] = _config_dict(captured["_config_struct"])
        snapshot["score_result"] = captured["score_result"]
        snapshot["_native_capture"] = captured
        snapshot["metadata"] = {
            "capture_version": 1, "independent_host_storage": True,
            "matrix_layout": "row-major mathematical upper triangle; signed diagonal",
            "target_weight_convention": "volume derivatives before -1/dVds",
            "target_adjoint_residual": "signed sum of volume weights times center radial root residual",
            "diagnostics": bool(diagnostics),
        }
        return snapshot
    finally:
        field._check(field.lib.sgpu_set_reverse_capture(0))
