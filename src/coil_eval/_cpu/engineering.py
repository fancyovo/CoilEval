"""Sampled engineering metrics and active-branch analytical CPU derivatives."""
from __future__ import annotations

import ctypes
from operator import index

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .._autograd.engineering import split_tokens
from .native import CpuLibrary, pointer

_NAMES = ("length_mean", "curvature_p95", "curvature_max", "min_intercoil_distance",
          "min_axis_distance", "high_mode_fraction", "current_abs_max")


def metric_gradient(tokens, library, nfp):
    """Return seven values and six native metric Jacobians on CPU."""
    if isinstance(nfp, bool):
        raise ValueError("nfp must be a positive integer")
    try:
        nfp = index(nfp)
    except TypeError as exc:
        raise ValueError("nfp must be a positive integer") from exc
    if nfp < 1:
        raise ValueError("nfp must be a positive integer")
    host = np.asarray(tokens, dtype=np.float64)
    if host.ndim != 2 or host.shape[0] < 1:
        raise ValueError("engineering metrics require nonempty [coil,3*ncoeff+1] tokens")
    arrays = [np.ascontiguousarray(a) for a in split_tokens(host)]
    if not np.isfinite(host).all():
        raise ValueError("coil tokens must be finite")
    if not isinstance(library, CpuLibrary):
        library = CpuLibrary(library)
    function = library.lib.ce_cpu_coil_metric_gradient
    function.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p]*3
    function.restype = ctypes.c_int
    coils, coefficients = arrays[0].shape
    values = np.empty(7, dtype=np.float64)
    derivatives = np.empty((6, 3, coils, coefficients), dtype=np.float64)
    spacing_pair = np.full(2, -1, dtype=np.int32)
    library.check(function(*[pointer(a) for a in arrays], coils, coefficients, nfp,
                           pointer(values), pointer(derivatives), pointer(spacing_pair)))
    return {"values": dict(zip(_NAMES, map(float, values))), "derivatives": derivatives,
            "spacing_pair": spacing_pair.tolist()}


class _EngineeringMetrics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, physical_tokens, library, nfp):
        if physical_tokens.device.type != "cpu" or physical_tokens.dtype not in (torch.float32, torch.float64):
            raise ValueError("CPU FP32 or FP64 coil tokens required")
        host = physical_tokens.detach().double().numpy()
        result = metric_gradient(host, library, nfp)
        coils, width = host.shape
        coefficients = (width - 1) // 3
        derivative = result["derivatives"].transpose(0, 2, 1, 3).reshape(6, coils, 3*coefficients)
        full = np.zeros((3, coils, width), dtype=np.float64)
        full[:, :, :-1] = derivative[[3, 2, 0]]
        ctx.save_for_backward(torch.as_tensor(full, dtype=physical_tokens.dtype))
        return physical_tokens.new_tensor([result["values"][name] for name in
             ("min_intercoil_distance", "curvature_max", "length_mean")])

    @staticmethod
    @once_differentiable
    def backward(ctx, adj_metrics):
        (jacobian,) = ctx.saved_tensors
        return torch.einsum("m,mci->ci", adj_metrics, jacobian), None, None


def engineering_metrics(physical_tokens, library, nfp):
    """Return (gap, maximum curvature, mean length) with live token derivatives."""
    return _EngineeringMetrics.apply(physical_tokens, library, nfp)
