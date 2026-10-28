"""CPU field and RK4 operators with analytic first derivatives.

Only the CPU shared library is loaded. Segment construction remains in Torch so
Fourier coefficients, currents, and query positions all retain their derivatives.
"""
from __future__ import annotations

import ctypes
from operator import index

import torch
from torch.autograd.function import once_differentiable

from .._autograd.field import coil_segments
from .native import resolve_cpu_library
from .threads import available_threads

_PRECISION = {torch.float32: "f32", torch.float64: "f64"}


class CpuAutogradLibrary:
    def __init__(self, path=None, threads=None):
        self.path = resolve_cpu_library(path)
        self.threads = available_threads(threads)
        self.lib = ctypes.CDLL(str(self.path))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        for precision in _PRECISION.values():
            for name, count in (("field", 5), ("potential", 4),
                                ("field_vjp", 8), ("potential_vjp", 7)):
                fn = getattr(self.lib, f"ce_cpu_autograd_{name}_{precision}")
                fn.argtypes = [pointer] * count + [integer] * 3
                fn.restype = integer
            for state in _PRECISION.values():
                for name, count in (("trace_forward", 9), ("trace_reverse", 7)):
                    fn = getattr(self.lib, f"ce_cpu_autograd_{name}_{precision}_{state}")
                    fn.argtypes = [pointer] * count + [integer] * 4
                    fn.restype = integer

    def _call(self, name, tensors, integers, precision):
        code = getattr(self.lib, f"ce_cpu_autograd_{name}_{precision}")(
            *[None if t is None else t.data_ptr() for t in tensors],
            *integers, self.threads)
        if code:
            raise RuntimeError(f"CPU {name} rejected its input buffers (code {code})")

    def call(self, name, tensors, ns, np, reference):
        self._call(name, tensors, (ns, np), _PRECISION[reference.dtype])

    def trace_call(self, name, tensors, integers, field_dtype, state_dtype):
        self._call(name, tensors, integers,
                   _PRECISION[field_dtype] + "_" + _PRECISION[state_dtype])


def _validate_field(p, w, q):
    if not all(t.device.type == "cpu" and t.dtype in _PRECISION and
               t.dtype == p.dtype and t.ndim == 2 and t.shape[1] == 3
               for t in (p, w, q)):
        raise ValueError("CPU [N,3] tensors of matching FP32 or FP64 precision required")
    if p.shape != w.shape or not len(p):
        raise ValueError("nonempty matching segment positions/elements required")


class _Field(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, w, q, library, gradient):
        p, w, q = [t.contiguous() for t in (p, w, q)]
        _validate_field(p, w, q)
        b = torch.empty_like(q)
        g = q.new_empty((len(q), 3, 3)) if gradient else None
        library.call("field", (p, w, q, b, g), len(p), len(q), q)
        ctx.save_for_backward(p, w, q)
        ctx.library = library
        ctx.set_materialize_grads(False)
        return (b, g) if gradient else b

    @staticmethod
    @once_differentiable
    def backward(ctx, adj_b, adj_g=None):
        p, w, q = ctx.saved_tensors
        dp, dw, dq = [torch.empty_like(t) if needed else None
                      for t, needed in zip((p, w, q), ctx.needs_input_grad)]
        ctx.library.call("field_vjp", (p, w, q,
            None if adj_b is None else adj_b.contiguous(),
            None if adj_g is None else adj_g.contiguous(), dp, dw, dq),
            len(p), len(q), q)
        return dp, dw, dq, None, None


class _Potential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, w, q, library):
        p, w, q = [t.contiguous() for t in (p, w, q)]
        _validate_field(p, w, q)
        a = torch.empty_like(q)
        library.call("potential", (p, w, q, a), len(p), len(q), q)
        ctx.save_for_backward(p, w, q)
        ctx.library = library
        return a

    @staticmethod
    @once_differentiable
    def backward(ctx, adj):
        p, w, q = ctx.saved_tensors
        dp, dw, dq = [torch.empty_like(t) if needed else None
                      for t, needed in zip((p, w, q), ctx.needs_input_grad)]
        ctx.library.call("potential_vjp", (p, w, q, adj.contiguous(), dp, dw, dq),
                         len(p), len(q), q)
        return dp, dw, dq, None


class _Trace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, w, starts, phi0, phi1, steps, library, samples):
        p, w, starts = [t.contiguous() for t in (p, w, starts)]
        _validate_field(p, w, p)
        if starts.device.type != "cpu" or starts.dtype not in _PRECISION:
            raise ValueError("trace states must be CPU FP32 or FP64 tensors")
        if starts.ndim != 2 or starts.shape[1] != 2 or not len(starts):
            raise ValueError("trace starts must be nonempty [lines,2]")
        lines = len(starts)
        states = starts.new_empty((steps + 1, lines, 2))
        xyz = p.new_empty((4 * steps * lines, 3))
        jacobian = starts.new_empty((4 * steps * lines, 2, 2))
        field_derivative = starts.new_empty((4 * steps * lines, 2, 3))
        library.trace_call("trace_forward", (p, w, starts, phi0, phi1, states,
            xyz, jacobian, field_derivative), (len(p), lines, steps), p.dtype, starts.dtype)
        ctx.save_for_backward(p, w, phi0, phi1, xyz, jacobian, field_derivative)
        ctx.steps, ctx.library, ctx.samples = steps, library, samples
        return states if samples else states[-1]

    @staticmethod
    @once_differentiable
    def backward(ctx, adj):
        p, w, phi0, phi1, xyz, jacobian, field_derivative = ctx.saved_tensors
        dy = phi0.new_empty((len(phi0), 2))
        db = torch.empty_like(xyz)
        ctx.library.trace_call("trace_reverse", (jacobian, field_derivative,
            phi0, phi1, adj.contiguous(), dy, db),
            (len(phi0), ctx.steps, int(ctx.samples)), p.dtype, phi0.dtype)
        dp = torch.empty_like(p) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(w) if ctx.needs_input_grad[1] else None
        if dp is not None or dw is not None:
            # The local state Jacobian already includes moving-point response.
            # Only explicit coil response is accumulated in this second pass.
            ctx.library.call("field_vjp", (p, w, xyz, db, None, dp, dw, None),
                             len(p), len(xyz), xyz)
        return dp, dw, dy, None, None, None, None, None


class CpuSegmentField:
    """SegmentField-compatible CPU primitives with explicit thread ownership."""
    def __init__(self, positions, elements, library):
        _validate_field(positions, elements, positions)
        self.positions, self.elements, self.library = positions, elements, library

    @classmethod
    def from_tokens(cls, tokens, library, nfp=4, count=256):
        if tokens.device.type != "cpu" or tokens.dtype not in _PRECISION:
            raise ValueError("coil tokens must be CPU FP32 or FP64 tensors")
        return cls(*coil_segments(tokens, nfp, count), library)

    def __call__(self, xyz, gradient=False):
        return _Field.apply(self.positions, self.elements,
                            xyz.to(self.positions.dtype), self.library, gradient)

    def potential(self, xyz):
        return _Potential.apply(self.positions, self.elements,
                                xyz.to(self.positions.dtype), self.library)

    def trace(self, rz, phi0, phi1, steps, return_samples=False):
        if isinstance(steps, bool):
            raise ValueError("positive integer RK4 step count required")
        try:
            steps = index(steps)
        except TypeError as exc:
            raise ValueError("positive integer RK4 step count required") from exc
        if steps < 1:
            raise ValueError("positive integer RK4 step count required")
        if rz.ndim != 2 or rz.shape[1] != 2 or not len(rz):
            raise ValueError("trace starts must be nonempty [lines,2]")
        bounds = [torch.broadcast_to(torch.as_tensor(phi, device=rz.device,
                    dtype=rz.dtype).detach(), (len(rz),)).contiguous()
                  for phi in (phi0, phi1)]
        return _Trace.apply(self.positions, self.elements, rz, *bounds,
                            steps, self.library, return_samples)
