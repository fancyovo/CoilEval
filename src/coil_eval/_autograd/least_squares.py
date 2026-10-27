"""Residual-controlled ridge solves and their first implicit derivatives."""
from contextlib import contextmanager
import math
import threading

import torch
from torch.autograd.function import once_differentiable


_AUTO_MIN_COLUMNS = 1024
_PRECONDITIONER_SHIFT = 1e-6
_RELATIVE_TOLERANCE = 1e-10
_MAX_ITERATIONS = 40
_PRECISION_LOCK = threading.RLock()


@contextmanager
def _ieee_fp32_matmul():
    # The preconditioner experiment was validated with FP32, not TF32 products.
    # Preserve the application's setting outside this one Gram construction.
    with _PRECISION_LOCK:
        previous = torch.get_float32_matmul_precision()
        if previous != 'highest':
            torch.set_float32_matmul_precision('highest')
        try:
            yield
        finally:
            if previous != 'highest':
                torch.set_float32_matmul_precision(previous)


class _NormalSystem:
    """One matrix and reusable preconditioner/fallback factorization."""
    def __init__(self, design, ridge, *, method, shift, tolerance, max_iterations):
        self.design, self.ridge = design, ridge
        self.method, self.shift = method, shift
        self.tolerance, self.max_iterations = tolerance, max_iterations
        self.cholesky = self.factor = self.pivots = None
        self.preconditioner_error = None
        if method == 'dense':
            self._factor_dense()
        else:
            with _ieee_fp32_matmul():
                single = design.float()
                gram = single.T@single
                gram.diagonal().add_(ridge+shift)
                self.cholesky, info = torch.linalg.cholesky_ex(gram)
            if int(info) != 0 or not bool(torch.isfinite(self.cholesky).all()):
                self.preconditioner_error = f'FP32 Cholesky failed, info={int(info)}'
                self.cholesky = None

    def _factor_dense(self):
        gram = self.design.T@self.design
        gram.diagonal().add_(self.ridge)
        self.factor, self.pivots = torch.linalg.lu_factor(gram)

    def matvec(self, vector):
        return self.design.T@(self.design@vector)+self.ridge*vector

    def _precondition(self, vector):
        scale = vector.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
        return torch.cholesky_solve((vector/scale).float()[:, None], self.cholesky)[:, 0].double()*scale

    def _dense_solve(self, right, transpose):
        return torch.linalg.lu_solve(self.factor, self.pivots, right[:, None],
                                    adjoint=transpose)[:, 0]

    def solve(self, right, *, transpose=False):
        norm = torch.linalg.vector_norm(right)
        if not bool(torch.isfinite(norm)):
            raise ValueError('ridge solve received a nonfinite right-hand side')
        stats = dict(iterations=0, residual_restarts=0, fallback=False,
                     fallback_factor_reused=False, transpose=transpose)
        if float(norm) == 0:
            stats.update(relative_normal_residual=0., pcg_relative_normal_residual=0.,
                         gate_passed=True, reason='zero right-hand side')
            return torch.zeros_like(right), stats
        if self.method == 'dense':
            solution = self._dense_solve(right, transpose)
            relative = float(torch.linalg.vector_norm(right-self.matvec(solution))/norm)
            stats.update(relative_normal_residual=relative, gate_passed=relative <= self.tolerance,
                         reason='FP64 dense solve')
            return solution, stats

        solution = torch.zeros_like(right)
        residual = right.clone()
        reason = self.preconditioner_error
        if reason is None:
            z = self._precondition(residual)
            direction, rho = z, residual@z
            for iteration in range(1, self.max_iterations+1):
                stats['iterations'] = iteration
                image = self.matvec(direction)
                curvature = direction@image
                if not bool(torch.isfinite(curvature) & (curvature > 0)
                            & torch.isfinite(rho) & (rho > 0)):
                    reason = 'nonpositive or nonfinite PCG curvature'
                    break
                step = rho/curvature
                solution = solution+step*direction
                residual = residual-step*image
                recursive_norm = torch.linalg.vector_norm(residual)
                if not bool(torch.isfinite(recursive_norm)):
                    reason = 'nonfinite PCG residual'
                    break
                if bool(recursive_norm <= self.tolerance*norm):
                    residual = right-self.matvec(solution)
                    relative = float(torch.linalg.vector_norm(residual)/norm)
                    if relative <= self.tolerance:
                        stats.update(relative_normal_residual=relative,
                                     pcg_relative_normal_residual=relative,
                                     gate_passed=True, reason='true residual accepted')
                        return solution, stats
                    stats['residual_restarts'] += 1
                    z = self._precondition(residual)
                    direction, rho = z, residual@z
                    continue
                z = self._precondition(residual)
                next_rho = residual@z
                direction = z+(next_rho/rho)*direction
                rho = next_rho
            if reason is None:
                reason = 'iteration limit'
        stats.update(pcg_relative_normal_residual=float(
            torch.linalg.vector_norm(right-self.matvec(solution))/norm),
            fallback=True, reason=reason, fallback_factor_reused=self.factor is not None)
        if self.factor is None:
            self._factor_dense()
        solution = self._dense_solve(right, transpose)
        relative = float(torch.linalg.vector_norm(right-self.matvec(solution))/norm)
        stats.update(relative_normal_residual=relative, gate_passed=relative <= self.tolerance)
        return solution, stats

    def saved_state(self):
        """Factors go through save_for_backward, so ordinary backward frees them."""
        empty = self.design.new_empty(0)
        return (self.cholesky if self.cholesky is not None else empty,
                self.factor if self.factor is not None else empty,
                self.pivots if self.pivots is not None else empty.to(torch.int32))

    @classmethod
    def restore(cls, design, ridge, method, shift, tolerance, max_iterations,
                preconditioner_error, cholesky, factor, pivots):
        system = cls.__new__(cls)
        system.design, system.ridge, system.method = design, ridge, method
        system.shift, system.tolerance, system.max_iterations = shift, tolerance, max_iterations
        system.preconditioner_error = preconditioner_error
        system.cholesky = cholesky if cholesky.numel() else None
        system.factor = factor if factor.numel() else None
        system.pivots = pivots if pivots.numel() else None
        return system


class _ImplicitRidgeSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, design, right, ridge, method, shift, tolerance, max_iterations, diagnostics):
        system = _NormalSystem(design, ridge, method=method, shift=shift,
                               tolerance=tolerance, max_iterations=max_iterations)
        solution, diagnostics['forward'] = system.solve(design.T@right)
        diagnostics['preconditioner_error'] = system.preconditioner_error
        ctx.save_for_backward(design, solution, right-design@solution, *system.saved_state())
        # Only scalar settings and a tensor-free diagnostic dict live on ctx.
        ctx.options = ridge, method, shift, tolerance, max_iterations, system.preconditioner_error
        ctx.diagnostics = diagnostics
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, output_adjoint):
        design, solution, residual, cholesky, factor, pivots = ctx.saved_tensors
        system = _NormalSystem.restore(design, *ctx.options, cholesky, factor, pivots)
        adjoint, stats = system.solve(output_adjoint, transpose=True)
        ctx.diagnostics['backward'].append(stats)
        projected = design@adjoint
        d_design = (residual[:, None]*adjoint[None, :]-projected[:, None]*solution[None, :]
                    if ctx.needs_input_grad[0] else None)
        d_right = projected if ctx.needs_input_grad[1] else None
        return d_design, d_right, None, None, None, None, None, None


def ridge_least_squares(design, right, ridge, *, method='auto',
                        preconditioner_shift=_PRECONDITIONER_SHIFT,
                        relative_tolerance=_RELATIVE_TOLERANCE,
                        max_iterations=_MAX_ITERATIONS):
    """Return the ridge solution and scalar-only forward/reverse diagnostics.

    ``auto`` selects mixed preconditioning for CUDA matrices with at least
    1024 columns, and FP64 LU otherwise. ``dense`` is the original reference
    solve; ``mixed`` also enables the bounded experimental path on CPU for
    verification. The FP32 shift affects only the preconditioner. Every PCG
    answer must pass a freshly evaluated FP64 normal residual, or the solve
    falls back to FP64 LU. A forward fallback factor is reused by the adjoint.

    The first reverse derivative is the implicit derivative of the original
    ridge equations, including design and right-hand-side derivatives. Solver
    iterations and preconditioner construction are outside the reverse graph.
    The returned diagnostics dict receives a record for each later backward.
    """
    if not isinstance(method, str) or method not in ('auto', 'dense', 'mixed'):
        raise ValueError('ridge method must be auto, dense or mixed')
    if (not isinstance(design, torch.Tensor) or not isinstance(right, torch.Tensor)
            or design.dtype != torch.float64 or right.dtype != torch.float64
            or design.ndim != 2 or right.shape != design.shape[:1]
            or min(design.shape) < 1 or right.device != design.device):
        raise ValueError('ridge solve needs a nonempty FP64 matrix and a matching FP64 vector')
    if (isinstance(ridge, bool) or not isinstance(ridge, (int, float))
            or not math.isfinite(ridge) or ridge < 0):
        raise ValueError('ridge must be finite and nonnegative')
    if (isinstance(preconditioner_shift, bool) or not math.isfinite(preconditioner_shift)
            or preconditioner_shift <= 0 or isinstance(relative_tolerance, bool)
            or not math.isfinite(relative_tolerance) or relative_tolerance <= 0
            or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError('ridge solver shift, tolerance and iteration limit must be positive')
    resolved = ('mixed' if design.is_cuda and design.shape[1] >= _AUTO_MIN_COLUMNS else 'dense') if method == 'auto' else method
    diagnostics = dict(requested_method=method, method=resolved,
        rows=int(design.shape[0]), columns=int(design.shape[1]), ridge=float(ridge),
        relative_tolerance=float(relative_tolerance), max_iterations=max_iterations,
        preconditioner_shift=float(preconditioner_shift) if resolved == 'mixed' else None,
        equation='FP64 original ridge normal equations', backward=[])
    solution = _ImplicitRidgeSolve.apply(design, right, float(ridge), resolved,
        float(preconditioner_shift), float(relative_tolerance), max_iterations, diagnostics)
    return solution, diagnostics
