"""Residual-controlled linear solve, exact implicit VJP and bounded fallback."""
import json

import pytest

torch = pytest.importorskip('torch')
from coil_eval._autograd.least_squares import ridge_least_squares


def problem(rows=36, columns=8, condition=20):
    generator = torch.Generator().manual_seed(62027)
    left = torch.linalg.qr(torch.randn(rows, columns, generator=generator, dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(columns, columns, generator=generator, dtype=torch.float64))[0]
    singular = torch.logspace(0, -torch.log10(torch.tensor(float(condition))), columns, dtype=torch.float64)
    design = ((left*singular)@right.T).requires_grad_()
    values = torch.randn(rows, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(columns, generator=generator, dtype=torch.float64)
    return design, values, weights


@pytest.mark.parametrize('method', ['auto', 'dense', 'mixed'])
@pytest.mark.parametrize('ridge', [0., 1e-12, .03])
def test_original_ridge_solution_and_rank_two_vjp(method, ridge):
    design, values, weights = problem()
    actual, diagnostics = ridge_least_squares(design, values, ridge, method=method)
    reference = torch.linalg.solve(design.T@design+ridge*torch.eye(8, dtype=design.dtype), design.T@values)
    expected_gradients = torch.autograd.grad(reference@weights, (design, values))
    actual_gradients = torch.autograd.grad(actual@weights, (design, values))
    torch.testing.assert_close(actual, reference, rtol=1e-8, atol=1e-9)
    for result, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(result, expected, rtol=2e-7, atol=2e-7)
    assert diagnostics['method'] == ('dense' if method == 'auto' else method)
    assert diagnostics['forward']['gate_passed']
    assert not diagnostics['forward']['fallback']
    assert diagnostics['backward'][0]['gate_passed']
    assert not diagnostics['backward'][0]['fallback']
    assert diagnostics['backward'][0]['transpose']
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize('method', ['dense', 'mixed'])
def test_design_and_right_gradcheck(method):
    design, values, _ = problem(rows=9, columns=3, condition=3)
    function = lambda a, b: ridge_least_squares(a, b, 1e-7, method=method)[0]
    assert torch.autograd.gradcheck(function, (design, values), atol=2e-6, rtol=2e-5)


def test_preconditioner_shift_does_not_change_the_ridge_objective():
    design, values, _ = problem(condition=50)
    actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed',
        preconditioner_shift=.01, max_iterations=100)
    expected, _ = ridge_least_squares(design, values, 1e-12, method='dense')
    shifted, _ = ridge_least_squares(design, values, .01, method='dense')
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
    assert torch.linalg.vector_norm(shifted-expected)/torch.linalg.vector_norm(expected) > .1
    assert diagnostics['forward']['gate_passed']


def test_iteration_limit_falls_back_and_forward_factor_is_reused(monkeypatch):
    design, values, weights = problem(condition=200)
    original = torch.linalg.lu_factor
    calls = []
    def counted(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(torch.linalg, 'lu_factor', counted)
    actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed',
        preconditioner_shift=.01, max_iterations=1)
    gradients = torch.autograd.grad(actual@weights, (design, values))
    assert calls == [torch.Size([8, 8])]
    forward, reverse = diagnostics['forward'], diagnostics['backward'][0]
    assert forward['fallback'] and reverse['fallback']
    assert forward['reason'] == reverse['reason'] == 'iteration limit'
    assert not forward['fallback_factor_reused'] and reverse['fallback_factor_reused']
    assert forward['pcg_relative_normal_residual'] > 1e-10
    expected, _ = ridge_least_squares(design, values, 1e-12, method='dense')
    expected_gradients = torch.autograd.grad(expected@weights, (design, values))
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_failed_fp32_cholesky_preserves_dense_fallback_and_setting(monkeypatch):
    design, values, weights = problem()
    previous = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', lambda matrix:
        (torch.full_like(matrix, float('nan')), torch.tensor(2)))
    actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed')
    torch.autograd.grad(actual@weights, (design, values))
    assert diagnostics['forward']['fallback']
    assert diagnostics['forward']['iterations'] == 0
    assert 'Cholesky failed' in diagnostics['preconditioner_error']
    assert diagnostics['backward'][0]['fallback_factor_reused']
    assert torch.backends.cuda.matmul.allow_tf32 == previous


@pytest.mark.parametrize('precision', ['highest', 'high', 'medium'])
def test_preconditioner_restores_exact_application_matmul_precision(monkeypatch, precision):
    design, values, _ = problem()
    previous = torch.get_float32_matmul_precision()
    seen = []
    original = torch.linalg.cholesky_ex
    def cholesky(matrix):
        seen.append(torch.get_float32_matmul_precision())
        return original(matrix)
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', cholesky)
    try:
        torch.set_float32_matmul_precision(precision)
        ridge_least_squares(design, values, 1e-12, method='mixed')
        assert seen == ['highest']
        assert torch.get_float32_matmul_precision() == precision
    finally:
        torch.set_float32_matmul_precision(previous)


def test_zero_rhs_still_has_correct_right_hand_side_derivative():
    design, values, weights = problem()
    values = torch.zeros_like(values, requires_grad=True)
    actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed')
    da, db = torch.autograd.grad(actual@weights, (design, values))
    assert torch.count_nonzero(actual) == torch.count_nonzero(da) == 0
    assert torch.isfinite(db).all() and torch.count_nonzero(db) > 0
    assert diagnostics['forward']['reason'] == 'zero right-hand side'
    assert diagnostics['forward']['iterations'] == 0


def test_factors_use_saved_tensor_hooks_and_are_released_after_backward():
    design, values, weights = problem()
    saved = []
    def pack(value):
        saved.append((tuple(value.shape), value.dtype))
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed')
    assert ((8, 8), torch.float32) in saved
    assert ((8, 8), torch.float64) not in saved
    function = actual.grad_fn
    assert set(function.__dict__) == {'options', 'diagnostics'}
    torch.autograd.grad(actual@weights, (design, values))
    with pytest.raises(RuntimeError, match='freed'):
        function.saved_tensors
    assert len(diagnostics['backward']) == 1


def test_multiple_backward_diagnostics_are_live_and_tensor_free():
    design, values, weights = problem()
    actual, diagnostics = ridge_least_squares(design, values, 1e-12, method='mixed')
    torch.autograd.grad(actual@weights, values, retain_graph=True)
    torch.autograd.grad(actual.square().sum(), values)
    assert len(diagnostics['backward']) == 2
    assert json.loads(json.dumps(diagnostics)) == diagnostics


@pytest.mark.parametrize('options', [dict(method='unknown'), dict(method=[]),
    dict(preconditioner_shift=0), dict(relative_tolerance=0), dict(max_iterations=0),
    dict(max_iterations=True), dict(max_iterations=1.5)])
def test_invalid_solver_settings(options):
    design, values, _ = problem()
    with pytest.raises(ValueError):
        ridge_least_squares(design, values, 1e-12, **options)


def test_invalid_matrix_or_ridge_inputs():
    design, values, _ = problem()
    for a, b, ridge in ((design.float(), values, 0), (design, values.float(), 0),
                        (design, values[:3], 0), (design, values, -1),
                        (design, values, float('nan')), (design, values, True)):
        with pytest.raises(ValueError):
            ridge_least_squares(a, b, ridge)
