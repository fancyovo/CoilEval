import pytest

torch = pytest.importorskip('torch')
from coil_eval._autograd.triangular import independent_triangular_solve


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('nrhs', [1, 4])
def test_independent_triangular_values_and_rhs_vjp(transpose, nrhs):
    generator = torch.Generator().manual_seed(164)
    factor = torch.triu(torch.randn(3, 9, 9, dtype=torch.float64, generator=generator) * .03)
    factor = factor + torch.eye(9, dtype=factor.dtype) * 2.
    matrix = factor.mT if transpose else factor
    rhs = torch.randn(3, 9, nrhs, dtype=matrix.dtype, generator=generator, requires_grad=True)
    result = independent_triangular_solve(matrix, rhs, upper=not transpose)
    torch.testing.assert_close(matrix @ result, rhs, rtol=2e-14, atol=2e-14)
    desired = torch.randn(rhs.shape, dtype=rhs.dtype, generator=generator)
    desired[1:] = 0.
    gradient, = torch.autograd.grad(result, rhs, matrix.mT @ desired)
    torch.testing.assert_close(gradient, desired, rtol=2e-14, atol=2e-14)
    assert torch.count_nonzero(gradient[1:]) == 0


def test_triangular_matrix_and_rhs_gradcheck():
    matrix = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    rhs = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
    def solve(a, b):
        upper = torch.triu(a) * .05 + 2. * torch.eye(4, dtype=a.dtype)
        return independent_triangular_solve(upper.mT, b, upper=False)
    assert torch.autograd.gradcheck(solve, (matrix, rhs))


def test_triangular_calls_are_independent(monkeypatch):
    original = torch.linalg.solve_triangular
    shapes = []
    def scalar_only(matrix, rhs, **kwargs):
        shapes.append((matrix.ndim, rhs.ndim))
        assert matrix.ndim == rhs.ndim == 2
        return original(matrix, rhs, **kwargs)
    monkeypatch.setattr(torch.linalg, 'solve_triangular', scalar_only)
    factor = torch.eye(4).repeat(16, 1, 1)
    rhs = torch.ones(16, 4, 1, requires_grad=True)
    independent_triangular_solve(factor.mT, rhs, upper=False).sum().backward()
    assert len(shapes) == 16
    torch.testing.assert_close(rhs.grad, torch.ones_like(rhs))


@pytest.mark.parametrize('matrix,rhs', [((0, 4, 4), (0, 4, 1)),
    ((2, 4, 4), (3, 4, 1)), ((2, 3, 4), (2, 4, 1)), ((2, 4, 4), (2, 3, 1))])
def test_triangular_rejects_mismatched_query_systems(matrix, rhs):
    with pytest.raises(ValueError, match='independent triangular systems'):
        independent_triangular_solve(torch.empty(matrix), torch.empty(rhs), upper=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA regression requires a GPU')
@pytest.mark.parametrize('n,dtype', [(1574, torch.float64), (2272, torch.float32)])
def test_cuda_large_transposed_batch_values_and_vjp(n, dtype):
    # Native psi/alpha-sized factors, B=16. Construct exact RHS/cotangents from
    # known solutions rather than using the problematic batched solve oracle.
    device = torch.device('cuda:0')
    i = torch.arange(n, device=device, dtype=dtype)
    upper = torch.triu(torch.sin(i[:, None] * .013 + i[None, :] * .017) * (.03/n), 1)
    upper = upper + torch.diag(1.5 + i/n)
    matrix = upper.repeat(16, 1, 1).mT
    desired = torch.cos(i * .019).reshape(1, n, 1).expand(16, -1, -1)
    rhs = (matrix @ desired).detach().requires_grad_()
    result = independent_triangular_solve(matrix, rhs, upper=False)
    tolerance = 5e-6 if dtype == torch.float32 else 5e-12
    torch.testing.assert_close(result, desired, rtol=tolerance, atol=tolerance)
    wanted_gradient = torch.sin(i * .023).reshape(1, n, 1).expand(16, -1, -1).clone()
    wanted_gradient[1:] = 0.
    gradient, = torch.autograd.grad(result, rhs, matrix.mT @ wanted_gradient)
    torch.testing.assert_close(gradient, wanted_gradient, rtol=tolerance, atol=tolerance)
    assert torch.count_nonzero(gradient[1:]) == 0
