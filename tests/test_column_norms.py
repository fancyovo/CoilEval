import pytest

torch = pytest.importorskip('torch')
from coil_eval._autograd.column_norms import column_norms


@pytest.mark.parametrize('shape', [(37, 19), (3, 37, 19)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_column_norm_values_and_vjp(shape, dtype):
    generator = torch.Generator().manual_seed(517)
    matrix = torch.randn(shape, dtype=dtype, generator=generator)
    matrix[..., 0] = 0
    matrix[..., 1] *= 1e-25
    matrix.requires_grad_()
    expected = torch.linalg.vector_norm(matrix, dim=-2, dtype=torch.float64)
    actual = column_norms(matrix, chunk=7)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=0.)
    weight = torch.randn(expected.shape, dtype=dtype, generator=generator)
    def objective(value):
        return (value.clamp_min(1e-20).to(dtype).square() * weight).sum()
    reference, = torch.autograd.grad(objective(expected), matrix)
    result, = torch.autograd.grad(objective(actual), matrix)
    torch.testing.assert_close(result, reference, rtol=2e-6 if dtype == torch.float32 else 2e-14,
                               atol=1e-38 if dtype == torch.float32 else 1e-45)


def test_column_norm_gradcheck():
    matrix = torch.randn((2, 7, 5), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: column_norms(value, chunk=3), (matrix,))
