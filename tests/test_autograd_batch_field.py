"""CPU contracts and Fourier query isolation; GPU batch parity runs separately."""
import unittest
import pytest
torch = pytest.importorskip("torch")

from coil_eval._autograd.batch_field import (
    BatchSegmentField,validate_batch_geometry,broadcast_trace_bounds,coil_segments_batch,
)
from coil_eval._autograd.field import coil_segments


class BatchFieldContractTests(unittest.TestCase):
    def test_explicit_query_axis_and_count_required(self):
        p=torch.zeros(3,37,3,dtype=torch.float64)
        w=torch.ones_like(p)
        self.assertEqual(validate_batch_geometry(p,w,torch.zeros(3,11,3)),(3,37,11))
        for bad in (torch.zeros(11,3),torch.zeros(1,11,3),torch.zeros(3,11,2)):
            with self.assertRaisesRegex(ValueError,'query'):
                validate_batch_geometry(p,w,bad)
        with self.assertRaisesRegex(ValueError,'identical'):
            validate_batch_geometry(p,w[:1])
        with self.assertRaisesRegex(ValueError,'CUDA'):
            validate_batch_geometry(p,w,require_cuda=True)

    def test_trace_bounds_unambiguous_query_and_line_broadcast(self):
        starts=torch.zeros((3,5,2),dtype=torch.float64)
        scalar=broadcast_trace_bounds(.2,starts)
        query=broadcast_trace_bounds(torch.tensor([[.1],[.2],[.3]]),starts)
        line=broadcast_trace_bounds(torch.arange(5.)[None],starts)
        self.assertEqual(scalar.shape,(3,5))
        torch.testing.assert_close(query[:,0],torch.tensor([.1,.2,.3],dtype=torch.float64),rtol=1e-7,atol=1e-8)
        torch.testing.assert_close(line[2],torch.arange(5,dtype=torch.float64))
        for ambiguous in (torch.arange(3.),torch.arange(5.)):
            with self.assertRaisesRegex(ValueError,'2-D'):
                broadcast_trace_bounds(ambiguous,starts)

    def test_full_quadrature_matches_independent_configurations(self):
        torch.manual_seed(711)
        tokens=torch.randn((3,4,28),dtype=torch.float64)
        tokens[...,-1]*=1e5
        p,w=coil_segments_batch(tokens,nfp=4,count=256)
        self.assertEqual(p.shape,(3,8192,3))
        for query in range(3):
            expected_p,expected_w=coil_segments(tokens[query],nfp=4,count=256)
            torch.testing.assert_close(p[query],expected_p,rtol=1e-13,atol=1e-13)
            torch.testing.assert_close(w[query],expected_w,rtol=1e-13,atol=1e-8)

    def test_quadrature_gradient_cannot_leak_to_other_query(self):
        torch.manual_seed(712)
        tokens=torch.randn((3,4,28),dtype=torch.float64,requires_grad=True)
        p,w=coil_segments_batch(tokens,nfp=4,count=256)
        grad,=torch.autograd.grad(p[1].square().sum()+w[1].square().sum(),tokens)
        self.assertEqual(torch.count_nonzero(grad[0]).item(),0)
        self.assertEqual(torch.count_nonzero(grad[2]).item(),0)
        self.assertGreater(float(torch.linalg.vector_norm(grad[1])),0.)


@pytest.mark.parametrize('coil_count,nfp',[(1,1),(2,3),(5,2)])
def test_generic_symmetry_segments_and_gradients_match_scalar(coil_count,nfp):
    torch.manual_seed(713)
    tokens=torch.randn((2,coil_count,10),dtype=torch.float64,requires_grad=True)
    p,w=coil_segments_batch(tokens,nfp=nfp,count=16)
    assert p.shape==w.shape==(2,2*nfp*coil_count*16,3)
    independent=tokens[1].detach().requires_grad_()
    expected_p,expected_w=coil_segments(independent,nfp=nfp,count=16)
    torch.testing.assert_close(p[1],expected_p,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(w[1],expected_w,rtol=1e-13,atol=1e-13)
    actual,=torch.autograd.grad(p[1].square().sum()+w[1].square().sum(),tokens)
    expected,=torch.autograd.grad(expected_p.square().sum()+expected_w.square().sum(),independent)
    torch.testing.assert_close(actual[1],expected,rtol=1e-12,atol=1e-12)
    assert torch.count_nonzero(actual[0]).item()==0


if __name__=='__main__':unittest.main()
