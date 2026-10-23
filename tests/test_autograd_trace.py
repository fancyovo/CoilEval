"""CPU tests of the fused RK4 RHS derivatives; GPU tape checks run separately."""
import unittest

import pytest
torch = pytest.importorskip("torch")

from coil_eval._autograd.trace import local_rhs_derivatives


class TraceLocalTests(unittest.TestCase):
    def test_rhs_total_state_and_field_derivatives(self):
        torch.manual_seed(621)
        for toroidal_field in (1., 0., 5e-15, -5e-15, 2e-14, -2e-14):
            # phi=0 makes Bphi=By, permitting exact controlled floor branches.
            rz=torch.tensor([1.3,.4],dtype=torch.float64,requires_grad=True)
            q=torch.stack((rz[0],rz[0]*0,rz[1]))
            base=torch.tensor([.2,toroidal_field,-.1],dtype=torch.float64,requires_grad=True)
            G=torch.randn(3,3,dtype=torch.float64)*.01
            B=base+G@(q-q.detach())
            rhs,J,M=local_rhs_derivatives(rz,torch.tensor(0.,dtype=rz.dtype),B,G)
            expected_J=torch.stack([torch.autograd.grad(rhs[i],rz,retain_graph=True)[0] for i in range(2)])
            expected_M=torch.stack([torch.autograd.grad(rhs[i],base,retain_graph=True)[0] for i in range(2)])
            torch.testing.assert_close(J,expected_J,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(M,expected_M,rtol=1e-12,atol=1e-12)

    def test_batched_angles_and_moving_point_jacobian(self):
        rz=torch.tensor([[1.1,.1],[1.4,-.1]],dtype=torch.float64,requires_grad=True)
        phi=torch.tensor([.3,-.2],dtype=rz.dtype)
        q=torch.stack((rz[:,0]*phi.cos(),rz[:,0]*phi.sin(),rz[:,1]),-1)
        matrix=torch.tensor([[.2,.3,-.1],[-.2,.4,.3],[.3,-.1,.4]],dtype=rz.dtype)
        B=q@matrix.T+q.new_tensor([.1,2.,.3])
        rhs,J,_=local_rhs_derivatives(rz,phi,B,matrix.expand(2,3,3))
        expected=torch.stack([torch.autograd.grad(rhs[:,i].sum(),rz,retain_graph=True)[0] for i in range(2)],1)
        torch.testing.assert_close(J,expected,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
