"""Full-basis independent-center batch algebra and gradient isolation checks."""
import math
import unittest

import pytest
torch = pytest.importorskip("torch")

from coil_eval._autograd.batch_front import (
    BatchTensorProductPsiOperator,batch_four_step_pcgls,differentiable_batch_axis,
)
from coil_eval._autograd.front import TensorProductPsiOperator,four_step_pcgls


def axes():
    phi=torch.arange(24,dtype=torch.float64)*2*math.pi/(4*24)
    base=torch.stack((1+.02*(4*phi).cos(),.01*(4*phi).sin(),-.08*(4*phi).sin(),.04*(4*phi).cos()),-1)
    return torch.stack((base,base+base.new_tensor([.1,.02,0.,0.])))


def field(points,gradient=False):
    x,y,z=points.unbind(-1)
    return torch.stack((1+.2*x+.1*z,.3+.6*y+.05*x*x,.7+.1*z*z),-1)


class BatchFrontTests(unittest.TestCase):
    def test_full_basis_per_query_radius_and_exclusion_matches_single(self):
        a=[.005,.15]
        samples=axes().requires_grad_()
        op=BatchTensorProductPsiOperator(field,samples,a,radial_grid=8,vertical_grid=8,phi_grid=9,dtype=torch.float64)
        self.assertEqual(op.coefficient_count,1574)
        self.assertNotEqual(int(op.valid_rows[0]),int(op.valid_rows[1]))
        torch.manual_seed(919)
        coeff=torch.randn(2,1574,dtype=torch.float64)
        rows=torch.randn_like(op.rhs)
        actual=op.apply(coeff)
        for query in range(2):
            single=TensorProductPsiOperator(field,samples[query],a=a[query],radial_grid=8,vertical_grid=8,phi_grid=9,dtype=torch.float64)
            selected=op.row_mask[query]
            torch.testing.assert_close(actual[query,selected],single.apply(coeff[query]),rtol=3e-12,atol=3e-9)
            torch.testing.assert_close(op.transpose(rows)[query],single.transpose(rows[query,selected]),rtol=3e-12,atol=3e-9)
            self.assertEqual(torch.count_nonzero(actual[query,~selected]).item(),0)
        gradient,=torch.autograd.grad(actual[0].square().mean(),samples)
        self.assertEqual(torch.count_nonzero(gradient[1]).item(),0)
        self.assertGreater(float(gradient[0].norm()),0.)

    def test_independent_four_step_solvers_and_parameter_gradients(self):
        torch.manual_seed(920)
        B,N,C=3,17,5
        A0=torch.randn(B,N,C,dtype=torch.float64)
        delta=torch.randn_like(A0)
        rhs=torch.randn(B,N,dtype=torch.float64)
        scale=A0.square().sum(1).sqrt()
        upper=torch.linalg.qr(torch.cat((A0/scale[:,None,:],math.sqrt(1e-6)*torch.eye(C,dtype=torch.float64)[None].expand(B,-1,-1)),1),mode='r').R
        c0=torch.linalg.lstsq(A0,rhs[...,None]).solution[...,0]
        t=torch.tensor([.01,-.03,.02],dtype=torch.float64,requires_grad=True)
        class Batch:
            def __init__(self,A):self.A,self.rhs,self.valid_rows=A,rhs[...,None],torch.full((B,),N)
            def apply(self,c):return (self.A@c[...,None])
            def transpose(self,r):return (self.A.transpose(-1,-2)@r)[...,0]
            def column_norms(self):return self.A.square().sum(1).sqrt()
        class Single:
            def __init__(self,A,b):self.A,self.rhs=A,b
            def apply(self,c):return self.A@c
            def transpose(self,r):return self.A.T@r
            def column_norms(self):return self.A.square().sum(0).sqrt()
        coefficients,rms=batch_four_step_pcgls(Batch(A0+t[:,None,None]*delta),c0,scale,upper)
        actual_grad,=torch.autograd.grad(coefficients.square().sum(),t)
        tr=t.detach().requires_grad_()
        singles=[four_step_pcgls(Single(A0[i]+tr[i]*delta[i],rhs[i]),c0[i],scale[i],upper[i]) for i in range(B)]
        expected=torch.stack([v[0] for v in singles]);expected_rms=torch.stack([v[1] for v in singles])
        expected_grad,=torch.autograd.grad(expected.square().sum(),tr)
        torch.testing.assert_close(coefficients,expected,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(rms,expected_rms,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-10,atol=1e-11)

    def test_batch_axis_implicit_response_and_query_isolation(self):
        # An exact affine Poincare map exposes incorrect query-axis aggregation.
        x=torch.tensor([[.3,.1],[.7,-.2]],dtype=torch.float64,requires_grad=True)
        A=torch.tensor([[[.2,.1],[-.1,.3]],[[.1,-.2],[.2,.2]]],dtype=torch.float64)
        C=torch.tensor([[[.3,.1],[.2,.4]],[[.1,.3],[.4,-.2]]],dtype=torch.float64)
        root=torch.linalg.solve(torch.eye(2,dtype=torch.float64)[None]-A,(C@x[...,None]))[...,0]
        class Field:
            def trace(self,y,phi0,phi1,steps,return_samples=False):
                end=(A@y[:,0,:,None]+C@x[...,None])[...,0][:,None,:]
                if return_samples:
                    t=torch.linspace(0.,1.,steps+1,dtype=y.dtype)[:,None,None,None]
                    return y[None]+t*(end-y)[None]
                return end
            def __call__(self,points):
                return torch.stack((-points[...,1],points[...,0],points[...,2]*0),-1)
        samples,diag=differentiable_batch_axis(Field(),root.detach(),trace_steps=8,sample_count=8)
        actual,=torch.autograd.grad(samples[0,0,0],x)
        reference=torch.linalg.solve((torch.eye(2,dtype=x.dtype)-A[0]).T,x.new_tensor([1.,0.]))
        torch.testing.assert_close(actual[0],C[0].T@reference)
        self.assertEqual(torch.count_nonzero(actual[1]).item(),0)
        self.assertTrue(bool(diag['valid'].all()))


if __name__=='__main__':unittest.main()
