"""Front-end mathematical checks; native GPU acceptance is a separate gate."""
import math
import unittest

import pytest
torch = pytest.importorskip("torch")

from coil_eval._autograd.front import (
    PsiGeometry, TensorProductPsiOperator, four_step_pcgls, implicit_axis_start,
    native_modes, periodic_hermite, predict_target_level,
)


DT = torch.float64


def axis(count=24):
    phi = torch.arange(count, dtype=DT)*2*math.pi/(4*count)
    return torch.stack((1+.02*(4*phi).cos(), .01*(4*phi).sin(),
                        -.08*(4*phi).sin(), .04*(4*phi).cos()), -1)


def field(points, gradient=False):
    x,y,z = points.unbind(-1)
    return torch.stack((1+.2*x+.1*z, .3+.6*y+.05*x*x, .7+.1*z*z), -1)


class FrontTests(unittest.TestCase):
    def test_full_basis_tensor_product_matches_direct_matrix_and_transpose(self):
        # Full 1574 columns; fewer rows keep this CPU algebra check small.
        op = TensorProductPsiOperator(field, axis(), a=.15, radial_grid=7,
                                      vertical_grid=7, phi_grid=9, dtype=DT)
        self.assertEqual(op.coefficient_count, 1574)
        X = op.X[:,None].expand(-1,9).reshape(-1)
        Y = op.Y[:,None].expand(-1,9).reshape(-1)
        phi = op.phi[None,:].expand(op.X.numel(),-1).reshape(-1)
        cr,cz,cp = [v.reshape(-1) for v in op.features]
        columns = []
        for a,b,m,kind in native_modes():
            phase = 4*m*phi
            trig = phase.cos() if kind == 0 else phase.sin()
            trig_p = (-phase.sin() if kind == 0 else phase.cos())*(4*m)
            columns.append((cr*a*X**max(a-1,0)*Y**b + cz*b*X**a*Y**max(b-1,0))*trig/.15
                           +cp*X**a*Y**b*trig_p)
        A = torch.stack(columns, -1)
        generator = torch.Generator().manual_seed(19)
        c = torch.randn(1574, generator=generator, dtype=DT)
        r = torch.randn(A.shape[0], generator=generator, dtype=DT)
        torch.testing.assert_close(op.apply(c).reshape(-1), A@c, rtol=2e-12, atol=2e-10)
        torch.testing.assert_close(op.transpose(r.reshape(op.rhs.shape)), A.T@r, rtol=2e-12, atol=2e-10)

    def test_moving_grid_and_axis_slope_reach_loss_gradient(self):
        displacement = torch.tensor(.003, dtype=DT, requires_grad=True)
        direction = torch.zeros_like(axis())
        direction[:,0] = 1
        direction[:,2] = torch.linspace(-.2,.2,direction.shape[0])
        c = torch.linspace(-.01,.01,1574,dtype=DT)
        def value(s):
            op = TensorProductPsiOperator(field, axis()+s*direction, a=.15,
                radial_grid=6, vertical_grid=6, phi_grid=7, dtype=DT)
            return (op.apply(c)-op.rhs).square().mean()
        actual, = torch.autograd.grad(value(displacement), displacement)
        h = 1e-6
        expected = (value(displacement.detach()+h)-value(displacement.detach()-h))/(2*h)
        torch.testing.assert_close(actual,expected,rtol=2e-7,atol=1e-7)
        self.assertGreater(abs(float(actual)),1e-4)

    def test_periodic_axis_response_is_not_frozen(self):
        x = torch.tensor([.7,-.2],dtype=DT,requires_grad=True)
        A = torch.tensor([[.2,.3],[-.1,.4]],dtype=DT)
        C = torch.tensor([[.3,-.2],[.4,.1]],dtype=DT)
        true_root = torch.linalg.solve(torch.eye(2,dtype=DT)-A,C@x)
        def trace(rz,phi0,phi1,steps,return_samples=False):
            return rz@A.T+(C@x)[None]
        root,diag = implicit_axis_start(trace,true_root.detach(),4,240)
        weights = x.new_tensor([2.,-.7])
        grad, = torch.autograd.grad(root@weights,x)
        expected = C.T@torch.linalg.solve((torch.eye(2,dtype=DT)-A).T,weights)
        torch.testing.assert_close(grad,expected)
        torch.testing.assert_close(root,true_root)
        self.assertLess(float(diag['root_residual']),1e-14)

    def test_periodic_axis_rejects_singular_root(self):
        with self.assertRaisesRegex(ValueError,'monodromy'):
            implicit_axis_start(lambda rz,*args,**kwargs: rz,torch.ones(2,dtype=DT),4,240)

    def test_geometry_psi_gradient_and_physical_ray_convention(self):
        c = torch.zeros(1574,dtype=DT)
        modes = native_modes()
        c[modes.index((0,2,0,0))] = 1.3
        c[modes.index((3,0,1,0))] = .04
        geometry = PsiGeometry(axis(),c,a=.15)
        phi = torch.linspace(.01,.4,13,dtype=DT,requires_grad=True)
        theta = torch.linspace(.1,5.,13,dtype=DT)
        r = torch.full((13,),.04,dtype=DT)
        aR,aZ,_,_ = geometry.axis_eval(phi)
        R = (aR+r*theta.cos()).detach().requires_grad_()
        Z = (aZ+r*theta.sin()).detach().requires_grad_()
        value,gR,gZ,gp = geometry.psi_eval(R,Z,phi)
        calculated = torch.autograd.grad(value.sum(),(R,Z,phi))
        for explicit,autodiff in zip((gR,gZ,gp),calculated):
            torch.testing.assert_close(explicit,autodiff,rtol=1e-10,atol=1e-10)
        polynomial = geometry.rays(phi,theta)
        rayvalue = (polynomial*r[:,None]**torch.arange(11)).sum(-1)
        torch.testing.assert_close(rayvalue,value,rtol=1e-12,atol=1e-12)
        recovered = geometry.solve_rays(polynomial,value.detach())
        torch.testing.assert_close(recovered,r,rtol=1e-9,atol=1e-11)

    def test_implicit_radial_root_level_and_coefficient_gradients(self):
        c = torch.zeros(1574,dtype=DT)
        c[native_modes().index((0,2,0,0))] = 1.
        geometry = PsiGeometry(axis(),c,a=.15)
        p = torch.zeros((5,11),dtype=DT)
        p[:,2] = torch.linspace(20.,40.,5,dtype=DT)
        p.requires_grad_()
        level = torch.tensor(.08,dtype=DT,requires_grad=True)
        roots = geometry.solve_rays(p,level)
        dp,dl = torch.autograd.grad(roots.sum(),(p,level))
        torch.testing.assert_close(dp[:,2],-.5*torch.sqrt(level)/p[:,2]**1.5)
        torch.testing.assert_close(dl,(.5/torch.sqrt(level*p[:,2])).sum())

    def test_target_level_includes_axis_and_psi_response(self):
        coefficients = torch.zeros(1574,dtype=DT,requires_grad=True)
        samples = axis().requires_grad_()
        geom = PsiGeometry(samples,coefficients,a=.15)
        phi = torch.arange(8,dtype=DT)*2*math.pi/(4*8)
        center = dict(psi_coefficients=torch.zeros(1574,dtype=DT),
            target_adjoint_weights=torch.linspace(.1,.2,1574,dtype=DT),
            target_axis_weights=torch.linspace(.01,.02,8,dtype=DT),
            target_phi=phi,target_axis_R=geom.axis_eval(phi)[0].detach(),
            target_level=.2,target_dVds=.3,target_adjoint_residual=2e-8)
        level = predict_target_level(geom,center)
        dc,da = torch.autograd.grad(level,(coefficients,samples))
        torch.testing.assert_close(dc,-center['target_adjoint_weights']/.3)
        torch.testing.assert_close(da[:,0].sum(),-center['target_axis_weights'].sum()/.3)
        self.assertGreater(float(torch.linalg.vector_norm(da)),0.)

    def test_four_step_solver_vjp_off_center_and_frozen_constants(self):
        torch.manual_seed(71)
        A0 = torch.randn(15,5,dtype=DT)
        direction = torch.randn_like(A0)
        rhs = torch.randn(15,dtype=DT)
        scale = A0.square().sum(0).sqrt().requires_grad_()
        upper = torch.linalg.qr(torch.cat((A0/scale.detach(),math.sqrt(1e-6)*torch.eye(5,dtype=DT))),mode='r').R.detach().requires_grad_()
        c0 = torch.linalg.lstsq(A0,rhs).solution.detach().requires_grad_()
        parameter = torch.tensor(.03,dtype=DT,requires_grad=True)
        class Operator:
            def __init__(self,A): self.A,self.rhs=A,rhs
            def apply(self,c): return self.A@c
            def transpose(self,r): return self.A.T@r
        def value(t):
            result,_ = four_step_pcgls(Operator(A0+t*direction),c0,scale,upper)
            return result.square().sum()
        grads = torch.autograd.grad(value(parameter),(parameter,c0,scale,upper),allow_unused=True)
        self.assertEqual(grads[1:],(None,None,None))
        h=1e-6
        expected=(value(parameter.detach()+h)-value(parameter.detach()-h))/(2*h)
        torch.testing.assert_close(grads[0],expected,rtol=1e-7,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
