"""Operator and dependency checks, not production GPU acceptance evidence."""
import math
import unittest
from dataclasses import replace

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from torch.autograd import gradcheck

from coil_eval._autograd.downstream import (
    DownstreamConfig, _AlphaDesign, _design_chunk, alpha_fit, clebsch_modes,
    flux_calibration, frozen_pcgls, qs_metric, surface_quality,
    volume_points,
)


class CircleGeometry:
    def __init__(self, R):
        self.R = R

    def axis_eval(self, phi):
        zero = torch.zeros_like(phi)
        return self.R + zero, zero, zero, zero

    def rays(self, phi, theta):
        return torch.stack((torch.zeros_like(phi), torch.zeros_like(phi),
                            torch.ones_like(phi)), -1)

    def solve_rays(self, poly, level):
        return torch.sqrt(level / poly[..., 2])

    def psi_eval(self, R, Z, phi):
        return (R - self.R).square() + Z.square(), 2 * (R - self.R), 2 * Z, torch.zeros_like(R)


def toroidal_field(xyz, gradient=False):
    x, y, z = xyz.unbind(-1)
    r2 = x.square() + y.square()
    zero = torch.zeros_like(z)
    B = torch.stack((-y / r2, x / r2, zero), -1)
    if not gradient:
        return B
    J = torch.stack((2 * x * y / r2.square(), -1 / r2 + 2 * y.square() / r2.square(), zero,
                     1 / r2 - 2 * x.square() / r2.square(), -2 * x * y / r2.square(), zero,
                     zero, zero, zero), -1).reshape(-1, 3, 3)
    return B, J


def toroidal_potential(xyz):
    zero = torch.zeros_like(xyz[:, 0])
    return torch.stack((zero, zero, -.5 * torch.log(xyz[:, :2].square().sum(-1))), -1)


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.dtype = torch.float64

    def test_production_basis_order_and_size(self):
        modes = clebsch_modes(12, 12, 12)
        self.assertEqual(len(modes[0]), 2268)
        self.assertEqual(modes[0][:4], (0, 0, 0, 0))
        self.assertEqual(modes[1][:4], (1, 1, 1, 1))
        self.assertEqual(modes[2][:4], (0, 1, 0, 1))

    def test_design_recompute_backward_all_features(self):
        c = replace(DownstreamConfig(), alpha_radial_order=2,
                    alpha_poloidal_order=2, alpha_toroidal_order=1, iota_degree=1)
        modes = tuple(torch.tensor(v, dtype=self.dtype) for v in clebsch_modes(2, 2, 1))
        n, nc = 7, len(modes[0]) + 2
        values = [(torch.rand(n, dtype=self.dtype) + .2).requires_grad_() for _ in range(6)]
        scale = torch.rand(nc, dtype=self.dtype) + .5
        actual = _AlphaDesign.apply(*values, scale, modes, c.nfp, c.iota_degree, 5)
        expected = _design_chunk(*values, scale, modes, c.nfp, 0, nc, c.iota_degree)
        torch.testing.assert_close(actual, expected)
        adj = torch.randn_like(actual)
        ga = torch.autograd.grad(actual, values, adj)
        ge = torch.autograd.grad(expected, values, adj)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)
        self.assertTrue(gradcheck(lambda *v: _AlphaDesign.apply(*v, scale, modes, 4, 1, 5),
                                  tuple(values), fast_mode=True))

    def test_four_step_pcgls_offcenter_and_frozen_cache(self):
        A0 = torch.randn(12, 6, dtype=self.dtype)
        b0 = torch.randn(12, dtype=self.dtype)
        ridge = 1e-4
        H = A0.T @ A0 + ridge * torch.eye(6, dtype=self.dtype)
        cache = {'upper': torch.linalg.cholesky(H).T.requires_grad_(),
                 'scaled_solution': torch.linalg.solve(H, A0.T @ b0).requires_grad_()}
        A = (A0 + .05 * torch.randn_like(A0)).requires_grad_()
        b = (b0 + .05 * torch.randn_like(b0)).requires_grad_()
        self.assertTrue(gradcheck(lambda a, bb: frozen_pcgls(a, bb, cache, ridge)[0],
                                  (A, b), fast_mode=True, atol=1e-6))
        frozen_pcgls(A, b, cache, ridge)[0].sum().backward()
        self.assertIsNone(cache['upper'].grad)
        self.assertIsNone(cache['scaled_solution'].grad)

    def test_flux_stokes_area_volume_and_axis_motion(self):
        c = replace(DownstreamConfig(), flux_phi_count=3, flux_theta_count=128,
                    flux_level_count=7, flux_radial_quadrature=12)
        R = torch.tensor(1.2, dtype=self.dtype, requires_grad=True)
        level = torch.tensor(.025, dtype=self.dtype, requires_grad=True)
        geom = CircleGeometry(R)
        stokes = flux_calibration(geom, level, toroidal_field, toroidal_potential, c)
        area = flux_calibration(geom, level, toroidal_field, None, replace(c, stokes_flux=False))
        torch.testing.assert_close(stokes['sections'], area['sections'], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(stokes['surface_volume'], 2 * math.pi ** 2 * R * level)
        got = torch.autograd.grad(stokes['surface_volume'], (R, level), retain_graph=True)
        torch.testing.assert_close(got[0], 2 * math.pi ** 2 * level)
        torch.testing.assert_close(got[1], 2 * math.pi ** 2 * R)
        gs = torch.autograd.grad(stokes['edge_flux'], (R, level), retain_graph=True)
        ga = torch.autograd.grad(area['edge_flux'], (R, level))
        for a, b in zip(gs, ga):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)

    def test_qs_all_moving_inputs(self):
        c = DownstreamConfig()
        n = 5
        B = (torch.randn(n, 3, dtype=self.dtype) + 2).requires_grad_()
        J = torch.randn(n, 3, 3, dtype=self.dtype, requires_grad=True)
        gs = torch.randn(n, 3, dtype=self.dtype, requires_grad=True)
        fd = torch.rand(n, dtype=self.dtype, requires_grad=True)
        rho = torch.rand(n, dtype=self.dtype, requires_grad=True)
        weight = (torch.rand(n, dtype=self.dtype) + .2).requires_grad_()
        iota = torch.randn(4, dtype=self.dtype, requires_grad=True)
        G = torch.tensor(.3, dtype=self.dtype, requires_grad=True)
        def fn(b, j, grad_s, fp, r, w, i, g):
            return qs_metric(dict(grad_s=grad_s, flux_derivative=fp, rho=r, volume_weight=w), b, j, i, g, c)['raw_qs']
        self.assertTrue(gradcheck(fn, (B, J, gs, fd, rho, weight, iota, G), fast_mode=True))
        for v in torch.autograd.grad(fn(B, J, gs, fd, rho, weight, iota, G), (B, J, gs, fd, rho, weight, iota, G)):
            self.assertGreater(v.norm().item(), 0)

    def test_real_front_interface_full_basis_moving_volume(self):
        from coil_eval._autograd.front import PsiGeometry, native_modes
        modes = native_modes()
        z_index = modes.index((0, 2, 0, 0))
        cfg = replace(DownstreamConfig(), flux_phi_count=2, flux_theta_count=16,
                      flux_level_count=5, volume_point_count=64,
                      volume_phi_count=4, volume_theta_count=8)
        def fn(major, z_coefficient, level):
            zeros = torch.zeros(12, dtype=self.dtype)
            axis = torch.stack((major + zeros, zeros, zeros, zeros), -1)
            mask = torch.zeros(len(modes), dtype=self.dtype)
            mask[z_index] = 1
            geom = PsiGeometry(axis, mask * z_coefficient, a=.15)
            flux = flux_calibration(geom, level, toroidal_field, toroidal_potential, cfg)
            points = volume_points(geom, level, flux, cfg)
            # This includes all volume coordinate fields that used to be frozen
            # in G3; arbitrary nonzero cotangents expose missing dependencies.
            scalar = flux['surface_volume'] + .03 * points['xyz'].square().mean()
            for name in ('grad_s', 'grad_theta', 'grad_phi', 'rho', 'volume_weight', 'flux_derivative'):
                scalar = scalar + .001 * points[name].square().mean()
            return scalar
        inputs = tuple(torch.tensor(v, dtype=self.dtype, requires_grad=True) for v in (1.2, .92, .2))
        self.assertTrue(gradcheck(fn, inputs, fast_mode=True, atol=3e-6, rtol=1e-4))
        self.assertTrue(all(g.abs() > 0 for g in torch.autograd.grad(fn(*inputs), inputs)))

    def test_quality_tracks_trace_and_mean_radius(self):
        cfg = replace(DownstreamConfig(), surface_theta_count=20)
        R = torch.tensor(1.2, dtype=self.dtype, requires_grad=True)
        level = torch.tensor(.03, dtype=self.dtype, requires_grad=True)
        shift = torch.tensor(.001, dtype=self.dtype, requires_grad=True)
        def fn(lvl, displacement):
            def trace(rz, *args):
                return rz + torch.stack((torch.ones_like(rz[:, 0]) * displacement,
                                          torch.ones_like(rz[:, 0]) * displacement * .23), -1)
            return surface_quality(CircleGeometry(R), lvl, trace, cfg)['quality_penalty']
        self.assertTrue(gradcheck(fn, (level, shift), fast_mode=True))
        g = torch.autograd.grad(fn(level, shift), (level, shift))
        self.assertTrue(all(v.abs().item() > 0 for v in g))

    def test_quality_promotes_native_fp32_coefficients_before_root_construction(self):
        from coil_eval._autograd.front import PsiGeometry, native_modes
        modes = native_modes()
        zeros = torch.zeros(24, dtype=torch.float64)
        axis = torch.stack((zeros+1.2, zeros, zeros, zeros), -1)
        coefficients = torch.zeros(len(modes), dtype=torch.float32)
        coefficients[modes.index((0, 2, 0, 0))] = .91
        geometry = PsiGeometry(axis, coefficients, a=.15)
        level = torch.tensor(.173942831237, dtype=torch.float64)
        config = replace(DownstreamConfig(), surface_theta_count=128)
        result = surface_quality(geometry, level, lambda rz, *args: rz, config)
        # An exactly closed trace must not acquire drift from an FP32 root.
        self.assertLess(result['target_quality_error'].item(), 1e-12)


    def test_engineering_native_layout_and_normalized_chain(self):
        from unittest.mock import patch
        from coil_eval._autograd.engineering import _EngineeringMetrics
        import coil_eval._autograd.engineering as evaluator
        native_jac = np.arange(6 * 3 * 4 * 9, dtype=float).reshape(6, 3, 4, 9) / 100
        x = torch.randn(4, 28, dtype=self.dtype, requires_grad=True)
        scale = torch.arange(1, 113, dtype=self.dtype).reshape(4, 28) / 50
        def fake_native(path, xx, yy, zz, currents, nfp):
            geometry = np.stack((xx, yy, zz))
            values = np.einsum('macj,acj->m', native_jac, geometry)
            names = ('length_mean', 'curvature_p95', 'curvature_max',
                     'min_intercoil_distance', 'min_axis_distance', 'high_mode_fraction')
            return dict(values=dict(zip(names, values)), derivatives=native_jac)
        with patch.object(evaluator.native, 'coil_metric_gradient_native', fake_native):
            def fn(z):
                return _EngineeringMetrics.apply(z * scale + .1, 'unused', 4)
            self.assertTrue(gradcheck(fn, (x,), fast_mode=True))
            g = torch.autograd.grad(fn(x).sum(), x)[0]
            self.assertTrue(torch.equal(g[:, 27], torch.zeros(4, dtype=self.dtype)))





if __name__ == '__main__':
    unittest.main()
