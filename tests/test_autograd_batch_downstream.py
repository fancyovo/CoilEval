"""Independent-query parity and cross-query isolation of complete downstream."""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from coil_eval._autograd.batch_downstream import (
    _BatchAlphaDesign, batch_frozen_pcgls, batch_downstream)
from coil_eval._autograd.downstream import (
    _AlphaDesign, DownstreamConfig, clebsch_modes, downstream, frozen_pcgls)
from test_autograd_downstream import CircleGeometry


class ToyBatchField:
    def __init__(self, amplitude):
        self.amplitude = amplitude
        self.calls = dict(field=0, potential=0, trace=0)

    def __call__(self, xyz, gradient=False):
        self.calls['field'] += 1
        x, y, z = xyz.unbind(-1)
        r2 = x.square() + y.square()
        a = self.amplitude[:, None]
        fac = 1 + a * z
        zero = torch.zeros_like(x)
        B = torch.stack((-y / r2 * fac, x / r2 * fac, a * x * .03), -1)
        if not gradient:
            return B
        G = torch.stack((2*x*y/r2.square()*fac, (-1/r2+2*y.square()/r2.square())*fac, -y/r2*a,
                         (1/r2-2*x.square()/r2.square())*fac, -2*x*y/r2.square()*fac, x/r2*a,
                         a.expand_as(x)*.03, zero, zero), -1).reshape(*x.shape, 3, 3)
        return B, G

    def potential(self, xyz):
        self.calls['potential'] += 1
        x, y, z = xyz.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack((zero, zero, -.5 * (1+self.amplitude[:, None]*.01)
                            * torch.log(x.square()+y.square())), -1)

    def trace(self, initial, *args):
        self.calls['trace'] += 1
        dx = self.amplitude[:, None].expand_as(initial[..., 0]) * .001
        return initial + torch.stack((dx, dx*.23), -1)

    def one(self, index):
        field = ToyBatchField(self.amplitude[index:index+1])
        def fn(xyz, gradient=False):
            value = field(xyz[None], gradient)
            return tuple(t[0] for t in value) if gradient else value[0]
        fn.potential = lambda xyz: field.potential(xyz[None])[0]
        fn.trace = lambda initial, *args: field.trace(initial[None], *args)[0]
        return fn


class BatchDownstreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.dtype = torch.float64

    def test_design_all_feature_vjps_match_independent_queries(self):
        modes = tuple(torch.tensor(v, dtype=self.dtype) for v in clebsch_modes(2, 2, 1))
        columns = len(modes[0]) + 2
        features = [(torch.rand(3, 8, dtype=self.dtype)+.2).requires_grad_() for _ in range(6)]
        scale = torch.rand(3, columns, dtype=self.dtype)+.4
        batch = _BatchAlphaDesign.apply(*features, scale, modes, 4, 1, 5)
        singles = torch.stack([_AlphaDesign.apply(*[v[i] for v in features], scale[i], modes, 4, 1, 5)
                               for i in range(3)])
        torch.testing.assert_close(batch, singles, atol=1e-13, rtol=1e-13)
        adjoint = torch.randn_like(batch)
        gb = torch.autograd.grad(batch, features, adjoint)
        gs = torch.autograd.grad(singles, features, adjoint)
        for a, b in zip(gb, gs):
            torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)

    def test_four_step_batched_recurrence_matches_each_query_vjp(self):
        A = torch.randn(3, 16, 7, dtype=self.dtype, requires_grad=True)
        b = torch.randn(3, 16, dtype=self.dtype, requires_grad=True)
        H = torch.randn(3, 7, 7, dtype=self.dtype)
        upper = torch.linalg.cholesky(H.transpose(1, 2)@H+torch.eye(7, dtype=self.dtype)).transpose(1, 2)
        cache = dict(upper=upper, scaled_solution=torch.randn(3, 7, dtype=self.dtype))
        actual = batch_frozen_pcgls(A, b, cache, 1e-4)
        singles = [frozen_pcgls(A[i], b[i], {k:v[i] for k,v in cache.items()}, 1e-4) for i in range(3)]
        expected = tuple(torch.stack([v[j] for v in singles]) for j in range(2))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-12)
        adjoint = torch.randn_like(actual[0])
        ga = torch.autograd.grad(actual[0], (A, b), adjoint)
        ge = torch.autograd.grad(expected[0], (A, b), adjoint)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=3e-11, rtol=3e-11)

    def test_complete_downstream_batch_value_gradient_and_isolation(self):
        config = replace(DownstreamConfig(), axis_sample_count=12, surface_theta_count=20,
            flux_phi_count=3, flux_theta_count=24, flux_level_count=5, flux_polynomial_degree=3,
            volume_point_count=50, volume_phi_count=3, volume_theta_count=5, alpha_fit_point_count=40,
            alpha_radial_order=2, alpha_poloidal_order=2, alpha_toroidal_order=1, iota_degree=1,
            radial_bin_count=4, alpha_chunk_columns=5)
        configs = [config, replace(config, volume_rho_min=.13, target_quality_threshold=.008)]
        values = torch.tensor([[1.2, .03, .8], [1.3, .04, 1.1]], dtype=self.dtype, requires_grad=True)
        front = SimpleNamespace(levels=values[:, 1], geometry=lambda i: CircleGeometry(values[i, 0]))
        field = ToyBatchField(values[:, 2])
        columns = len(clebsch_modes(2, 2, 1)[0])+2
        cache = dict(upper=torch.eye(columns, dtype=self.dtype)[None].expand(2, -1, -1),
                     scale=torch.ones(2, columns, dtype=self.dtype),
                     scaled_solution=torch.zeros(2, columns, dtype=self.dtype))
        actual = batch_downstream(front, cache, field, configs, return_arrays=True)
        self.assertEqual(field.calls, dict(field=1, potential=1, trace=1))
        expected = []
        for i in range(2):
            one = field.one(i)
            expected.append(downstream(front.geometry(i), front.levels[i], {k:v[i] for k,v in cache.items()},
                                       one, one.trace, one.potential, configs[i], return_arrays=True))
        keys = ('raw_qs', 'raw_qs_outer', 'iota_boundary', 'iota_min', 'iota_max', 'quality_penalty', 'target_quality_error',
                'surface_volume', 'flux_edge', 'major_radius', 'flux_coefficients')
        for key in keys:
            torch.testing.assert_close(actual[key], torch.stack([e[key] for e in expected]),
                                       atol=2e-10, rtol=2e-10)
        for key in ('raw_qs', 'raw_qs_outer', 'iota_boundary', 'iota_min', 'iota_max', 'quality_penalty', 'surface_volume', 'flux_edge'):
            ga = torch.autograd.grad(actual[key].sum(), values, retain_graph=True)[0]
            ge = torch.autograd.grad(sum(e[key] for e in expected), values, retain_graph=True)[0]
            torch.testing.assert_close(ga, ge, atol=1e-8, rtol=1e-8)
            only_first = torch.autograd.grad(actual[key][0], values, retain_graph=True)[0]
            torch.testing.assert_close(only_first[1], torch.zeros_like(only_first[1]), atol=0, rtol=0)



if __name__ == '__main__':
    unittest.main()
