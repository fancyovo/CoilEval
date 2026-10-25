"""Formal selected rays retain moving-point derivatives and snapshot ownership."""
from dataclasses import replace
import math

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from coil_eval._autograd.downstream import DownstreamConfig, volume_points
from coil_eval._autograd.front import _RadialRoot


class Geometry:
    def __init__(self, major, *, bad_phi=None):
        self.major = major
        self.bad_phi = bad_phi
        self.solved = []

    def axis_eval(self, phi):
        zero = torch.zeros_like(phi)
        return self.major + zero, zero, zero, zero

    def rays(self, phi, theta):
        # A ray without a boundary: s=r²-r⁴ has maximum 1/4.
        # Its selected counterpart uses the same genuinely implicit root code.
        p = torch.stack((torch.zeros_like(phi), torch.zeros_like(phi),
                         torch.ones_like(phi), torch.zeros_like(phi),
                         torch.zeros_like(phi)), -1)
        if self.bad_phi is not None:
            p[..., 4] = torch.where((phi > self.bad_phi) & (theta > 5.), -1., 0.)
        return p

    def solve_rays(self, polynomial, level):
        self.solved.append(polynomial.detach().clone())
        return _RadialRoot.apply(polynomial, level, 1., 1., 20, 1e-12)

    def psi_eval(self, R, Z, phi):
        return ((R-self.major).square()+Z.square(), 2*(R-self.major),
                2*Z, torch.zeros_like(R))


def fixture():
    config = replace(DownstreamConfig(), volume_point_count=6,
                     volume_phi_count=2, volume_theta_count=2)
    # Four angular rays, two radial candidates each; only the final ray is bad.
    return config, torch.tensor([0, 1, 2, 3, 4, 5]), torch.tensor([2, 2, 2, 8])


def test_formal_indices_skip_unselected_no_root_rays_and_preserve_point_order():
    config, _, shape = fixture()
    # Need six rows from eight candidates while omitting one complete bad ray.
    indices = torch.arange(6)
    geometry = Geometry(torch.tensor(2., dtype=torch.float64), bad_phi=.7)
    level = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    flux = {'coefficients': torch.tensor([1.], dtype=torch.float64), 'edge_flux': level}
    points = volume_points(geometry, level, flux, config,
                           source_indices=indices, sampling_shape=shape)
    assert geometry.solved[0].shape == (3, 5)
    assert points['xyz'].shape == (6, 3)
    assert points['boundary'][1, 1] == 0
    with pytest.raises(ValueError, match='radial root'):
        volume_points(geometry, level, flux, config)
    # Every point's cylindrical radius changes with the implicit boundary.
    actual = torch.autograd.grad(points['volume_weight'].sum(), level)[0]
    def value(l):
        return volume_points(geometry, l, {**flux, 'edge_flux': l}, config,
                             source_indices=indices, sampling_shape=shape)['volume_weight'].sum()
    h = 1e-6
    expected = (value(level.detach()+h)-value(level.detach()-h))/(2*h)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


def test_formal_full_selection_matches_historical_all_valid_coordinates_and_vjp():
    config = replace(DownstreamConfig(), volume_point_count=20, volume_phi_count=3,
                     volume_theta_count=2)
    nr = math.ceil(math.ceil(1.25*20)/6)
    indices = torch.arange(20)*30//20
    shape = torch.tensor([3, 2, nr, 6*nr])
    major = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    level = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    flux = {'coefficients': torch.tensor([1.], dtype=torch.float64), 'edge_flux': level}
    a = volume_points(Geometry(major), level, flux, config)
    b = volume_points(Geometry(major), level, flux, config,
                      source_indices=indices, sampling_shape=shape)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0., rtol=0.)
    ga = torch.autograd.grad(a['xyz'].square().sum()+a['volume_weight'].sum(),
                             (major, level), retain_graph=True)
    gb = torch.autograd.grad(b['xyz'].square().sum()+b['volume_weight'].sum(), (major, level))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('indices,shape,message', [
    ([0, 1], [2, 2, 2, 8], 'point count'),
    ([0, 1, 2, 3, 4, 8], [2, 2, 2, 8], 'candidate indices'),
    ([0, 1, 2, 3, 4, 4], [2, 2, 2, 8], 'candidate indices'),
    ([0, 1, 2, 3, 4, 4.5], [2, 2, 2, 8], 'candidate indices'),
    ([0, 1, 2, 3, 4, 5], [2, 2, 3, 12], 'sampling shape'),
    ([0, 1, 2, 3, 4, 5], None, 'require their sampling shape'),
])
def test_invalid_formal_sampling_contract(indices, shape, message):
    config, _, _ = fixture()
    level = torch.tensor(.04, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        volume_points(Geometry(level+2), level,
                       {'coefficients': level.new_tensor([1.]), 'edge_flux': level}, config,
                       source_indices=indices, sampling_shape=shape)


def test_capture_selected_indices_are_owned_int64(monkeypatch):
    from test_capture_ownership import CaptureField
    from coil_eval._autograd.capture import copy_reverse_snapshot
    field = CaptureField()
    field.arrays = {'volume_source_indices': np.array([0, 4, 7]),
                    'volume_sampling_shape': np.array([2, 2, 2, 8])}
    snapshot = copy_reverse_snapshot(field)
    assert all(a.dtype == np.int64 for a in snapshot.values())
    field.arrays['volume_source_indices'][...] = 1
    np.testing.assert_array_equal(snapshot['volume_source_indices'], [0, 4, 7])
