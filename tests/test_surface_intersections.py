"""Independent analytic geometry and final-acceptance regression checks."""
import sys
import types

import numpy as np
import pytest

from coil_eval.simsopt.intersections import detect_self_intersection
from coil_eval.simsopt.solver import SolverConfig, StandardSimsoptSolver, validate_surface
from coil_eval.simsopt.state import SurfaceSeed


class AnalyticSurface:
    nfp = 4
    ntor = 2

    def __init__(self, shape="circle", twist=0., folded=False, size=25, nfp=4):
        self.shape, self.twist, self.folded, self.size, self.nfp = shape, twist, folded, size, nfp

    def values(self, phi, theta):
        phi, theta = np.broadcast_arrays(phi, theta)
        t = 2 * np.pi * theta
        if self.shape == "eight":
            r, z = 2 + .2 * np.sin(t), .1 * np.sin(2 * t)
            dr, dz = .4 * np.pi * np.cos(t), .4 * np.pi * np.cos(2 * t)
        elif self.shape == "bean":
            r, z = 2 + .2 * (np.cos(t) + .8 * np.cos(2 * t)), .14 * np.sin(t)
            dr, dz = .4 * np.pi * (-np.sin(t) - 1.6 * np.sin(2 * t)), .28 * np.pi * np.cos(t)
        else:
            r, z = 2 + .2 * np.cos(t), .2 * np.sin(t)
            dr, dz = -.4 * np.pi * np.sin(t), .4 * np.pi * np.cos(t)
        angle = 2 * np.pi * phi + self.twist * np.sin(t)
        dphi, dtheta = np.full_like(phi, 2 * np.pi), 2 * np.pi * self.twist * np.cos(t)
        if self.folded:
            angle += 1.2 / self.nfp * np.sin(2 * np.pi * self.nfp * phi)
            dphi += 2.4 * np.pi * np.cos(2 * np.pi * self.nfp * phi)
        xyz = np.stack((r * np.cos(angle), r * np.sin(angle), z), axis=-1)
        gp = np.stack((-r * np.sin(angle) * dphi, r * np.cos(angle) * dphi, np.zeros_like(r)), axis=-1)
        gt = np.stack((dr * np.cos(angle) - r * np.sin(angle) * dtheta,
                       dr * np.sin(angle) + r * np.cos(angle) * dtheta, dz), axis=-1)
        return xyz, gp, gt

    def gamma_lin(self, out, phi, theta):
        out[:] = self.values(phi, theta)[0]

    def grid(self):
        return self.values(np.arange(self.size)[:, None] / (self.nfp * self.size),
                           np.arange(self.size)[None, :] / self.size)

    def gamma(self): return self.grid()[0]
    def gammadash1(self): return self.grid()[1]
    def gammadash2(self): return self.grid()[2]
    def volume(self): return .1
    def get_dofs(self): return np.ones(3)


@pytest.mark.parametrize("nfp,twist", [(1, 0.), (1, 4.), (4, 0.), (4, 4.)])
def test_embedded_torus_and_branch_cut_safe_parameter_twist(nfp, twist):
    result = detect_self_intersection(AnalyticSurface(nfp=nfp, twist=twist))
    assert result["status"] == "not_detected_in_sampled_sections"
    assert not result["confirmed_self_intersection"]
    assert result["sampled_sections"] == 8 and result["absence_is_not_a_certificate"]
    assert result["surface_point_evaluations"] < 40000


def test_nonstarshaped_embedded_section_is_not_rejected():
    result = detect_self_intersection(AnalyticSurface(shape="bean", twist=.8))
    assert result["status"] == "not_detected_in_sampled_sections"
    # This analytic simple curve has a reversing polar angle about its center.
    t = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    numerator = (np.cos(t) + .8 * np.cos(2 * t)) * np.cos(t) + np.sin(t) * (np.sin(t) + 1.6 * np.sin(2 * t))
    assert numerator.min() < 0 < numerator.max()


def test_original_surface_witness_stops_after_first_section():
    pytest.importorskip("scipy")
    surface = AnalyticSurface(shape="eight", twist=.8)
    result = detect_self_intersection(surface)
    assert result["status"] == "confirmed_self_intersection"
    assert result["confirmed_self_intersection"] and result["sampled_sections"] == 1
    witness = result["witness"]
    pair = surface.values(np.asarray(witness["parameter_phi_cycles"]), np.asarray(witness["parameter_theta_cycles"]))[0]
    assert np.linalg.norm(pair[0] - pair[1]) < 1e-12
    assert witness["normalized_parameter_separation"] > .49


def test_nonmonotone_phi_is_inconclusive_not_a_clean_geometry_result():
    result = detect_self_intersection(AnalyticSurface(folded=True))
    assert result["status"] == "inconclusive_toroidal_map"
    assert not result["confirmed_self_intersection"] and result["sampled_sections"] == 0


def test_validate_surface_does_not_require_a_source_axis():
    pytest.importorskip("scipy")
    surface = AnalyticSurface(shape="eight")
    result = validate_surface(surface, lambda xyz: np.tile([0, 1., 0], (len(xyz), 1)), iota=1.6, G=1.)
    assert result["finite"] and result["normal_min"] > 1e-12
    assert result["angular_covering"]["reason"] == "source_axis_unavailable"
    assert result["self_intersection"]["confirmed_self_intersection"]


@pytest.mark.parametrize("shape,forced_inconclusive,accepted,quality", [
    ("circle", False, True, "strict_pass"),
    ("eight", False, False, "rejected"),
    ("circle", True, True, "accepted_with_quality_warning"),
])
def test_solver_applies_one_final_guard_and_preserves_inconclusive_warning(
        monkeypatch, shape, forced_inconclusive, accepted, quality):
    if shape == "eight":
        pytest.importorskip("scipy")
    from coil_eval.simsopt import intersections, solver
    calls = []
    original = intersections.detect_self_intersection

    def checked(surface):
        calls.append(surface)
        if forced_inconclusive:
            return dict(status="inconclusive_toroidal_map", checked=True, confirmed_self_intersection=False)
        return original(surface)

    monkeypatch.setattr(intersections, "detect_self_intersection", checked)
    monkeypatch.setattr(solver, "surface_from_dofs", lambda dofs, **kwargs: AnalyticSurface(shape=shape, size=kwargs["nphi"]))

    class FakeBoozer:
        def __init__(self, *args): pass
        def minimize_boozer_penalty_constraints_ls(self, **kwargs):
            return dict(success=True, iota=1.6, G=1., iter=1)
        def solve_residual_equation_exactly_newton(self, **kwargs):
            return dict(success=True, iota=1.6, G=1., iter=1)

    geo = types.ModuleType("simsopt.geo")
    geo.BoozerSurface, geo.Volume = FakeBoozer, lambda surface: surface
    geo.boozer_surface_residual = lambda *args, **kwargs: (np.zeros(3),)
    package = types.ModuleType("simsopt"); package.geo = geo
    monkeypatch.setitem(sys.modules, "simsopt", package)
    monkeypatch.setitem(sys.modules, "simsopt.geo", geo)
    seed = SurfaceSeed(np.ones(3), 4, 1, 1.6, 1., .2, .1, 0.)
    config = SolverConfig(validation_grids=(17, 25), max_final_relative_l2=1e9, max_final_normal_p95=1e9)
    solved = StandardSimsoptSolver(config=config).solve(seed, lambda xyz: np.tile([0, 1., 0], (len(xyz), 1)))
    assert len(calls) == 1
    assert solved.accepted is accepted and solved.quality == quality
    hard = solved.diagnostics["hard_downstream_checks"]
    assert hard["no_detected_self_intersection"] is (shape != "eight")
    assert "single_angular_covering" not in hard
    assert solved.diagnostics["strict_quality_checks"]["self_intersection_audit_resolved"] is (shape == "circle" and not forced_inconclusive)
