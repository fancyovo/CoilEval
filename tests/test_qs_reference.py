"""Independent manufactured-field tests for the Simsopt QS references."""
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval.simsopt import qs_reference as reference
from coil_eval.simsopt.state import INITIALIZER_KIND, SOURCE_KIND, STANDARD_KIND


def manufactured(shape=(65, 67), *, nfp=4, iota=1.27, G_rad=1.8, radius=1.2,
                 modes=((.20, 1, 1), (.07, 2, -1), (.03, 3, 2))):
    """Construct consistent local Boozer coordinate/field jets.

    This algebraic fixture tests the angular/Cartesian chain rule, rather than
    asserting that its local tangent jets embed as a global vacuum surface.
    """
    phi, theta = np.meshgrid((np.arange(shape[0])+.5)*2*np.pi/(nfp*shape[0]),
                             (np.arange(shape[1])+.5)*2*np.pi/shape[1], indexing="ij")
    b = np.full(shape, 2.0)
    bt, bp, btt, bpp, btp = (np.zeros(shape) for _ in range(5))
    for amplitude, m, n in modes:
        phase = m*theta-n*nfp*phi
        co, si = np.cos(phase), np.sin(phase)
        b += amplitude*co
        bt -= amplitude*m*si
        bp += amplitude*n*nfp*si
        btt -= amplitude*m*m*co
        bpp -= amplitude*(n*nfp)**2*co
        btp += amplitude*m*n*nfp*co
    G_turn = 2*np.pi*G_rad
    gt = np.zeros((*shape, 3)); gt[..., 1] = 2*np.pi
    gp = -iota*gt; gp[..., 0] = G_turn/b
    field = np.zeros_like(gp); field[..., 0] = b
    gradient = np.zeros((*shape, 3, 3))
    gradient[..., 0, 0] = b/G_rad*(bp+iota*bt)
    gradient[..., 0, 1] = bt
    expected = b**4/G_rad**2*(bt*(bpp+iota*btp)-bp*(btp+iota*btt))
    xyz = np.stack((phi, theta, np.zeros(shape)), axis=-1)
    inputs = dict(xyz=xyz, gphi=gp, gtheta=gt, B=field, dBdx=gradient, iota=iota,
                  G_turn=G_turn, nfp=nfp, major_radius=radius)
    return inputs, expected


def test_analytic_multimode_reference_matches_closed_form():
    inputs, expected = manufactured()
    result = reference.analyze_qs_arrays(**inputs)
    np.testing.assert_allclose(result["f_T_raw"], expected, atol=3e-12, rtol=2e-11)
    weights = abs(inputs["G_turn"]/(2*np.pi))/result["B_magnitude"]**2
    normalized = inputs["major_radius"]**2*expected/result["B_magnitude"]**4
    np.testing.assert_allclose(result["f_T"], np.sqrt(np.sum(weights*normalized**2)/weights.sum()), rtol=2e-13)
    # (m,n)=(1,1) is QH; (2,-1) and (3,2) are asymmetric.
    expected_fb = np.sqrt((.07**2+.03**2)/(2*2.0**2+.20**2+.07**2+.03**2))
    assert result["f_B"] == pytest.approx(expected_fb, rel=2e-14)
    assert result["relative_boozer_equation_error_rms"] < 1e-15
    assert result["field_vs_spectral_B_derivative_relative_rms"] < 2e-13


@pytest.mark.parametrize("modes,helicity", [(((.2, 1, 1), (.04, 3, 3)), (1, 4)),
                                           (((.2, 2, 0), (.04, 3, 0)), (1, 0))])
def test_exact_helical_and_axisymmetric_fields(modes, helicity):
    inputs, _ = manufactured(modes=modes)
    result = reference.analyze_qs_arrays(**inputs, helicity=helicity)
    assert result["f_B"] < 1e-14
    assert result["f_T"] < 1e-12


def test_normalization_and_field_jacobian_component_order():
    inputs, expected = manufactured()
    base = reference.analyze_qs_arrays(**inputs)
    scaled = dict(inputs, B=3*inputs["B"], dBdx=3*inputs["dBdx"], G_turn=3*inputs["G_turn"])
    result = reference.analyze_qs_arrays(**scaled)
    assert result["f_B"] == pytest.approx(base["f_B"], rel=1e-13)
    assert result["f_T"] == pytest.approx(base["f_T"], rel=1e-13)
    np.testing.assert_allclose(result["f_T_raw"], 3**4*expected, atol=3e-10)
    reversed_gradient = dict(inputs, dBdx=inputs["dBdx"].swapaxes(-1, -2))
    bad = reference.analyze_qs_arrays(**reversed_gradient)
    assert np.linalg.norm(bad["f_T_raw"]-expected) > .1*np.linalg.norm(expected)
    assert bad["field_vs_spectral_B_derivative_relative_rms"] > .1


def test_signed_G_and_explicit_radius_normalization():
    positive, _ = manufactured(G_rad=1.8)
    negative, _ = manufactured(G_rad=-1.8)
    a = reference.analyze_qs_arrays(**positive)
    b = reference.analyze_qs_arrays(**negative)
    assert b["f_T"] == pytest.approx(a["f_T"], rel=1e-13)
    doubled = reference.analyze_qs_arrays(**dict(positive, major_radius=2*positive["major_radius"]))
    assert doubled["f_T"] == pytest.approx(4*a["f_T"], rel=1e-13)


@pytest.mark.parametrize("change", [dict(major_radius=0), dict(major_radius=np.nan),
                                      dict(G_turn=0), dict(iota=np.inf),
                                      dict(nfp=True), dict(helicity=(0, 0)),
                                      dict(helicity=(1., 4)), dict(helicity=4)])
def test_invalid_scalar_inputs(change):
    inputs, _ = manufactured()
    with pytest.raises(ValueError):
        reference.analyze_qs_arrays(**dict(inputs, **change))


def test_invalid_grid_and_nonfinite_field():
    inputs, _ = manufactured(shape=(64, 65))
    with pytest.raises(ValueError, match="odd"):
        reference.analyze_qs_arrays(**inputs)
    inputs, _ = manufactured()
    inputs["B"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reference.analyze_qs_arrays(**inputs)


def surface_record(*, accepted=True):
    seed = SimpleNamespace(kind=INITIALIZER_KIND, source_kind=SOURCE_KIND,
                           nfp=4, order=12, stellsym=True)
    return SimpleNamespace(kind=STANDARD_KIND, seed=seed, accepted=accepted,
                           quality="strict_pass", dofs=np.ones(1), iota=1.27,
                           G=2*np.pi*1.8, diagnostics={})


def install_grid_fixture(monkeypatch, intersection_status="not_detected_in_sampled_sections"):
    cache = {}

    class Grid:
        def __init__(self, shape):
            self.inputs, _ = manufactured(shape=shape)
            cache[shape] = self.inputs
        def gamma(self): return self.inputs["xyz"]
        def gammadash1(self): return self.inputs["gphi"]
        def gammadash2(self): return self.inputs["gtheta"]
        def volume(self): return -.23

    def make_grid(dofs, *, nphi, ntheta, **kwargs):
        return Grid((nphi, ntheta))

    def field(points):
        # All wrapper tests use square grids; points are supplied in the same order.
        size = int(round(np.sqrt(len(points))))
        inputs = cache[(size, size)]
        return inputs["B"].reshape(-1, 3), inputs["dBdx"].reshape(-1, 3, 3)

    monkeypatch.setattr(reference, "surface_from_dofs", make_grid)
    from coil_eval.simsopt import intersections
    monkeypatch.setattr(intersections, "detect_self_intersection", lambda surface: {
        "status": intersection_status,
        "confirmed_self_intersection": intersection_status == "confirmed_self_intersection",
        "checked": True,
    })
    return field


def test_surface_wrapper_grid_convergence_and_optional_arrays(monkeypatch):
    field = install_grid_fixture(monkeypatch)
    result = reference.surface_qs_metrics(surface_record(), field, major_radius=1.2,
                                          sizes=(33, 65, 97), return_arrays=True)
    assert result["accepted"] and result["quality"] == "strict_pass"
    assert result["actual_volume_m3"] == .23
    assert result["quadrature_converged"] is True
    assert result["arrays"]["B_magnitude"].shape == (97, 97)
    assert len(result["grids"]) == 3
    assert "f_T_raw" not in result["grids"][-1]


def test_reference_rejects_old_accepted_self_intersection(monkeypatch):
    field = install_grid_fixture(monkeypatch, "confirmed_self_intersection")
    with pytest.raises(ValueError, match="self-intersection"):
        reference.surface_qs_metrics(surface_record(), field, major_radius=1.2, sizes=(33,))


def test_inconclusive_intersection_remains_warning(monkeypatch):
    field = install_grid_fixture(monkeypatch, "inconclusive")
    result = reference.surface_qs_metrics(surface_record(), field, major_radius=1.2, sizes=(33,))
    assert result["quality"] == "accepted_with_quality_warning"
    assert result["quadrature_converged"] is None


def test_rejected_or_wrong_provenance_surface_cannot_be_reference():
    with pytest.raises(ValueError, match="accepted"):
        reference.surface_qs_metrics(surface_record(accepted=False), None, major_radius=1.2)
    surface = surface_record()
    surface.seed.kind = "direct_point_cloud"
    with pytest.raises(ValueError, match="alpha_nu"):
        reference.surface_qs_metrics(surface, None, major_radius=1.2)


@pytest.mark.parametrize("sizes", [(), (32,), (33, 33), (65, 33), ((33, 34),), (True,)])
def test_surface_wrapper_grid_validation(sizes):
    with pytest.raises(ValueError):
        reference.surface_qs_metrics(surface_record(), None, major_radius=1.2, sizes=sizes)
