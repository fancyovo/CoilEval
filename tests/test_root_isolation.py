"""Exact resolution of ambiguous radial derivative roots."""
from collections import Counter
from fractions import Fraction as F
from math import lcm
from random import Random

import numpy as np
import pytest

from coil_eval._two_stage import boundary
from coil_eval._two_stage.branches import branch_limits
from coil_eval._two_stage.root_isolation import certified_first_branch


# Actual rejected rays: the near-complex pair is outside the first real fold.
REJECTED_RAYS = [
    ([0., 0., .8763182170184891, .42412150681374694, -1.379976234004204,
      -1.1403185763565953, .8018317811549384, 1.0293949637860929,
      .11699065205942054, -.30361880778175293, -.2057937609041515],
     .7705067548446554),
    ([0., 0., .8761673791422774, .424254456395925, -1.3793219202739362,
      -1.140742768332754, .8014081800338968, 1.0300187327356685,
      .11660598617462274, -.30393480059593536, -.2054443552768517],
     .7707138334738932),
]


@pytest.mark.parametrize('polynomial,expected', REJECTED_RAYS)
def test_observed_outer_near_complex_pair_does_not_invalidate_first_fold(polynomial, expected):
    with pytest.raises(boundary.MaxClosedBoundaryError, match='uncertain near-real'):
        boundary._floating_first_connected_branch(polynomial, 1.)
    result = boundary.first_connected_branch(polynomial, 1.)
    assert result['maximum_u'] == pytest.approx(expected, abs=5e-14)
    assert result['interval_signs'] == [1, -1]
    assert result['exact_root_isolation_applied']
    assert len(result['critical_points']) == 1
    assert result['floating_root_diagnostics']['root_real'] > result['maximum_u']+.17
    assert not result['native_bernstein_applied']


def test_batched_branch_limits_resolve_only_ambiguous_rays():
    p = np.zeros((3, 11))
    p[0, 2] = 1.
    p[1:] = [row for row, _ in REJECTED_RAYS]
    caps, folds, flat, fallbacks = branch_limits(p, np.ones(3), boundary.first_connected_branch)
    np.testing.assert_allclose(caps, [1., *[expected for _, expected in REJECTED_RAYS]], atol=5e-14, rtol=0)
    np.testing.assert_array_equal(folds, [False, True, True])
    assert not flat.any()
    assert fallbacks == 2


def test_normal_fast_path_and_invalid_inputs_do_not_use_exact_resolver(monkeypatch):
    def forbidden(*args):
        raise AssertionError('exact resolver should not run')
    monkeypatch.setattr(boundary, 'certified_first_branch', forbidden)
    assert boundary.first_connected_branch([0., 0., 1., 0., .1], 1.)['maximum_u'] == 1.
    with pytest.raises(boundary.MaxClosedBoundaryError, match='positive near-axis quadratic'):
        boundary.first_connected_branch([0., 0., -1.], 1.)


def test_exact_solver_failure_remains_an_explicit_error(monkeypatch):
    def unresolved(*args):
        raise ArithmeticError('unresolved exact interval')
    monkeypatch.setattr(boundary, 'certified_first_branch', unresolved)
    with pytest.raises(boundary.MaxClosedBoundaryError, match='exact derivative-root isolation could not resolve'):
        boundary.first_connected_branch(REJECTED_RAYS[0][0], 1.)


def test_flat_nonfold_and_root_at_physical_cap():
    flat = certified_first_branch([0., 0., 1.5, -2., .75], 2.)
    assert flat['maximum_u'] == 2.
    assert flat['flat_nonfold_count'] == 1
    assert flat['critical_points'][0]['exact_root']
    cap = certified_first_branch([0., 0., 1.5, -1.], 1.)
    assert cap['maximum_u'] == 1.
    assert cap['critical_points'] == []


def test_tiny_real_negative_interval_is_distinguished_from_positive_near_complex_pair():
    negative = certified_first_branch([0., 0., .125-2**-41, -1/3, .25], 1.)
    assert abs(negative['maximum_u']-(.5-2**-20)) < 1e-9
    assert [event['kind'] for event in negative['critical_points']] == ['fold_maximum', 'fold_minimum']
    positive = certified_first_branch([0., 0., .125+2**-41, -1/3, .25], 1.)
    assert positive['maximum_u'] == 1.
    assert positive['critical_points'] == []


def test_rounded_derivative_double_root_does_not_hide_real_negative_interval():
    polynomial=[0.,0.,.125,np.nextafter(-1/3,-np.inf),.25]
    # Float multiplication rounds the q coefficient to exactly -1. The exact
    # derivative of the supplied p coefficients has a small negative pocket.
    assert (3*np.asarray(polynomial)[3]) == -1.
    result=boundary.first_connected_branch(polynomial,1.)
    assert result['exact_root_isolation_trigger']=='merged_companion_roots'
    assert result['maximum_u'] == pytest.approx(.4999999925494212,abs=5e-14)
    assert result['interval_signs']==[1,-1,1]
    caps,folds,_,_=branch_limits(np.asarray([polynomial]),np.ones(1),boundary.first_connected_branch)
    assert caps[0] == result['maximum_u'] and folds[0]


def test_positive_near_real_complex_pair_uses_exact_isolation_in_scalar_and_batch():
    polynomial=[0.,0.,.125+2**-35,-1/3,.25]
    result=boundary.first_connected_branch(polynomial,1.)
    assert result['maximum_u']==1.
    assert result['critical_points']==[]
    assert result['exact_root_isolation_trigger']=='near_real_complex_pair'
    caps,folds,_,fallbacks=branch_limits(np.asarray([polynomial]),np.ones(1),boundary.first_connected_branch)
    assert caps[0]==1. and not folds[0] and fallbacks==1


def test_known_real_complex_and_repeated_root_factors():
    rng = Random(513)
    def multiply(a, b):
        answer = [F(0)]*(len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                answer[i+j] += x*y
        return answer
    for _ in range(60):
        real = [F(rng.choice([-4, 1, 2, 3, 4, 6, 7, 8, 12]), 8) for _ in range(rng.randrange(1, 6))]
        q = [F(1)]
        for root in real:
            q = multiply(q, [-root, F(1)])
        q = multiply(q, [F(25, 64)+F(1, 2**30), F(-10, 8), F(1)])
        scale = lcm(*range(2, len(q)+2))*(1 if q[0] > 0 else -1)
        p = [0., 0.]+[float(scale*x/(i+2)) for i, x in enumerate(q)]
        assert all(F.from_float(p[i+2]) == scale*x/(i+2) for i, x in enumerate(q))
        result = certified_first_branch(p, 1.)
        folds = [r for r, multiplicity in Counter(real).items() if 0 < r < 1 and multiplicity%2]
        expected = float(min(folds)) if folds else 1.
        assert abs(result['maximum_u']-expected) < 1e-13
