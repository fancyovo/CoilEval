"""Real native sampled geometry values, active branches and six Jacobians."""
import os
from pathlib import Path

import numpy as np
import pytest

from coil_eval import CoilSet
from coil_eval._binding import coil_metric_gradient_native


LIBRARY=os.environ.get('COIL_EVAL_TEST_LIBRARY')
REFERENCE=os.environ.get('COIL_EVAL_REFERENCE_LIBRARY')
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not LIBRARY,reason='set COIL_EVAL_TEST_LIBRARY')]
NAMES=('length_mean','curvature_p95','curvature_max','min_intercoil_distance',
       'min_axis_distance','high_mode_fraction')


def coils_for_order(order):
    example=CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    nk=2*order+1;rng=np.random.default_rng(120+order)
    arrays=[]
    for original in (example.coeffs_x,example.coeffs_y,example.coeffs_z):
        a=np.zeros((4,nk));a[:,:9]=original
        if nk>9:a[:,9:]=rng.normal(0,1e-4,(4,nk-9))
        arrays.append(a)
    return arrays,example.currents_a.copy()


@pytest.mark.parametrize('order',[4,16])
@pytest.mark.parametrize('nfp',[1,4])
def test_all_geometry_metrics_and_derivatives_match_reference(order,nfp):
    if not REFERENCE:pytest.skip('set COIL_EVAL_REFERENCE_LIBRARY for revision parity')
    arrays,current=coils_for_order(order)
    candidate=coil_metric_gradient_native(LIBRARY,*arrays,current,nfp)
    old=coil_metric_gradient_native(REFERENCE,*arrays,current,nfp)
    assert candidate['spacing_pair']==old['spacing_pair']
    np.testing.assert_allclose(list(candidate['values'].values()),list(old['values'].values()),rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(candidate['derivatives'],old['derivatives'],rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('order',[4,16])
def test_all_six_analytic_jacobians_against_local_directions(order):
    arrays,current=coils_for_order(order)
    center=coil_metric_gradient_native(LIBRARY,*arrays,current,4)
    jac=np.asarray(center['derivatives'])
    rng=np.random.default_rng(551+order)
    for _ in range(4):
        direction=rng.normal(size=(3,4,2*order+1))
        direction/=np.linalg.norm(direction)
        h=1e-7
        plus=coil_metric_gradient_native(LIBRARY,*(a+h*d for a,d in zip(arrays,direction)),current,4)
        minus=coil_metric_gradient_native(LIBRARY,*(a-h*d for a,d in zip(arrays,direction)),current,4)
        expected=np.array([(plus['values'][n]-minus['values'][n])/(2*h) for n in NAMES])
        actual=np.einsum('mcik,cik->m',jac,direction)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-7)


def test_duplicate_coils_preserve_zero_gap_and_its_zero_derivative():
    arrays,current=coils_for_order(4)
    arrays=[np.repeat(a[:1],4,axis=0) for a in arrays]
    result=coil_metric_gradient_native(LIBRARY,*arrays,current,4)
    assert result['values']['min_intercoil_distance']==0
    np.testing.assert_array_equal(result['derivatives'][3],0)
    if REFERENCE:
        old=coil_metric_gradient_native(REFERENCE,*arrays,current,4)
        assert result['spacing_pair']==old['spacing_pair']
        np.testing.assert_allclose(result['derivatives'],old['derivatives'],rtol=2e-12,atol=2e-12)
