"""GPU integration checks; set COIL_EVAL_TEST_LIBRARY to enable."""
import os
from pathlib import Path
import numpy as np
import pytest
from coil_eval import CoilSet, EvaluationConfig, Evaluator, quality_penalty

LIBRARY = os.environ.get('COIL_EVAL_TEST_LIBRARY')
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not LIBRARY, reason='set COIL_EVAL_TEST_LIBRARY')]

@pytest.mark.parametrize('volume,current_factor', [(0.05, 1.), (0.04, 1.), (0.05, 1.1)])
def test_value_and_full_gradient(volume, current_factor):
    coils = CoilSet.read_json(Path(__file__).parents[1]/'examples/coils.json')
    coils = CoilSet(coils.coeffs_x, coils.coeffs_y, coils.coeffs_z,
                    coils.currents_a*current_factor, coils.nfp)
    cfg = EvaluationConfig(target_volume=volume)
    scales = np.r_[np.ones(3*coils.coeffs_x.size), np.full(coils.n_base_coils, 100000.)]
    def loss(m):
        return np.log(m.qs_volume+1e-8)+quality_penalty(m.relative_normal_drift_p95)
    with Evaluator(LIBRARY, config=cfg) as ev:
        result = ev.loss_and_grad(coils, loss, scales=scales)
        assert result.center.valid
        assert len(result.endpoints) == 2*coils.parameter_count
        assert np.isfinite(result.gradient).all()
        assert np.linalg.norm(result.gradient) > 0
        invalid = CoilSet(coils.coeffs_x, coils.coeffs_y, coils.coeffs_z,
                          np.zeros_like(coils.currents_a), coils.nfp)
        assert not ev.evaluate(invalid).valid
    reference = os.environ.get('COIL_EVAL_REFERENCE_LIBRARY')
    if reference:
        with Evaluator(reference, config=cfg) as ev:
            previous = ev.loss_and_grad(coils, loss, scales=scales)
        np.testing.assert_allclose(result.value, previous.value, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.gradient, previous.gradient, rtol=1e-7, atol=1e-8)
