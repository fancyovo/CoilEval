from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

def test_physical_tensor_layout_and_configuration():
    coils=CoilSet.read_json(Path(__file__).parents[1]/'examples/coils.json')
    x=pack_coils(coils,device='cpu')
    assert x.shape==(4,28)
    np.testing.assert_array_equal(x[:,:9],coils.coeffs_x)
    np.testing.assert_array_equal(x[:,9:18],coils.coeffs_y)
    np.testing.assert_array_equal(x[:,18:27],coils.coeffs_z)
    np.testing.assert_array_equal(x[:,27],coils.currents_a)
    with pytest.raises(ValueError,match='fast=True'):
        TensorEvaluator('unused',config=EvaluationConfig(target_volume=.05))
    with pytest.raises(ValueError,match='256 segments'):
        TensorEvaluator('unused',config=EvaluationConfig(target_volume=.05,fast=True,
                         flux_method='stokes',segments_per_coil=128))
