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


@pytest.mark.parametrize("nfp,nc", [(1,1),(2,3),(3,2),(5,6),(8,12)])
def test_pack_coils_supports_general_symmetric_geometry(nfp,nc):
    coefficients=np.arange(nc*5,dtype=float).reshape(nc,5)
    coils=CoilSet(coefficients,coefficients+1,coefficients+2,np.arange(nc)+1e5,nfp)
    x=pack_coils(coils,device='cpu')
    assert x.shape==(nc,16)
    np.testing.assert_array_equal(x[:,:5],coils.coeffs_x)
    np.testing.assert_array_equal(x[:,-1],coils.currents_a)


@pytest.mark.parametrize("nfp", [0,-1,1.5,True,'3'])
def test_tensor_rejects_invalid_nfp_before_loading_native(nfp):
    config=EvaluationConfig(.05,fast=True,flux_method='stokes',helicity=(1,0))
    with pytest.raises(ValueError,match='nfp must be a positive integer'):
        TensorEvaluator('unused',config=config,nfp=nfp)


@pytest.mark.parametrize("helicity", [(1,0),(1,2),(1,5),(2,3),(1,-3)])
def test_tensor_accepts_explicit_qa_and_qh_helicities(helicity):
    from coil_eval._tensor_config import validate_tensor_config
    validate_tensor_config(EvaluationConfig(.05,fast=True,flux_method='stokes',helicity=helicity))
