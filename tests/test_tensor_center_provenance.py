"""Captured physical field periods must survive tensor center conversion."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from coil_eval.torch import TensorEvaluator


def evaluator_and_snapshot():
    evaluator = object.__new__(TensorEvaluator)
    evaluator.device = torch.device("cpu")
    evaluator.nfp = 3
    snapshot = {
        name: np.zeros((1,), dtype=np.float64)
        for name in (
            "axis_samples", "psi_coefficients", "psi_scale", "psi_upper",
            "target_adjoint_weights", "target_axis_weights", "target_phi",
            "target_axis_R", "target_level", "target_adjoint_residual",
            "target_dVds",
        )
    }
    snapshot["config"] = {"psi_a": 0.05}
    return evaluator, snapshot


@pytest.mark.parametrize("source", ["top_level", "config"])
@pytest.mark.parametrize("captured_nfp", [2, np.array([2.0])])
def test_tensor_center_rejects_mismatched_field_period_metadata(source, captured_nfp):
    evaluator, snapshot = evaluator_and_snapshot()
    snapshot["nfp"] = np.array([3.0])
    snapshot["config"]["nfp"] = 3
    (snapshot if source == "top_level" else snapshot["config"])["nfp"] = captured_nfp
    with pytest.raises(ValueError, match="snapshot field periods do not match evaluator nfp"):
        evaluator._center(snapshot)


@pytest.mark.parametrize("metadata", ["missing", "top_level", "config", "both"])
def test_tensor_center_preserves_matching_and_legacy_metadata(metadata):
    evaluator, snapshot = evaluator_and_snapshot()
    if metadata in ("top_level", "both"):
        snapshot["nfp"] = np.array([3.0])
    if metadata in ("config", "both"):
        snapshot["config"]["nfp"] = 3
    center = evaluator._center(snapshot)
    assert center["config"]["nfp"] == 3
    assert center["config"]["psi_a"] == 0.05
    torch.testing.assert_close(center["psi_coefficients"], torch.zeros(1, dtype=torch.float64))
    if metadata == "missing":
        assert "nfp" not in snapshot["config"]
