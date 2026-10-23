"""A center is live only for its creating thread and evaluation generation."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from coil_eval import (CenterContextError, CoilSet, EvaluationConfig,
                       EvaluationError, EvaluationResult, Evaluator)
from _oracle import FakeBackend, make_coils


def session():
    coils = make_coils()
    backend = FakeBackend(coils)
    return coils, backend, Evaluator(config=EvaluationConfig(0.03), backend=backend)


def test_new_independent_center_invalidates_older_center():
    coils, backend, evaluator = session()
    old = evaluator.evaluate(coils)
    latest = evaluator.evaluate(coils)
    with pytest.raises(CenterContextError):
        evaluator.evaluate_nearby(old, [coils])
    assert evaluator.evaluate_nearby(latest, [coils])[0].valid
    assert [call[0] for call in backend.calls] == ["evaluate", "evaluate", "nearby"]


def test_configuration_cannot_change_under_a_live_center():
    coils, backend, evaluator = session()
    center = evaluator.evaluate(coils)
    with pytest.raises(AttributeError):
        evaluator.config = EvaluationConfig(0.07)
    assert evaluator.config.target_volume == 0.03
    assert evaluator.evaluate_nearby(center, [coils])[0].valid


def test_center_cannot_cross_evaluator_or_survive_json_roundtrip():
    coils, backend, evaluator = session()
    center = evaluator.evaluate(coils)
    other = Evaluator(config=EvaluationConfig(0.03), backend=FakeBackend(coils))
    restored = EvaluationResult.from_dict(json.loads(json.dumps(center.to_dict())))
    with pytest.raises(CenterContextError):
        other.evaluate_nearby(center, [coils])
    with pytest.raises(CenterContextError):
        evaluator.evaluate_nearby(restored, [coils])
    assert [call[0] for call in backend.calls] == ["evaluate"]


def test_nearby_result_does_not_become_a_live_center_and_batch_can_be_repeated():
    coils, backend, evaluator = session()
    center = evaluator.evaluate(coils)
    nearby = evaluator.evaluate_nearby(center, [coils])[0]
    with pytest.raises(CenterContextError):
        evaluator.evaluate_nearby(nearby, [coils])
    assert evaluator.evaluate_nearby(center, [coils])[0].valid
    assert evaluator.evaluate_nearby(center, []) == ()


def test_batch_evaluation_preserves_order_and_only_last_center_is_live():
    coils, backend, evaluator = session()
    p = coils.parameter_vector()
    variants = [coils.with_parameters(p + shift) for shift in (0, 0.01, -0.02)]
    results = evaluator.evaluate_many(variants)
    assert len(results) == 3
    for result, candidate in zip(results, variants):
        assert result.metrics.qs_volume == pytest.approx(backend.oracle.metrics(candidate)["qs_volume"])
    for center in results[:-1]:
        with pytest.raises(CenterContextError):
            evaluator.evaluate_nearby(center, [coils])
    assert evaluator.evaluate_nearby(results[-1], [coils])[0].valid


def test_session_rejects_cross_thread_calls_before_touching_backend():
    coils, backend, evaluator = session()
    center = evaluator.evaluate(coils)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for operation in (lambda: evaluator.evaluate(coils),
                          lambda: evaluator.evaluate_nearby(center, [coils]),
                          lambda: evaluator.close()):
            with pytest.raises(CenterContextError):
                executor.submit(operation).result()
    assert [call[0] for call in backend.calls] == ["evaluate"]
    assert evaluator.evaluate_nearby(center, [coils])[0].valid


def test_close_invalidates_session_and_context_manager_releases_backend():
    coils, backend, evaluator = session()
    with evaluator:
        center = evaluator.evaluate(coils)
    assert [call[0] for call in backend.calls] == ["evaluate", "close"]
    for operation in (lambda: evaluator.evaluate(coils),
                      lambda: evaluator.evaluate_nearby(center, [coils]),
                      lambda: evaluator.loss_and_grad(coils, backend.oracle.loss,
                                                      scales=backend.oracle.scales)):
        with pytest.raises(CenterContextError):
            operation()


def test_nearby_requires_same_topology_and_correct_backend_batch_length():
    coils, backend, evaluator = session()
    center = evaluator.evaluate(coils)
    bad_nfp = CoilSet(coils.coeffs_x, coils.coeffs_y, coils.coeffs_z, coils.currents_a, 3)
    with pytest.raises(ValueError):
        evaluator.evaluate_nearby(center, [bad_nfp])
    backend.nearby = lambda *args: []
    with pytest.raises(EvaluationError):
        evaluator.evaluate_nearby(center, [coils])


@pytest.mark.parametrize("stage", ["evaluate", "nearby"])
def test_native_gpu_error_is_a_public_evaluation_error_with_original_cause(stage):
    from coil_eval._binding import GpuError

    coils, backend, evaluator = session()
    original = GpuError("simulated native kernel failure")

    def fail(*args):
        raise original

    if stage == "evaluate":
        backend.evaluate = fail
        operation = lambda: evaluator.evaluate(coils)
    else:
        center = evaluator.evaluate(coils)
        backend.nearby = fail
        operation = lambda: evaluator.evaluate_nearby(center, [coils])
    with pytest.raises(EvaluationError, match=f"native {stage} failed") as captured:
        operation()
    assert type(captured.value) is EvaluationError
    assert captured.value.__cause__ is original
