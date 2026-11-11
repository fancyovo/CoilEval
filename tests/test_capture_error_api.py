"""Public batch errors retain native failure, query identity and cleanup."""
from dataclasses import replace

import pytest

from test_surface_qs_api import tensor_api
from test_two_stage_native import factory, Function
from coil_eval import CoilSet
from coil_eval.evaluator import InvalidEvaluationError
from coil_eval._two_stage import capture_batch
from coil_eval._two_stage.native import TwoStageNative


@pytest.mark.parametrize('pipeline', [False, True])
def test_native_error_query_identity(tensor_api, factory, monkeypatch, pipeline):
    from coil_eval._autograd.engineering import split_tokens

    torch, api, evaluator, _ = tensor_api
    _, library, _ = factory()
    original_begin = library.sgpu_two_stage_begin_v1
    began = []

    def begin(*args):
        began.append(len(began))
        code = original_begin(*args)
        if len(began) == 2:
            args[8]._obj.error_message = b'injected native stage1 failure'
            return 19
        return code

    library.sgpu_two_stage_begin_v1 = Function(begin)
    real_pipeline = capture_batch.capture_two_stage_batch

    def pipeline_on_fake_device(*args, **kwargs):
        kwargs['device_id'] = 0
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(capture_batch, 'capture_two_stage_batch', pipeline_on_fake_device)
    tokens = torch.ones((2, 4, 10), dtype=torch.float64)
    tokens[1, 0, 0] += .01
    with evaluator(capture_pipeline=pipeline, share_geometry=False) as engine:
        engine.config = replace(engine.config, psi_fit='two_stage', psi_refit_points=4,
                                psi_refit_sampling='cartesian')

        def scalar_capture(row, config):
            coils = CoilSet(*split_tokens(row), nfp=4)
            return TwoStageNative('fake.so', 0).evaluate(
                coils, config, capture=True, include_torsion=False)[1]

        engine._capture = scalar_capture
        with pytest.raises(InvalidEvaluationError) as caught:
            engine.evaluate_batch(tokens)

    message = str(caught.value)
    print(f'pipeline={pipeline}: {message}')
    assert 'injected native stage1 failure' in message
    assert 'geometry 1:' in message
    if pipeline:
        assert ': stage1:' in message
    assert len(began) == 2
    assert sum(name == 'close' for name, _ in library.calls) == 2
    assert library.flags[-1] == 0


def test_pipelined_target_failure_names_scalar_fit(tensor_api, monkeypatch):
    torch, _, evaluator, _ = tensor_api
    failure = dict(status='no_surface', diagnostics=dict(
        failure_stage='s_target_surface', error_message='scalar fit 2 is radially nonmonotone'))
    monkeypatch.setattr(capture_batch, 'capture_two_stage_batch',
                        lambda *args, **kwargs: ([(failure, None), (failure, None)], {}))
    tokens = torch.ones((2, 4, 10), dtype=torch.float64)
    with evaluator(capture_pipeline=True, share_geometry=False) as engine:
        engine.config = replace(engine.config, psi_fit='two_stage')
        with pytest.raises(InvalidEvaluationError, match=r'geometry 0: no_surface\. scalar fit 2 is radially nonmonotone'):
            engine.evaluate_batch(tokens)
