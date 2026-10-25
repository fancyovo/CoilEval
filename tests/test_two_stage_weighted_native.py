"""Weighted C-ABI dispatch and legacy-library compatibility on host fakes."""
import numpy as np
import pytest

from coil_eval._two_stage import native as module
from _oracle import make_coils
from test_two_stage_native import factory, Function, configurations


def sampling_with_weights(monkeypatch, weights):
    original=module.sample_frozen_sections
    def sample(*args,**kwargs):
        result=original(*args,**kwargs)
        result['row_weights']=np.asarray(weights)
        return result
    monkeypatch.setattr(module,'sample_frozen_sections',sample)


def test_weighted_fit_dispatches_explicit_weights_and_captures_them(factory,monkeypatch):
    driver,library,_=factory();calls=[]
    weights=np.array([.5,1.,1.,1.5])
    sampling_with_weights(monkeypatch,weights)
    def weighted(*args):
        calls.append(np.ctypeslib.as_array(args[5],shape=(args[4],)).copy())
        return library.fit(*(args[:5]+args[6:]))
    library.sgpu_two_stage_fit_weighted_v1=Function(weighted)
    result,snapshot=driver.evaluate(make_coils(),configurations()[0],capture=True)
    assert result['status']=='ok' and len(calls)==1
    np.testing.assert_array_equal(calls[0],weights)
    np.testing.assert_array_equal(snapshot['two_stage']['sampling']['row_weights'],weights)
    assert result['diagnostics']['psi_refit_branch']['weighted_fit']
    assert driver._context is None


@pytest.mark.parametrize('weights',[[1.,1.],[0.,1.,1.,2.],[-1.,1.,1.,3.],
                                    [np.nan,1.,1.,1.],[2.,2.,2.,2.]])
def test_invalid_weights_fail_before_native_fit_and_release_context(factory,monkeypatch,weights):
    driver,library,_=factory();sampling_with_weights(monkeypatch,weights)
    with pytest.raises(ValueError,match='row_weights'):
        driver.evaluate(make_coils(),configurations()[0])
    assert not any(name=='fit' for name,_ in library.calls)
    assert driver._context is None and library.flags==[0,0]


def test_weighted_sampling_explains_missing_new_symbol_while_legacy_path_remains_usable(factory,monkeypatch):
    driver,library,_=factory();sampling_with_weights(monkeypatch,np.ones(4))
    with pytest.raises(RuntimeError,match='rebuild'):
        driver.evaluate(make_coils(),configurations()[0])
    assert driver._context is None
