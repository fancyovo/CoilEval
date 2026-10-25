"""One-shot cache rejection is checked before any native GPU work begins."""
import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest

from coil_eval import CoilSet
from coil_eval import _binding as native


LIBRARY=os.environ.get('COIL_EVAL_TEST_LIBRARY')
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not LIBRARY,reason='set COIL_EVAL_TEST_LIBRARY')]


@pytest.fixture
def cache():
    library=ct.CDLL(LIBRARY);native._bind_native_score(library)
    p=ct.POINTER(ct.c_double);integer=ct.c_int;cfgp=ct.POINTER(native._SgpuScoreConfig)
    for name in ('sgpu_axis_coarse_cache_clear','sgpu_axis_coarse_cache_status'):
        f=getattr(library,name);f.argtypes=[];f.restype=integer
    library.sgpu_axis_coarse_grid.argtypes=[p,p,p,integer,integer,cfgp,p,p,integer]
    library.sgpu_axis_coarse_cache_set.argtypes=[p,p,p,p,integer,integer,integer,cfgp,p,p,p,p,integer]
    coils=CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    arrays=[np.ascontiguousarray(a) for a in (coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,coils.currents_a)]
    cfg=native._SgpuScoreConfig();assert library.sgpu_default_score_config(ct.byref(cfg))==0
    cfg.axis_grid=8
    R=np.empty(64);Z=np.empty(64);ptr=lambda a:a.ctypes.data_as(p)
    assert library.sgpu_axis_coarse_grid(*(ptr(a) for a in arrays[:3]),4,9,ct.byref(cfg),ptr(R),ptr(Z),64)==0
    def arm():
        # Endpoints are immaterial: these tests must reject before tracing.
        assert library.sgpu_axis_coarse_cache_set(*(ptr(a) for a in arrays),4,9,4,
            ct.byref(cfg),ptr(R),ptr(Z),ptr(R),ptr(Z),64)==0
        assert library.sgpu_axis_coarse_cache_status()==1
    assert library.sgpu_axis_coarse_cache_clear()==0
    yield library,arrays,cfg,arm,ptr
    library.sgpu_axis_coarse_cache_clear()


def test_null_score_result_consumes_pending_cache_before_return(cache):
    library,arrays,cfg,arm,ptr=cache
    arm()
    assert library.sgpu_score_coils(*(ptr(a) for a in arrays),4,9,4,ct.byref(cfg),None)!=0
    assert library.sgpu_axis_coarse_cache_status()==4
    arm()  # A second arm is possible only if the first entry was taken.


def test_physical_mismatch_rejects_and_consumes_before_cuda(cache):
    library,arrays,cfg,arm,ptr=cache
    arm();changed=arrays[0].copy();changed[0,0]=np.nextafter(changed[0,0],np.inf)
    result=native._SgpuScoreResult()
    assert library.sgpu_score_coils(ptr(changed),*(ptr(a) for a in arrays[1:]),4,9,4,
                                    ct.byref(cfg),ct.byref(result))!=0
    assert library.sgpu_axis_coarse_cache_status()==3
    arm()


def test_null_two_stage_result_also_consumes_pending_cache(cache):
    library,arrays,cfg,arm,ptr=cache
    p=ct.POINTER(ct.c_double);i=ct.c_int;void=ct.c_void_p
    fun=library.sgpu_two_stage_begin_v1
    fun.argtypes=[p,p,p,p,i,i,i,ct.POINTER(native._SgpuScoreConfig),void,void,void,p,i,void,p,i,void]
    fun.restype=i
    arm()
    assert fun(*(ptr(a) for a in arrays),4,9,4,ct.byref(cfg),None,None,None,None,0,None,None,0,None)!=0
    assert library.sgpu_axis_coarse_cache_status()==4
    arm()
