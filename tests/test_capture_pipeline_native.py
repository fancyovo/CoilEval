"""Real native capture equivalence with independent geometries and shared volumes."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

from coil_eval import CoilSet,EvaluationConfig
from coil_eval._autograd.axis_batch import prepare_axis_batch
from coil_eval._two_stage.capture_batch import capture_two_stage_batch,capture_two_stage_workers
from coil_eval._two_stage.native import TwoStageNative


LIBRARY=os.environ.get('COIL_EVAL_TEST_LIBRARY')
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not LIBRARY,reason='set COIL_EVAL_TEST_LIBRARY')]


def assert_tree_equal(left,right,path='snapshot'):
    # Timings and host-worker diagnostics differ by scheduling. All captured
    # coefficients, factors, sample membership and branch diagnostics must agree.
    if isinstance(left,np.ndarray):
        np.testing.assert_allclose(left,right,rtol=2e-12,atol=2e-14,err_msg=path)
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:
            if key not in {'timing','target_volume_cache_id',
                            'geometry_workers_requested','geometry_workers_effective'}:
                assert_tree_equal(left[key],right[key],path+'.'+key)
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for index,(a,b) in enumerate(zip(left,right)):
            assert_tree_equal(a,b,path+f'[{index}]')
    elif isinstance(left,(float,np.floating)):
        np.testing.assert_allclose(left,right,rtol=2e-12,atol=2e-14,err_msg=path)
    else:
        assert left==right,path


@pytest.mark.parametrize('sampling',['cartesian','polar_quadrature'])
@pytest.mark.parametrize('execution',['pipeline','workers1','workers2','workers4'])
def test_pipelined_capture_equals_serial_with_one_shot_axis_caches(sampling,execution):
    # Concurrent cuSOLVER calls require the explicit per-handle stream
    # isolation probe. Keep the default-stream path covered by the one-worker
    # and pipeline cases; the opt-in stream job covers workers 2/4.
    if execution in {'workers2','workers4'}:
        pytest.skip('workers2/4 are disabled pending native cuSOLVER handle isolation')
    coil=CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    tokens=np.concatenate((coil.coeffs_x,coil.coeffs_y,coil.coeffs_z,coil.currents_a[:,None]),axis=-1)
    physical=np.stack((tokens,tokens,tokens.copy()))
    physical[2,0,0]+=1e-5
    config=EvaluationConfig(target_volume=.03,psi_a=.15,fast=True,flux_method='stokes',
        psi_fit='two_stage',psi_refit_sampling=sampling)
    configs=[config,replace(config,target_volume=.02),config]
    groups=[[0,1],[2]]
    if execution=='workers4':
        extra=np.stack((tokens.copy(),tokens.copy()))
        extra[:,0,0]+=[-1e-5,2e-5]
        physical=np.concatenate((physical,extra))
        configs.extend((config,config));groups.extend(([3],[4]))
    representatives=[group[0] for group in groups]
    prepared=prepare_axis_batch(physical[representatives],[configs[i] for i in representatives],LIBRARY)
    if execution=='pipeline':
        parallel,metadata=capture_two_stage_batch(physical,configs,groups,LIBRARY,prepared_axis=prepared)
    else:
        parallel,metadata=capture_two_stage_workers(physical,configs,groups,LIBRARY,
            prepared_axis=prepared,workers=int(execution[-1]))
    reference=[None]*len(physical)
    prepared_reference=prepare_axis_batch(physical[representatives],[configs[i] for i in representatives],LIBRARY)
    ncoeff=coil.coeffs_x.shape[-1]
    for j,group in enumerate(groups):
        row=physical[group[0]]
        current=CoilSet(row[:,:ncoeff],row[:,ncoeff:2*ncoeff],row[:,2*ncoeff:3*ncoeff],row[:,-1],4)
        with prepared_reference.capture(j):
            pairs=TwoStageNative(LIBRARY).evaluate_many(current,[configs[i] for i in group],
                capture=True,include_torsion=False)
        for i,pair in zip(group,pairs):reference[i]=pair
    for (raw,snapshot),(expected,expected_snapshot) in zip(parallel,reference):
        assert raw['status']==expected['status']=='ok'
        assert_tree_equal(raw,expected,'result')
        assert_tree_equal(snapshot,expected_snapshot)
    assert all(items[0]['consumed'] for items in prepared.observations())
    assert all(items[0]['refinement_consumed'] for items in prepared.observations())
    assert 1<=metadata['maximum_pending_contexts']<=len(groups)
