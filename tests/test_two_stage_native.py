"""C-ABI lifecycle and multi-volume fit reuse without a CUDA installation."""
import ctypes as ct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval import EvaluationConfig
from coil_eval._two_stage import native as module
from _oracle import make_coils


class Function:
    def __init__(self,function):self.function=function
    def __call__(self,*args):return self.function(*args)


class FakeLibrary:
    def __init__(self,fail=None,statuses=None):
        self.calls=[];self.flags=[];self.fail=fail;self.statuses=statuses or {}
        self.volume=None;self.factor_count=0;self.snapshot_count=0
        for name,method in (
            ('sgpu_default_score_config',self.default),('sgpu_set_reverse_capture',self.capture),
            ('sgpu_two_stage_begin_v1',self.begin),('sgpu_two_stage_fit_v1',self.fit),
            ('sgpu_two_stage_resume_v1',self.resume),('sgpu_two_stage_close_v1',self.close),
            ('sgpu_copy_psi_warm_preconditioner',self.factor)):
            setattr(self,name,Function(method))

    def default(self,config):
        cfg=config._obj
        for name,value in dict(psi_poly_degree=2,psi_m_tor=0,axis_sample_count=8,psi_n_phi=4,
                               psi_a=.15,iota_degree=3,segments_per_coil=256).items():
            setattr(cfg,name,value)
        return 0

    def capture(self,value):self.flags.append(value);return 0

    def begin(self,*args):
        self.calls.append(('begin',args[7]._obj.target_volume))
        args[9]._obj.value=37;args[10]._obj.value=41
        if self.fail=='begin':return 19
        if self.fail=='stage1_status':
            args[8]._obj.status=3;args[9]._obj.value=None;return 0
        n=args[12]
        values=np.ctypeslib.as_array(args[11],shape=(4*n,)).reshape(4,n)
        values[:]=0;values[0]=1.
        args[13]._obj.value=n
        np.ctypeslib.as_array(args[14],shape=(args[15],))[:]=[0.,1.]
        args[16]._obj.value=args[15]
        args[8]._obj.status=0
        return 0

    def factor(self,upper,scale,count):
        self.factor_count+=1;self.calls.append(('factor',count))
        np.ctypeslib.as_array(upper,shape=(count*count,))[:]=np.eye(count).ravel()*self.factor_count
        np.ctypeslib.as_array(scale,shape=(count,))[:]=self.factor_count
        return 0

    def fit(self,*args):
        self.calls.append(('fit',args[4]))
        if self.fail=='fit':return 19
        np.ctypeslib.as_array(args[5],shape=(2,))[:]=[.01,1.1]
        args[6]._obj.value=.012
        return 0

    def resume(self,*args):
        cfg,raw=args[4]._obj,args[5]._obj
        self.volume=float(cfg.target_volume)
        self.calls.append(('resume',self.volume))
        if self.fail=='resume':return 19
        raw.status=self.statuses.get(self.volume,0)
        raw.score=self.volume
        raw.qs_target_global_error_per_helicity=2*self.volume
        raw.target_volume_requested=self.volume
        np.ctypeslib.as_array(args[6],shape=(args[7],))[:]=np.arange(args[7])+self.volume
        args[8]._obj.value=args[7];args[9]._obj.value=.5+self.volume
        return 0

    def close(self,context):self.calls.append(('close',context.value))


@pytest.fixture
def factory(monkeypatch):
    libraries=[];geometry_calls=[]
    monkeypatch.setattr(module.native,'_bind_native_score',lambda lib:None)
    monkeypatch.setattr(module,'bind_capture',lambda lib:None)
    def check(lib,code):
        if code:raise RuntimeError(f'mock C ABI failure {code}')
    monkeypatch.setattr(module.native,'_check_lib_code',check)
    def boundary(*args,**kwargs):
        geometry_calls.append('boundary')
        return dict(volume_m3=.3,polygons_rz=np.ones((4,8,2)),diagnostics=dict(
            limiting_ray=dict(phi_index=0,theta_index=1),active_capacity_source='fold',active_capacity_gap=.2))
    def sample(*args,**kwargs):
        geometry_calls.append('sample')
        return dict(rz=np.array([[1.,.01],[1.02,-.03],[.98,.02],[1.01,.04]]),
            phi=np.arange(4)*.1,section_index=np.arange(4),
            lattice_indices=np.arange(8).reshape(4,2),spacing=.02,metadata=dict(target_count=4))
    monkeypatch.setattr(module,'max_closed_boundary',boundary)
    monkeypatch.setattr(module,'sample_frozen_sections',sample)
    def snapshot(driver,*,shared=None,names=None):
        lib=driver.lib;lib.snapshot_count+=1
        result=dict(axis_samples=np.ones((8,4)),psi_coefficients=np.array([.01,1.1]),
            target_level=np.array(lib.volume),alpha_mode_l=np.array([0]),
            alpha_center_scaled_solution=np.array([lib.volume+1,.2,.3]),alpha_scale=np.ones(3))
        result.update({key:value for key,value in (shared or {}).items() if key in result})
        return {key:value for key,value in result.items() if names is None or key in names}
    monkeypatch.setattr(module,'copy_reverse_snapshot',snapshot)
    def create(**kwargs):
        library=FakeLibrary(**kwargs);libraries.append(library)
        monkeypatch.setattr(module.ct,'CDLL',lambda path:library)
        return module.TwoStageNative('unused-native.so'),library,geometry_calls
    return create


def configurations():
    base=EvaluationConfig(.03,psi_fit='two_stage',psi_refit_points=4)
    return [base,replace(base,target_volume=.01),replace(base,target_volume=.04)]


def test_multiple_volumes_begin_and_refit_once_resume_each_in_order(factory):
    driver,library,geometry=factory()
    results=driver.evaluate_many(make_coils(),configurations())
    assert [value for name,value in library.calls if name=='begin']==[0.]
    assert [value for name,value in library.calls if name=='resume']==[.03,.01,.04]
    assert sum(name=='fit' for name,_ in library.calls)==1
    assert sum(name=='close' for name,_ in library.calls)==1
    assert geometry==['boundary','sample'] and library.factor_count==0
    assert [r['score'] for r,s in results]==[.03,.01,.04]
    assert all(s is None for r,s in results)
    assert driver._context is None and library.flags==[0,0]
    record=driver.last_many_timing
    assert record['query_count']==record['completed_volumes']==3
    assert len(record['each_formal_s'])==len(record['each_postprocess_s'])==3
    expected=record['shared_preparation_s']+sum(record['each_formal_s'])+sum(record['each_postprocess_s'])+record['overhead_s']
    assert record['total_s']==pytest.approx(expected,rel=1e-12,abs=1e-12)
    assert all(r['timing']['two_stage_group']==record for r,s in results)


def test_capture_owns_volume_state_and_shares_readonly_common_arrays(factory):
    driver,library,_=factory()
    results=driver.evaluate_many(make_coils(),configurations(),capture=True,diagnostics=True)
    assert library.factor_count==2 and library.snapshot_count==3
    assert library.flags==[3,0] and driver._context is None
    first,second=results[0][1],results[1][1]
    assert [s['config']['target_volume'] for _,s in results]==[.03,.01,.04]
    assert [float(s['target_level']) for _,s in results]==[.03,.01,.04]
    for name in ('psi_scale','psi_upper'):
        assert first[name] is second[name] and not first[name].flags.writeable
    for name in ('stage1_scale','stage1_upper'):
        assert first['two_stage'][name] is second['two_stage'][name]
        assert not first['two_stage'][name].flags.writeable
    first['target_level'][...]=-99
    first['alpha_center_scaled_solution'][0]=-99
    for name in ('axis_samples','psi_coefficients','alpha_mode_l'):
        assert first[name] is second[name] and not first[name].flags.writeable
    for scope,name in ((None,'stage1_coefficients'),('sampling','rz'),('boundary','polygons_rz')):
        left=first['two_stage'] if scope is None else first['two_stage'][scope]
        right=second['two_stage'] if scope is None else second['two_stage'][scope]
        assert left[name] is right[name] and not left[name].flags.writeable
        with pytest.raises(ValueError):left[name].flat[0]=-99
    first['two_stage']['boundary']['diagnostics']['limiting_ray']['theta_index']=-99
    assert second['target_level']==.01
    assert second['alpha_center_scaled_solution'][0]==1.01
    assert second['two_stage']['stage1_coefficients'][0]==0
    assert second['two_stage']['sampling']['rz'][0,0]==1.
    assert second['two_stage']['boundary']['diagnostics']['limiting_ray']['theta_index']==1
    assert first['score_result'] is results[0][0]


def test_optional_torsion_diagnostic_preserves_the_default_and_can_be_skipped(factory,monkeypatch):
    driver,library,_=factory();calls=[]
    def torsion(lib,result,*arrays):
        calls.append(True)
        result['diagnostics']['coil_torsion_abs_arclength_p95']=7.
        return result
    monkeypatch.setattr(module.native,'_with_torsion',torsion)
    default=driver.evaluate_many(make_coils(),configurations())
    assert len(calls)==3 and all(r['diagnostics']['coil_torsion_abs_arclength_p95']==7. for r,s in default)
    calls.clear()
    minimal=driver.evaluate_many(make_coils(),configurations(),capture=True,include_torsion=False)
    assert not calls and all('coil_torsion_abs_arclength_p95' not in r['diagnostics'] for r,s in minimal)
    assert [r['score'] for r,s in default]==[r['score'] for r,s in minimal]


@pytest.mark.parametrize('version,expected_flag,backend',[(None,1,'legacy_host_filter'),(0,1,'legacy_host_filter'),(1,4,'native')])
def test_source_only_capture_uses_capability_and_skips_python_factor_exports(factory,version,expected_flag,backend):
    driver,library,_=factory()
    if version is not None:
        library.sgpu_source_capture_version=Function(lambda:version)
    results=driver.evaluate_many(make_coils(),configurations(),capture='sources',include_torsion=False)
    assert library.flags==[expected_flag,0] and library.factor_count==0
    assert library.snapshot_count==3 and driver._context is None
    first,second=results[0][1],results[1][1]
    assert 'psi_upper' not in first and 'psi_scale' not in first
    assert first['two_stage']['stage1_upper'] is None
    assert first['metadata']['capture_kind']=='sources'
    assert first['metadata']['source_capture_backend']==backend
    assert first['psi_coefficients'] is second['psi_coefficients']
    assert first['alpha_coefficients'][0]==pytest.approx(1.03)
    assert second['alpha_coefficients'][0]==pytest.approx(1.01)


def test_source_capture_rejects_reverse_diagnostic_mix_before_native_calls(factory):
    driver,library,_=factory()
    with pytest.raises(ValueError,match='source-only'):
        driver.evaluate_many(make_coils(),configurations(),capture='sources',diagnostics=True)
    assert not library.calls and not library.flags


def test_optional_alpha_basis_setter_records_the_selected_basis(factory):
    driver,library,_=factory();calls=[]
    library.sgpu_two_stage_set_alpha_radial_basis_v1=Function(
        lambda context,mode:calls.append((context.value,mode)) or 0)
    base=configurations()[0]
    config=SimpleNamespace(**{**base.to_dict(),'alpha_radial_basis':'flux_legendre'},
                           native_options=base.native_options)
    result,snapshot=driver.evaluate(make_coils(),config,capture=True)
    assert calls==[(37,1)]
    assert result['diagnostics']['alpha_radial_basis']=='flux_legendre'
    assert snapshot['config']['alpha_radial_basis']=='flux_legendre'


def test_new_alpha_basis_explains_missing_native_support_and_closes(factory):
    driver,library,_=factory();base=configurations()[0]
    config=SimpleNamespace(**{**base.to_dict(),'alpha_radial_basis':'flux_legendre'},
                           native_options=base.native_options)
    with pytest.raises(RuntimeError,match='selected alpha radial basis; rebuild'):
        driver.evaluate(make_coils(),config)
    assert driver._context is None and library.flags==[0,0]


@pytest.mark.parametrize('field,value',[('psi_a',.2),('psi_refit_points',8),('alpha_iterations',2),('fast',True),('h',2e-4)])
def test_configs_may_only_change_target_volume_and_fail_before_native_begin(factory,field,value):
    driver,library,_=factory();configs=configurations()
    configs[1]=replace(configs[1],**{field:value})
    with pytest.raises(ValueError,match='only target_volume'):
        driver.evaluate_many(make_coils(),configs)
    assert library.calls==[] and library.flags==[] and driver._context is None


@pytest.mark.parametrize('failure',['begin','fit','resume'])
def test_c_api_failure_always_closes_allocated_context_and_restores_capture_flags(factory,failure):
    driver,library,_=factory(fail=failure)
    with pytest.raises(RuntimeError,match='mock C ABI failure'):
        driver.evaluate_many(make_coils(),configurations(),capture=True)
    assert [value for name,value in library.calls if name=='close']==[37]
    assert driver._context is None and library.flags==[1,0]
    assert driver.last_many_timing['completed_volumes']==0


def test_python_preparation_failure_always_closes_context(factory,monkeypatch):
    driver,library,_=factory()
    def fail(*args,**kwargs):raise ValueError('mock boundary failure')
    monkeypatch.setattr(module,'max_closed_boundary',fail)
    with pytest.raises(ValueError,match='boundary failure'):
        driver.evaluate_many(make_coils(),configurations(),capture=True)
    assert [value for name,value in library.calls if name=='close']==[37]
    assert library.flags==[1,0] and driver._context is None


def test_snapshot_copy_failure_still_closes_context_and_restores_capture(factory,monkeypatch):
    driver,library,_=factory()
    def fail(*args,**kwargs):raise RuntimeError('mock snapshot copy failure')
    monkeypatch.setattr(module,'copy_reverse_snapshot',fail)
    with pytest.raises(RuntimeError,match='snapshot copy failure'):
        driver.evaluate_many(make_coils(),configurations(),capture=True)
    assert [value for name,value in library.calls if name=='close']==[37]
    assert library.flags==[1,0] and driver._context is None


def test_failed_formal_volume_is_retained_without_skipping_following_volumes(factory):
    driver,library,_=factory(statuses={.01:3})
    results=driver.evaluate_many(make_coils(),configurations(),capture=True)
    assert [r['status'] for r,s in results][0::2]==['ok','ok']
    assert results[1][0]['status']!='ok' and results[1][1] is None
    assert library.snapshot_count==2
    assert [value for name,value in library.calls if name=='resume']==[.03,.01,.04]
    assert sum(name=='fit' for name,_ in library.calls)==1


def test_common_stage1_failure_returns_independent_failures_without_formal_calls(factory):
    driver,library,geometry=factory(fail='stage1_status')
    results=driver.evaluate_many(make_coils(),configurations(),capture=True)
    assert len(results)==3 and all(r['status']!='ok' and s is None for r,s in results)
    assert not any(name in ('fit','resume','close') for name,value in library.calls)
    assert geometry==[] and library.factor_count==0
    results[0][0]['diagnostics']['score']=-99
    assert results[1][0]['diagnostics']['score']!=-99
    assert library.flags==[1,0] and driver._context is None


def test_existing_single_evaluate_returns_one_raw_snapshot_pair(factory):
    driver,library,_=factory()
    result,snapshot=driver.evaluate(make_coils(),configurations()[0],capture=True)
    assert result['status']=='ok' and result['score']==.03 and snapshot['config']['target_volume']==.03
    assert driver.last_many_timing['query_count']==1
    assert result['timing']['two_stage']['total']==driver.last_many_timing['total_s']
    assert sum(name=='fit' for name,value in library.calls)==1


def test_single_evaluate_keeps_config_protocol_without_requiring_to_dict(factory):
    driver,library,_=factory()
    config=configurations()[0]
    legacy=SimpleNamespace(helicity=config.helicity,psi_refit_points=config.psi_refit_points,
                           native_options=config.native_options)
    result,snapshot=driver.evaluate(make_coils(),legacy)
    assert result['status']=='ok' and snapshot is None


def test_grouped_results_and_volume_capture_equal_independent_evaluations(factory):
    driver,library,_=factory()
    grouped=driver.evaluate_many(make_coils(),configurations(),capture=True)
    for config,(result,snapshot) in zip(configurations(),grouped):
        independent,_,_=factory()
        reference,reference_snapshot=independent.evaluate(make_coils(),config,capture=True)
        assert {k:v for k,v in result.items() if k!='timing'}=={
            k:v for k,v in reference.items() if k!='timing'}
        for name in ('config','metadata'):
            assert snapshot[name]==reference_snapshot[name]
        for name in ('axis_samples','target_level','psi_coefficients','alpha_coefficients',
                     'alpha_center_scaled_solution','psi_scale','psi_upper'):
            np.testing.assert_array_equal(snapshot[name],reference_snapshot[name])


def test_empty_multi_volume_call_does_not_touch_native_api(factory):
    driver,library,_=factory()
    assert driver.evaluate_many(None,[])==[]
    assert library.calls==[] and library.flags==[] and driver.last_many_timing['query_count']==0
