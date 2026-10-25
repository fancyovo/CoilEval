"""Host capture selection, reuse and optional diagnostics without CUDA."""
import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest

from coil_eval import _binding as native
from coil_eval._autograd import capture


class CaptureField:
    def __init__(self):
        self.arrays={'axis_samples':np.ones((8,4)), 'psi_upper':np.eye(3),
                     'psi_mode_a':np.array([2,3,4]), 'target_level':np.array(.4)}
        self.copied=[]
        self.lib=SimpleNamespace(_full_chain_capture_bound=True,
            sgpu_reverse_capture_array_count=lambda:len(self.arrays),
            sgpu_reverse_capture_array_info=self.info,sgpu_reverse_capture_copy=self.copy)

    def info(self,index,name,capacity,shape,shape_capacity,ndim):
        key=list(self.arrays)[index];array=self.arrays[key]
        name.value=key.encode();ndim._obj.value=array.ndim
        for i,size in enumerate(array.shape):shape[i]=size
        return 0

    def copy(self,index,pointer,capacity):
        key=list(self.arrays)[index];self.copied.append(key)
        np.ctypeslib.as_array(pointer,shape=(capacity,))[:]=self.arrays[key].ravel()
        return 0

    def _check(self,code):
        if code:raise RuntimeError(code)


def test_snapshot_selection_skips_unrequested_matrix_copy_and_keeps_mode_dtype():
    field=CaptureField()
    result=capture.copy_reverse_snapshot(field,names={'psi_mode_a','target_level'})
    assert field.copied==['psi_mode_a','target_level']
    assert result['psi_mode_a'].dtype==np.int64
    assert set(result)=={'psi_mode_a','target_level'}


def test_snapshot_reuses_only_matching_readonly_group_arrays():
    field=CaptureField();upper=np.eye(3);upper.setflags(write=False)
    result=capture.copy_reverse_snapshot(field,shared={'psi_upper':upper})
    assert result['psi_upper'] is upper and 'psi_upper' not in field.copied
    result['target_level'][...]=8.
    assert field.arrays['target_level']==.4


@pytest.mark.parametrize('shape,readonly',[((2,2),True),((3,3),False)])
def test_unsafe_shared_arrays_are_rejected(shape,readonly):
    field=CaptureField();value=np.ones(shape);value.setflags(write=not readonly)
    with pytest.raises(ValueError,match='readonly with shape'):
        capture.copy_reverse_snapshot(field,shared={'psi_upper':value})
    assert 'psi_upper' not in field.copied


@pytest.mark.parametrize('include_torsion',[True,False])
def test_capture_center_passes_torsion_option_and_restores_flags(monkeypatch,include_torsion):
    flags=[];calls=[]
    class Config(ct.Structure):_fields_=[('target_volume',ct.c_double)]
    class Field:
        lib=SimpleNamespace(_full_chain_capture_bound=True,sgpu_set_reverse_capture=lambda value:flags.append(value) or 0)
        _check=staticmethod(lambda code:None)
        def capture_psi_center(self,*args,**kwargs):
            calls.append(kwargs)
            return {'_config_struct':Config(.4),'score_result':{'status':'ok'}}
    monkeypatch.setattr(capture,'copy_reverse_snapshot',lambda field:{})
    capture.capture_center(Field(),*[np.zeros((1,3))]*4,include_torsion=include_torsion)
    assert calls[0]['include_torsion'] is include_torsion and flags==[1,0]


def test_binding_capture_torsion_default_and_opt_out_match_scores(monkeypatch):
    field=native.BatchCoilFieldGpu.__new__(native.BatchCoilFieldGpu)
    field.n_base_coils=1;field.n_coeff=3;field.nfp=4;field.device_id=0
    def score(*args):
        args[8]._obj.status=0
        args[-1]._obj.value=0
        return 0
    field.lib=SimpleNamespace(sgpu_default_score_config=lambda config:0,
        sgpu_score_coils_capture_psi_center=score)
    field._check=lambda code:None
    calls=[]
    monkeypatch.setattr(native,'_with_boundary_iota',lambda lib,result:result)
    def torsion(lib,result,*arrays):
        calls.append(True);result['diagnostics']['optional_torsion']=3.
        return result
    monkeypatch.setattr(native,'_with_torsion',torsion)
    arrays=[np.zeros((1,3))]*3+[np.ones(1)]
    original=field.capture_psi_center(*arrays)['score_result']
    assert len(calls)==1 and original['diagnostics']['optional_torsion']==3.
    calls.clear();minimal=field.capture_psi_center(*arrays,include_torsion=False)['score_result']
    assert not calls and 'optional_torsion' not in minimal['diagnostics']
    assert original['status']==minimal['status']=='ok'
