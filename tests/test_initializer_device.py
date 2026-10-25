"""The native extractor and Torch initializer select one CUDA device."""
import numpy as np
import pytest

from coil_eval.simsopt import AlphaNuInitializer, InitializerConfig


@pytest.mark.parametrize('device,device_id,expected_device,expected_id',[
    ('cpu',None,'cpu',0),('cpu:0',2,'cpu:0',2),('cuda',None,'cuda:0',0),
    ('cuda',1,'cuda:1',1),('cuda:2',None,'cuda:2',2),('cuda:2',2,'cuda:2',2),
    ('cuda',np.int64(1),'cuda:1',1),
])
def test_initializer_device_resolution_is_consistent_without_cuda(monkeypatch,device,device_id,expected_device,expected_id):
    from coil_eval import _native_loader
    monkeypatch.setattr(_native_loader,'resolve_library',lambda:'packaged.so')
    config=InitializerConfig(device=device)
    with AlphaNuInitializer(lambda xyz:xyz,config=config,device_id=device_id) as initializer:
        assert initializer.device_id==expected_id
        assert initializer.config.device==expected_device
        assert initializer.projector.device==initializer.geometry.device==expected_device
    assert config.device==device


@pytest.mark.parametrize('device,device_id',[
    ('cuda:1',0),('cuda:0',1),('cuda',-1),('cuda',True),('cuda',1.5),('cuda','1'),
    ('cuda:-1',None),('wrong',0),
])
def test_conflicts_and_invalid_device_ids_fail_before_optional_resource_setup(monkeypatch,device,device_id):
    import coil_eval.simsopt.initializer as module
    calls=[]
    def forbidden(**kwargs):calls.append(kwargs);raise AssertionError('resource setup was reached')
    monkeypatch.setattr(module,'SurfaceProjector',forbidden)
    with pytest.raises(ValueError,match='device'):
        AlphaNuInitializer(lambda xyz:xyz,config=InitializerConfig(device=device),device_id=device_id)
    assert not calls
