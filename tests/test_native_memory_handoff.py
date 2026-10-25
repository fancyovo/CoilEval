from contextlib import nullcontext
import pytest

torch=pytest.importorskip('torch')
from coil_eval._autograd.cuda_memory import prepare_native_memory


@pytest.mark.parametrize('free,reserved,allocated,release', [
    (1,30,1,True), (12,18,1,False), (1,29,29,False), (4,27,1,False),
])
def test_handoff_preserves_live_allocations_and_uses_device(monkeypatch,free,reserved,allocated,release):
    gib=1024**3; calls=[]
    def memory_info(device):
        assert device=='cuda:2'
        return free*gib,32*gib
    monkeypatch.setattr(torch.cuda,'mem_get_info',memory_info)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device: reserved*gib)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device: allocated*gib)
    monkeypatch.setattr(torch.cuda,'device',lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda: calls.append('release_inactive_cache'))
    result=prepare_native_memory('cuda:2')
    assert result['cache_released']==release
    assert calls==(['release_inactive_cache'] if release else [])
