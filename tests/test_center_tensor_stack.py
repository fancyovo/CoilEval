import warnings
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval._autograd.two_stage_front import _stack_live


def test_mixed_owned_host_and_live_tensor_centers():
    host=np.ones((3,4));host.flags.writeable=False
    live=torch.ones((3,4),dtype=torch.float64,requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        result=_stack_live([host,live],device='cpu',dtype=torch.float64)
    assert not caught
    gradient,=torch.autograd.grad(result.square().sum(),live)
    torch.testing.assert_close(gradient,torch.full_like(live,2.))
    result[0,0,0]=9.
    assert host[0,0]==1.
