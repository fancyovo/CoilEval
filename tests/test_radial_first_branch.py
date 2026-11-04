"""Physical radial coordinates select the first axis-connected simple root."""
import json
import math
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from coil_eval._autograd.front import _RadialRoot


def solve(p,level,maximum=1.):
    return _RadialRoot.apply(p,level,1.,maximum,20,1e-12)


@pytest.mark.parametrize('coefficients,level,expected',[
    # Quadratic initialization is exactly the second root, with negative slope.
    ([0,0,1,0,4,0,-8],.5,2**(-.75)),
    # It can also be exactly a later root with POSITIVE slope: checking only
    # the derivative at the result would still choose the disconnected branch.
    ([0,0,1,0,8,0,-16,0,8],1.,math.sqrt(.5)),
])
def test_first_root_precedes_negative_and_later_positive_branches(coefficients,level,expected):
    p=torch.tensor(coefficients,dtype=torch.float64,requires_grad=True)
    target=torch.tensor(level,dtype=torch.float64,requires_grad=True)
    root=solve(p,target)
    torch.testing.assert_close(root,root.new_tensor(expected),rtol=1e-10,atol=1e-12)
    dp,dl=torch.autograd.grad(root,(p,target))
    powers=torch.arange(len(p),dtype=p.dtype)
    slope=(p[1:]*powers[1:]*root**(powers[1:]-1)).sum()
    torch.testing.assert_close(dl,1/slope)
    torch.testing.assert_close(dp,-root**powers/slope)


def test_all_twelve_frozen_wrong_branch_rays_choose_the_inner_root():
    fixture=json.loads((Path(__file__).parent/'fixtures/radial_first_branch.json').read_text())
    p=torch.tensor([r['polynomial'] for r in fixture['rays']],dtype=torch.float64)
    expected=torch.tensor([r['expected_first_root'] for r in fixture['rays']],dtype=torch.float64)
    result=solve(p,p.new_tensor(fixture['level']))
    torch.testing.assert_close(result,expected,rtol=1e-10,atol=1e-12)


def test_later_root_cannot_rescue_exceeded_axis_connected_capacity():
    p=torch.tensor([0,0,1,0,8,0,-16,0,8],dtype=torch.float64)
    with pytest.raises(ValueError,match='axis-connected capacity'):
        solve(p,p.new_tensor(1.2),maximum=2.)


def test_flat_stationary_point_does_not_terminate_a_monotone_branch():
    # psi'=2*r*(r-.5)^2; the zero at .5 has no sign change.
    p=torch.tensor([0,0,.25,-2/3,.5],dtype=torch.float64)
    expected=p.new_tensor(.8)
    target=(p*expected**torch.arange(len(p))).sum()
    torch.testing.assert_close(solve(p,target),expected,rtol=1e-10,atol=1e-12)
