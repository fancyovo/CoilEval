import numpy as np
import pytest
from coil_eval.simsopt.topology import angular_covering


def geometry(poloidal=1,toroidal=0):
    phi=np.arange(97)*2*np.pi/(4*97)
    theta=np.arange(129)*2*np.pi/129
    angle=poloidal*theta[None,:]+toroidal*4*phi[:,None]
    radius=1.+.1*np.cos(angle)
    xyz=np.stack((radius*np.cos(phi[:,None]),radius*np.sin(phi[:,None]),-.07*np.sin(angle)),axis=-1)
    axis=np.column_stack((np.ones(32),np.zeros((32,3))))
    return xyz,axis


def test_regular_single_cover_with_offsets():
    xyz,axis=geometry()
    result=angular_covering(xyz,axis,4)
    assert result['valid_convention']
    assert result['degrees']['theta_poloidal']['mean']==pytest.approx(1.)


@pytest.mark.parametrize('poloidal,toroidal',[(9,-5),(1,1),(-1,0),(0,0)])
def test_multiple_cover_and_changed_angle_convention_are_rejected(poloidal,toroidal):
    xyz,axis=geometry(poloidal,toroidal)
    result=angular_covering(xyz,axis,4)
    assert not result['valid_convention']
    assert result['degrees']['theta_poloidal']['mean']==pytest.approx(poloidal,abs=1e-12)
    assert result['degrees']['theta_toroidal']['mean']==pytest.approx(toroidal,abs=1e-12)


def test_nonfinite_axis_rejected():
    xyz,axis=geometry();axis[0,0]=np.nan
    with pytest.raises(ValueError):angular_covering(xyz,axis,4)
