"""Full-order Fourier quadrature and engineering-derivative layout checks."""
import math
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from coil_eval._autograd import engineering
from coil_eval._autograd.engineering import split_tokens
from coil_eval._autograd.field import coil_segments
from coil_eval._autograd.batch_field import coil_segments_batch


def numpy_segments(tokens,nfp=4,count=64):
    coefficients=(tokens.shape[-1]-1)//3
    order=(coefficients-1)//2
    physical=tokens[:,:-1].reshape(len(tokens),3,coefficients)
    t=(np.arange(count)+.5)/count
    positions=np.broadcast_to(physical[:,:,0,None],(len(tokens),3,count)).copy()
    tangent=np.zeros_like(positions)
    for mode in range(1,order+1):
        angle=2*np.pi*mode*t
        positions+=physical[:,:,2*mode-1,None]*np.sin(angle)+physical[:,:,2*mode,None]*np.cos(angle)
        tangent+=2*np.pi*mode*(physical[:,:,2*mode-1,None]*np.cos(angle)-physical[:,:,2*mode,None]*np.sin(angle))
    positions,tangent=positions.transpose(0,2,1),tangent.transpose(0,2,1)/count
    p,w=[],[]
    for period in range(nfp):
        angle=2*np.pi*period/nfp;c,s=math.cos(angle),math.sin(angle)
        for mirror in (False,True):
            sign=-1 if mirror else 1
            x,y,z=np.moveaxis(positions,-1,0)
            dx,dy,dz=np.moveaxis(tangent,-1,0)
            p.append(np.stack((c*x-s*sign*y,s*x+c*sign*y,sign*z),axis=-1).reshape(-1,3))
            rotated=np.stack((c*dx-s*sign*dy,s*dx+c*sign*dy,sign*dz),axis=-1)
            w.append((rotated*tokens[:,-1,None,None]*sign).reshape(-1,3))
    return np.concatenate(p),np.concatenate(w)


@pytest.mark.parametrize('order',[4,16])
def test_midpoint_quadrature_matches_independent_fourier_sum(order):
    rng=np.random.default_rng(154+order)
    tokens=rng.normal(0,.1,(4,3*(2*order+1)+1));tokens[:,-1]*=1e6
    p,w=coil_segments(torch.tensor(tokens,dtype=torch.float64),count=64)
    expected_p,expected_w=numpy_segments(tokens)
    np.testing.assert_allclose(p.numpy(),expected_p,rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(w.numpy(),expected_w,rtol=3e-13,atol=3e-8)


def test_full16_quadrature_gradient_matches_physical_parameter_differences():
    rng=np.random.default_rng(1601)
    physical=rng.normal(0,.03,(4,100));physical[:,-1]=rng.uniform(5e4,1.5e5,4)
    tokens=torch.tensor(physical,dtype=torch.float64,requires_grad=True)
    p,w=coil_segments(tokens,count=64)
    rp=rng.normal(size=p.shape);rw=rng.normal(size=w.shape)
    objective=(p*torch.tensor(rp)).sum()+(w*torch.tensor(rw)/1e5).sum()
    actual,=torch.autograd.grad(objective,tokens)
    def value(parameters):
        pos,element=numpy_segments(parameters)
        return np.sum(pos*rp)+np.sum(element*rw/1e5)
    for coil in range(4):
        for column in (0,32,33,65,66,98,99):
            step=1e-6*(1e5 if column==99 else 1.)
            plus,minus=physical.copy(),physical.copy()
            plus[coil,column]+=step;minus[coil,column]-=step
            expected=(value(plus)-value(minus))/(2*step)
            assert actual[coil,column].item()==pytest.approx(expected,rel=1e-7,abs=2e-8 if column<99 else 2e-13)


@pytest.mark.parametrize('order',[4,16])
def test_batch_fourier_values_and_query_gradient_isolation(order):
    generator=torch.Generator().manual_seed(554+order)
    tokens=torch.randn((2,4,3*(2*order+1)+1),generator=generator,dtype=torch.float64,requires_grad=True)
    p,w=coil_segments_batch(tokens,count=64)
    for query in range(2):
        expected_p,expected_w=coil_segments(tokens[query],count=64)
        torch.testing.assert_close(p[query],expected_p,rtol=2e-13,atol=2e-13)
        torch.testing.assert_close(w[query],expected_w,rtol=2e-13,atol=2e-13)
    gradient,=torch.autograd.grad(p[1].square().sum()+w[1].square().sum(),tokens)
    assert torch.count_nonzero(gradient[0])==0
    assert torch.linalg.vector_norm(gradient[1])>0


@pytest.mark.parametrize('order',[4,16])
def test_engineering_native_derivative_layout_and_current_column(order,monkeypatch):
    # This closed-form C-ABI stand-in exercises the metric/coordinate/coil/mode
    # layout and reverse contraction. Native metric accuracy is checked on GPU.
    def closed_form(library,x,y,z,current,nfp):
        assert nfp==4 and x.shape==(4,2*order+1) and current.shape==(4,)
        derivatives=np.zeros((6,3,*x.shape))
        derivatives[0]=np.stack((2*x,4*y,6*z))
        derivatives[2]=np.stack((np.cos(x),-2*np.sin(y),3*z*z))
        derivatives[3]=np.stack((y+z,x+z,x+y))
        return {'derivatives':derivatives,'values':{
            'length_mean':np.sum(x*x+2*y*y+3*z*z),
            'curvature_max':np.sum(np.sin(x)+2*np.cos(y)+z**3),
            'min_intercoil_distance':np.sum(x*y+y*z+z*x)}}
    monkeypatch.setattr(engineering.native,'coil_metric_gradient_native',closed_form)
    rng=np.random.default_rng(71+order)
    physical=rng.normal(0,.3,(4,3*(2*order+1)+1));physical[:,-1]*=1e5
    tokens=torch.tensor(physical,dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([.7,-.2,.4],dtype=torch.float64)
    result=engineering._EngineeringMetrics.apply(tokens,'closed_form',4)
    actual,=torch.autograd.grad((result*weights).sum(),tokens)
    assert actual.shape==tokens.shape and torch.count_nonzero(actual[:,-1])==0
    def value(parameters):
        out=closed_form('closed_form',*split_tokens(parameters),4)['values']
        return .7*out['min_intercoil_distance']-.2*out['curvature_max']+.4*out['length_mean']
    for _ in range(10):
        direction=rng.normal(size=physical.shape);direction[:,-1]*=1e5
        step=1e-6
        expected=(value(physical+step*direction)-value(physical-step*direction))/(2*step)
        assert np.sum(actual.numpy()*direction)==pytest.approx(expected,rel=3e-8,abs=3e-8)


@pytest.mark.parametrize('width',[0,1,25,99])
def test_invalid_fourier_blocks_are_rejected(width):
    tokens=torch.zeros((4,width),dtype=torch.float64)
    with pytest.raises(ValueError,match='Fourier'):
        split_tokens(tokens)
    with pytest.raises(ValueError,match='Fourier'):
        coil_segments(tokens)
    with pytest.raises(ValueError,match='Fourier'):
        coil_segments_batch(tokens[None])
