"""Selected field rows retain their graph and independent query cotangents."""
import pytest
torch=pytest.importorskip('torch')

from coil_eval._autograd import batch_field


def test_selected_field_values_and_all_input_vjps_match_scalar_queries(monkeypatch):
    generator=torch.Generator().manual_seed(128)
    positions=torch.randn(5,7,3,generator=generator,dtype=torch.float64,requires_grad=True)
    elements=torch.randn(5,7,3,generator=generator,dtype=torch.float64,requires_grad=True)
    points=torch.randn(2,11,3,generator=generator,dtype=torch.float64,requires_grad=True)
    library=object();field=batch_field.BatchSegmentField(positions,elements,library)
    # An independent smooth scalar field operator gives a closed-form
    # nonzero derivative for segment positions, elements and moving points.
    # The real native operator is covered by GPU field regressions.
    class Operator:
        @staticmethod
        def apply(p,w,q,lib,gradient,potential):
            delta=q[:,:,None,:]-p[:,None,:,:]
            scalar=(delta.square()+.4).sum(-1).reciprocal()
            return (scalar[...,None]*w[:,None,:,:]).sum(2)
    monkeypatch.setattr(batch_field,'_BatchField',Operator)
    selected=field.select_queries([4,1])
    assert selected.library is library
    actual=selected(points)
    expected=torch.cat([Operator.apply(positions[i:i+1],elements[i:i+1],points[j:j+1],library,False,False)
                        for j,i in enumerate([4,1])])
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual.square().sum(),(positions,elements,points),retain_graph=True)
    ge=torch.autograd.grad(expected.square().sum(),(positions,elements,points))
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for gradient in ga[:2]:
        assert torch.count_nonzero(gradient[[0,2,3]])==0
        assert bool(torch.all(torch.linalg.vector_norm(gradient[[4,1]],dim=(1,2))>0))
    assert torch.linalg.vector_norm(ga[2])>0


def test_replaced_subset_has_area_vjp_only_on_selected_queries():
    x=torch.arange(12,dtype=torch.float64).reshape(4,3).requires_grad_()
    indices=torch.tensor([1,3])
    stokes=x.square()
    area=3*x.index_select(0,indices)
    hybrid=stokes.index_copy(0,indices,area)
    expected=2*x.detach();expected[indices]=3
    actual,=torch.autograd.grad(hybrid.sum(),x)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('indices',[[],[[0]],[True,False],[0,0],[-1],[4],[.5]])
def test_invalid_field_selection_rejected(indices):
    field=batch_field.BatchSegmentField(torch.zeros(4,2,3),torch.ones(4,2,3),object())
    with pytest.raises(ValueError):field.select_queries(indices)
