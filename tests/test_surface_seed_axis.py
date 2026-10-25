"""Source-axis provenance survives new and historical seed records."""
import numpy as np
import pytest

from coil_eval.simsopt import SurfaceSeed, StandardSurface


def seed(**kwargs):
    return SurfaceSeed(np.ones(3),4,1,1.3,2.1,.2,.1,0.,**kwargs)


def axis():
    values=np.zeros((8,4));values[:,0]=1.
    return values


def test_axis_roundtrip_retains_array_shape_and_owned_readonly_storage(tmp_path):
    original=axis();value=seed(source_axis_samples=original)
    original[0,0]=9.
    assert value.source_axis_samples[0,0]==1. and not value.source_axis_samples.flags.writeable
    path=tmp_path/'seed.npz';value.save(path)
    restored=SurfaceSeed.read(path)
    np.testing.assert_array_equal(restored.source_axis_samples,axis())
    assert restored.source_axis_samples.shape==(8,4)
    assert not np.shares_memory(value.source_axis_samples,restored.source_axis_samples)
    with np.load(path,allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind!='O' for name in archive.files)


def test_legacy_seed_omits_none_and_loads_without_fabricating_an_axis(tmp_path):
    value=seed();path=tmp_path/'legacy.npz';value.save(path)
    with np.load(path,allow_pickle=False) as archive:
        assert 'source_axis_samples' not in archive
        assert all(archive[name].dtype.kind!='O' for name in archive.files)
    restored=SurfaceSeed.read(path)
    assert restored.source_axis_samples is None
    assert restored.iota==value.iota and restored.source_kind==value.source_kind


def test_existing_campaign_extra_axis_field_is_loaded_as_an_array(tmp_path):
    value=seed();data={key:getattr(value,key) for key in value.__dataclass_fields__
                       if key not in ('diagnostics','source_axis_samples')}
    data.update(source_axis_samples=axis(),unrelated_provenance='retained by campaign')
    path=tmp_path/'campaign.npz';np.savez(path,**data)
    restored=SurfaceSeed.read(path)
    np.testing.assert_array_equal(restored.source_axis_samples,axis())


@pytest.mark.parametrize('invalid',[np.ones(4),np.ones((1,4)),np.ones((8,3)),np.full((8,4),np.nan)])
def test_malformed_source_axis_is_rejected(invalid):
    with pytest.raises(ValueError,match='source_axis_samples'):
        seed(source_axis_samples=invalid)


def test_standard_surface_preserves_source_axis_when_available(tmp_path):
    for available in (True,False):
        initial=seed(source_axis_samples=axis() if available else None)
        solved=StandardSurface(initial,np.ones(3),1.3,2.1,True,'strict_pass',{})
        path=tmp_path/f'standard-{available}.npz';solved.save(path)
        with np.load(path,allow_pickle=False) as archive:
            assert ('source_axis_samples' in archive)==available
            if available:np.testing.assert_array_equal(archive['source_axis_samples'],axis())
            assert all(archive[name].dtype.kind!='O' for name in archive.files)
