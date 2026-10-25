"""Pure numerical and ownership checks for the optional surface adapter."""
from dataclasses import replace
import numpy as np
import pytest
from coil_eval.simsopt import AlphaFit,SourcePsi,SurfaceSeed,StandardSurface,volume_average_fc
from coil_eval.simsopt.clebsch import build_modes,radial,fit_alpha
from coil_eval.simsopt.fourier import fit_correction
from coil_eval.simsopt.geometry import SurfaceProjector
from coil_eval.simsopt.reference import analyze_arrays,surface_fc
from coil_eval.simsopt.solver import StandardSimsoptSolver


def source():
    axis=np.column_stack((np.ones(64),np.zeros((64,3))))
    return SourcePsi(np.array([4.]),np.array([0]),np.array([2]),np.array([0]),np.array([0]),
                     axis,4,.3,.36)


def test_source_elliptical_roots_and_owned_arrays():
    model=source();xyz,radii,stats=model.surface_points(order=12)
    rr=np.hypot(xyz[:,:,0],xyz[:,:,1])-1
    ss=(rr/.3)**2+4*(xyz[:,:,2]/.3)**2
    np.testing.assert_allclose(ss,.36,atol=1e-11)
    assert stats['maximum_s_residual']<1e-11
    assert radii.min()>=.09-1e-12 and radii.max()<=.18+1e-12
    with pytest.raises(ValueError):model.coefficients[0]=9


def test_source_snapshot_keeps_selected_level_and_nfp():
    model=source()
    snap={'nfp':np.array([4]),'config':{'psi_a':.3},'target_level':np.array(.36),
          'axis_samples':model.axis_samples,'psi_coefficients':model.coefficients,
          **{'psi_mode_'+k:getattr(model,'mode_'+k) for k in ('a','b','m','kind')}}
    copied=SourcePsi.from_snapshot(snap)
    assert copied.level==model.level
    np.testing.assert_array_equal(copied.coefficients,model.coefficients)
    with pytest.raises(ValueError):replace(copied,kind='alpha')


def test_gpu_initializer_resolves_packaged_library_only_when_not_explicit(monkeypatch):
    from coil_eval import _native_loader
    from coil_eval.simsopt import AlphaNuInitializer,InitializerConfig
    calls=[]
    def resolve():calls.append(True);return 'packaged-native.so'
    monkeypatch.setattr(_native_loader,'resolve_library',resolve)
    cpu=AlphaNuInitializer(lambda x:x)
    assert cpu.library is None and not calls
    gpu=AlphaNuInitializer(lambda x:x,config=InitializerConfig(device='cuda'))
    assert gpu.library=='packaged-native.so' and len(calls)==1
    explicit=AlphaNuInitializer(lambda x:x,config=InitializerConfig(device='cuda'),library='custom.so')
    assert explicit.library=='custom.so' and len(calls)==1


@pytest.mark.parametrize('regularization',[0.,1e-5])
def test_nu_fft_matches_direct_projection_and_manufactured_derivative(regularization):
    nfp=4;iota=1.31
    phi=np.arange(65)[:,None]/(nfp*65)
    theta=np.arange(67)[None,:]/67
    target=1.7*np.cos(2*np.pi*(3*theta-2*nfp*phi))+.3*np.sin(2*np.pi*(2*theta+nfp*phi))+.05
    fft=fit_correction(target,iota=iota,nfp=nfp,order=4,regularization=regularization)
    direct=fit_correction(target,iota=iota,nfp=nfp,order=4,regularization=regularization,method='direct')
    np.testing.assert_allclose(fft.cos_coeffs,direct.cos_coeffs,rtol=1e-12,atol=3e-15)
    np.testing.assert_allclose(fft.sin_coeffs,direct.sin_coeffs,rtol=1e-12,atol=3e-15)
    if not regularization:
        _,_,_,along=fft.evaluate(theta,phi)
        np.testing.assert_allclose(along,target-.05,atol=4e-14)


def test_nu_true_resonance_is_omitted_explicitly():
    phi=np.arange(31)[:,None]/(4*31);theta=np.arange(33)[None,:]/33
    target=np.cos(2*np.pi*(2*theta-4*phi))
    result=fit_correction(target,iota=2,nfp=4,order=4)
    assert result.diagnostics['skipped_resonance_count']>0
    np.testing.assert_allclose(result.evaluate(theta,phi)[3],0,atol=2e-14)


def test_native_alpha_snapshot_coefficient_scaling():
    snap={'alpha_mode_l':np.array([2]),'alpha_mode_m':np.array([2]),
          'alpha_mode_n':np.array([1]),'alpha_mode_kind':np.array([0]),
          'alpha_center_scaled_solution':np.array([.7,2.4]),'alpha_scale':np.array([7.,2.]),
          'alpha_iota_coefficients':np.array([1.2]),'nfp':4}
    fit=AlphaFit.from_snapshot(snap)
    np.testing.assert_allclose(fit.lambda_coefficients,[.1])
    rho=np.array([.2,.5,.9]);theta=np.array([.1,.4,.7]);phi=np.array([.03,.07,.11])
    phase=2*theta-4*phi
    lam,dt,dp=fit.evaluate(rho,theta,phi)
    np.testing.assert_allclose(lam,.1*rho**2*np.cos(phase),atol=1e-16)
    np.testing.assert_allclose(dt,-.2*rho**2*np.sin(phase),atol=1e-16)
    np.testing.assert_allclose(dp,.4*rho**2*np.sin(phase),atol=1e-16)


def test_dense_alpha_qr_recovers_manufactured_coefficients():
    pytest.importorskip('torch')
    rng=np.random.default_rng(317)
    count=100;mode=np.asarray(build_modes(2,2,1))
    alpha=AlphaFit(mode,rng.normal(0,.01,len(mode)),np.array([1.21]),4)
    coords={'rho':rng.uniform(.1,1,count),'theta':rng.uniform(-np.pi,np.pi,count),
            'phi':rng.uniform(0,np.pi/2,count),'cross_theta':rng.normal(size=(count,3)),
            'cross_phi':rng.normal(size=(count,3))}
    _,dt,dp=alpha.evaluate(coords['rho'],coords['theta'],coords['phi'])
    B=(1+dt)[:,None]*coords['cross_theta']+(dp-alpha.iota(1))[:,None]*coords['cross_phi']
    fitted=fit_alpha(coords,B,nfp=4,orders=(2,2,1),device='cpu',precision='fp64')
    np.testing.assert_allclose(fitted.lambda_coefficients,alpha.lambda_coefficients,atol=2e-15)
    np.testing.assert_allclose(fitted.iota_coefficients,alpha.iota_coefficients,atol=2e-15)


@pytest.mark.parametrize('cached_geometry',[False,True])
def test_surface_projection_cache_reuses_basis_without_sharing_fitted_state(monkeypatch,cached_geometry):
    from coil_eval.simsopt import geometry
    design=np.random.default_rng(7).normal(size=(24,4))
    base=np.ones((4,2,3))*.03
    class FakeSurface:
        def __init__(self):self.coefficients=np.zeros(4)
        def get_dofs(self):return self.coefficients
        def set_dofs(self,values):self.coefficients=np.asarray(values).copy()
        def gamma(self):return base+(design@self.coefficients).reshape(base.shape)
        def dgamma_by_dcoeff(self):return design.reshape(4,2,3,4)
    monkeypatch.setattr(geometry,'surface_from_dofs',lambda *a,**k:FakeSurface())
    projector=SurfaceProjector(cached_geometry=cached_geometry)
    original=np.array([.1,.2,-.3,.4]);second=-original
    one,_=projector.fit(base+(design@original).reshape(base.shape),nfp=4,order=1)
    two,stats=projector.fit(base+(design@second).reshape(base.shape),nfp=4,order=1)
    np.testing.assert_allclose(one.get_dofs(),original,atol=2e-16)
    np.testing.assert_allclose(two.get_dofs(),second,atol=2e-16)
    assert stats['projection_factor_reused'] and projector.hits==1 and projector.misses==1


@pytest.mark.parametrize('device',['cpu','cpu:0'])
def test_geometry_basis_cache_matches_affine_surface_tangents_and_bounds_storage(monkeypatch,device):
    from coil_eval.simsopt import geometry
    created=[]
    class FakeSurface:
        def __init__(self,options):
            created.append(options);self.coefficients=np.zeros(4)
            self.shape=(options['nphi'],options['ntheta'],3)
            self.base=np.ones(self.shape)*(.01*options['nfp']+options['offset'][0])
            rng=np.random.default_rng(options['nphi']*100+options['ntheta'])
            self.maps={key:rng.normal(size=(*self.shape,4)) for key in ('gamma','gammadash1','gammadash2')}
        def get_dofs(self):return self.coefficients
        def set_dofs(self,coefficients):self.coefficients=np.asarray(coefficients)
        def value(self,name):return self.base+np.einsum('...k,k->...',self.maps[name],self.coefficients)
        def gamma(self):return self.value('gamma')
        def gammadash1(self):return self.value('gammadash1')
        def gammadash2(self):return self.value('gammadash2')
        def dgamma_by_dcoeff(self):return self.maps['gamma']
        def dgammadash1_by_dcoeff(self):return self.maps['gammadash1']
        def dgammadash2_by_dcoeff(self):return self.maps['gammadash2']
    monkeypatch.setattr(geometry,'surface_from_dofs',lambda dofs,**options:FakeSurface(options))
    cache=geometry.SurfaceGeometryCache(device=device,method='cached_basis',max_bytes=6000)
    parameters=dict(nfp=4,order=1,nphi=4,ntheta=2,derivatives=True)
    one=cache.evaluate(np.array([.2,-.3,.1,.4]),**parameters)
    two=cache.evaluate(np.array([-.2,.3,-.1,-.4]),**parameters)
    assert len(created)==1 and cache.hits==1 and cache.misses==1
    for name in one:np.testing.assert_allclose(one[name]+two[name],.08,atol=8e-16)
    cache.evaluate(np.zeros(4),**dict(parameters,nfp=2))
    cache.evaluate(np.zeros(4),**dict(parameters,nfp=3))
    assert cache.bytes<=6000 and len(cache._cache)==2
    assert cache.misses==3
    with pytest.raises(ValueError,match='coefficient count'):cache.evaluate(np.ones(3),**parameters)
    cache.clear();assert cache.bytes==0 and not cache._cache


def test_native_normalized_reference_units_and_angular_identity():
    shape=(4,5);count=np.prod(shape);iota=1.3;G=2.1
    gp=np.tile([1.,-iota,0.],(count,1));gt=np.tile([0.,1.,0.],(count,1))
    B=np.tile([G,0.,0.],(count,1));gradient=np.zeros((count,3,3))
    gradient[:,0,:]=np.array([.4,.2,-.1])
    kwargs=dict(xyz=np.zeros((*shape,3)),gphi=gp,gtheta=gt,B=B,dBdx=gradient,iota=iota,G_turn=G,nfp=4)
    result=analyze_arrays(**kwargs)
    expected=abs(((iota-4)*.2-.4)/(2*np.pi*G*np.sqrt(17)))
    assert result['surface_fC_flux_shell_rms']==pytest.approx(expected,abs=1e-16)
    assert result['differential_minus_angular_shell_rms']<1e-16
    scaled=analyze_arrays(**dict(kwargs,B=3*B,dBdx=3*gradient,G_turn=3*G))
    assert scaled['surface_fC_flux_shell_rms']==pytest.approx(expected,abs=1e-16)


def test_volume_aggregation_matches_linear_squared_shell_profile():
    volume=np.linspace(0,.4,7);fc=np.sqrt(.01+.2*volume)
    result=volume_average_fc(volume,fc)
    np.testing.assert_allclose(result['volume_fC_rms'],np.sqrt(.01+.1*volume),atol=1e-16)
    with pytest.raises(ValueError,match='inner'):volume_average_fc(volume[1:],fc[1:])
    explicit=volume_average_fc(volume[1:],fc[1:],inner_fc=fc[0])
    np.testing.assert_allclose(explicit['volume_fC_rms'],result['volume_fC_rms'])
    with pytest.raises(ValueError):volume_average_fc(volume[::-1],fc[::-1])


def test_alpha_only_and_rejected_surface_provenance_are_rejected(tmp_path):
    seed=SurfaceSeed(np.ones(3),4,1,1.3,2.1,.2,.1,0.)
    seed.save(tmp_path/'seed.npz');copied=SurfaceSeed.read(tmp_path/'seed.npz')
    np.testing.assert_array_equal(seed.dofs,copied.dofs)
    with pytest.raises(ValueError,match='alpha_nu'):
        StandardSimsoptSolver().solve(replace(seed,kind='alpha'),object())
    rejected=StandardSurface(seed,np.ones(3),1.3,2.1,False,'rejected',{})
    with pytest.raises(ValueError,match='rejected'):rejected.save(tmp_path/'bad.npz')
    with pytest.raises(ValueError,match='accepted'):surface_fc(rejected,object())


def test_callable_simsopt_field_uses_field_methods_instead_of_optimizer_call(monkeypatch):
    from coil_eval.simsopt import reference,solver
    class CallableField:
        def __call__(self,*args):raise AssertionError('Optimizable call changes DOFs')
        def set_points(self,points):self.points=points
        def B(self):return np.tile([2.,0.,0.],(len(self.points),1))
        def dB_by_dX(self):return np.zeros((len(self.points),3,3))
    class FakeSurface:
        def gamma(self):return np.zeros((4,4,3))
        def gammadash1(self):return np.broadcast_to([1.,0.,0.],(4,4,3))
        def gammadash2(self):return np.broadcast_to([0.,1.,0.],(4,4,3))
        def volume(self):return .05
    monkeypatch.setattr(reference,'surface_from_dofs',lambda *a,**k:FakeSurface())
    seed=SurfaceSeed(np.ones(3),4,1,1.3,2.1,.2,.1,0.)
    standard=StandardSurface(seed,np.ones(3),1.3,2.1,True,'strict_pass',{})
    field=CallableField()
    result=surface_fc(standard,field,sizes=(4,))
    assert result['surface_fC_flux_shell_rms']==0
    assert solver._field_values(field,np.zeros((4,4,3))).shape==(4,4,3)


def test_physical_volume_preserves_clockwise_initializer_orientation():
    from coil_eval.simsopt.solver import _oriented_target_volume
    assert _oriented_target_volume(-.397,.4)==-.4
    assert _oriented_target_volume(.397,.4)==.4
    assert _oriented_target_volume(-.397,None)==-.397
    with pytest.raises(ValueError):_oriented_target_volume(0.,.4)
    with pytest.raises(ValueError):_oriented_target_volume(-.397,-.4)


def test_resident_toroidal_inverse_matches_manufactured_map_and_reference():
    from coil_eval.simsopt.fourier import ToroidalCorrection,invert_toroidal_map
    correction=ToroidalCorrection(np.array([[1,-1],[2,1],[0,1]]),
        np.array([.002,.001,-.003]),np.array([.001,-.002,.001]),1.27,4,{})
    phi,theta=np.meshgrid(np.linspace(0,.25,11,endpoint=False),
                         np.linspace(0,1,13,endpoint=False),indexing='ij')
    alpha=theta-correction.iota*phi
    nu,*_=correction.evaluate(theta,phi)
    target=phi+nu
    reference,stats=invert_toroidal_map(correction,alpha,target,method='reference')
    # cpu:0 executes the resident tensor implementation without a CUDA device.
    resident,tensor_stats=invert_toroidal_map(correction,alpha,target,device='cpu:0')
    np.testing.assert_allclose(reference,phi,atol=8e-17)
    np.testing.assert_allclose(resident,reference,atol=8e-17)
    assert stats['inversion_residual_max_turns']<1e-15
    assert tensor_stats['mapping_jacobian_min']==pytest.approx(stats['mapping_jacobian_min'],abs=1e-15)
