"""Native-normalized differential QS on standard Simsopt surfaces."""
from __future__ import annotations
import time
import numpy as np
from .geometry import surface_from_dofs
from .state import STANDARD_KIND,INITIALIZER_KIND,require_kind


def _rms(values,weights):
    return float(np.sqrt(np.sum(weights*values**2)/np.sum(weights)))


def analyze_arrays(*,xyz,gphi,gtheta,B,dBdx,iota,G_turn,nfp):
    """Coordinates are turns; dBdx[i,j] is dB_i/dx_j.

    The signed flux variable is toroidal flux/(2*pi), and G_rad=G_turn/(2*pi).
    Surface and whole-volume RMS are returned by separate functions.
    """
    xyz=np.asarray(xyz,dtype=float)
    B=np.asarray(B,dtype=float).reshape(-1,3)
    derivative=np.asarray(dBdx,dtype=float).reshape(-1,3,3)
    gp=np.asarray(gphi,dtype=float).reshape(-1,3);gt=np.asarray(gtheta,dtype=float).reshape(-1,3)
    if B.shape[0]!=xyz.reshape(-1,3).shape[0] or derivative.shape[0]!=len(B):
        raise ValueError('surface and field point counts differ')
    if not all(np.isfinite(v).all() for v in (xyz,B,derivative,gp,gt)) or not np.isfinite(iota):
        raise ValueError('nonfinite surface reference data')
    if not np.isfinite(G_turn) or G_turn==0 or nfp<1:raise ValueError('invalid G_turn or nfp')
    magnitude=np.linalg.norm(B,axis=1)
    cross=np.cross(gt,gp);area=np.linalg.norm(cross,axis=1)
    if np.any(magnitude<=0) or np.any(area<=0):raise ValueError('zero field or degenerate surface normal')
    grad_modB=np.einsum('nij,ni->nj',derivative,B)/magnitude[:,None]
    G_rad=float(G_turn)/(2*np.pi)
    gradpsi=magnitude[:,None]**2/(2*np.pi*G_turn)*cross
    denominator=magnitude**3*np.hypot(1.,nfp)
    first=(iota-nfp)*np.sum(np.cross(B,gradpsi)*grad_modB,axis=1)/denominator
    second=G_rad*np.sum(B*grad_modB,axis=1)/denominator
    residual=first-second
    shell=abs(G_rad)/magnitude**2
    angular=-(np.sum(gp*grad_modB,axis=1)+nfp*np.sum(gt*grad_modB,axis=1))/(
        2*np.pi*magnitude*np.hypot(1.,nfp))
    tangent=gp+iota*gt
    boozer_error=G_turn*B-magnitude[:,None]**2*tangent
    normalized=np.linalg.norm(boozer_error,axis=1)/(abs(G_turn)*magnitude)
    return {'surface_fC_flux_shell_rms':_rms(residual,shell),
            'surface_fC_area_rms':_rms(residual,area),
            'surface_fC_angular_rms':_rms(residual,np.ones_like(shell)),
            'relative_boozer_equation_error_rms':float(np.sqrt(np.mean(normalized**2))),
            'differential_minus_angular_shell_rms':_rms(residual-angular,shell),
            'point_count':len(B),'iota':float(iota),'G_turn':float(G_turn),
            'G_rad':G_rad,'definition':'native-normalized differential local residual, surface flux-shell RMS'}


def surface_fc(surface,field,*,sizes=(64,128)):
    """Evaluate accepted standard surfaces with an independent analytic field.

    ``field`` may be a Simsopt field or a Cartesian callback returning (B,dBdx).
    """
    require_kind(surface.kind,STANDARD_KIND,'reference f_C surface')
    require_kind(surface.seed.kind,INITIALIZER_KIND,'reference f_C source')
    if not surface.accepted:raise ValueError('reference f_C requires an accepted standard surface')
    start=time.perf_counter();rows=[]
    for size in sizes:
        grid=surface_from_dofs(surface.dofs,nfp=surface.seed.nfp,order=surface.seed.order,
            nphi=int(size),stellsym=surface.seed.stellsym,offset=(.5,.5))
        xyz=np.asarray(grid.gamma());flat=np.ascontiguousarray(xyz.reshape(-1,3))
        if hasattr(field,'set_points') and hasattr(field,'B') and hasattr(field,'dB_by_dX'):
            field.set_points(flat);B=field.B();gradient=field.dB_by_dX()
        elif callable(field):
            B,gradient=field(flat)
        else:
            raise TypeError('field must provide Simsopt field methods or return (B,dBdx)')
        rows.append(dict(size=int(size),**analyze_arrays(xyz=xyz,gphi=grid.gammadash1(),
            gtheta=grid.gammadash2(),B=B,dBdx=gradient,iota=surface.iota,G_turn=surface.G,nfp=surface.seed.nfp)))
    if not rows:raise ValueError('at least one reference grid is required')
    result=dict(rows[-1],grids=rows,actual_volume_m3=abs(float(grid.volume())),seconds=time.perf_counter()-start)
    if len(rows)>1:
        result['last_grid_relative_change']=rows[-1]['surface_fC_flux_shell_rms']/max(rows[-2]['surface_fC_flux_shell_rms'],1e-300)-1
    return result


def volume_average_fc(volumes,surface_fc_values,*,inner_fc=None):
    """Integrate shell f_C² over actual enclosed volume by trapezoidal quadrature.

    A value at V=0 or an explicit ``inner_fc`` is required. The latter specifies
    the unresolved inner-layer approximation and is recorded in the result.
    This aggregation assumes the solved layers form one nested branch.
    """
    volumes=np.asarray(volumes,dtype=float);values=np.asarray(surface_fc_values,dtype=float)
    if volumes.ndim!=1 or volumes.shape!=values.shape or len(volumes)<2:
        raise ValueError('volumes and surface f_C must be matching one-dimensional arrays of length>=2')
    if not np.isfinite(volumes).all() or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('nonfinite or negative shell data')
    if np.any(np.diff(volumes)<=0) or volumes[0]<0:raise ValueError('actual volumes must increase strictly')
    supplied_inner=inner_fc
    if volumes[0]>0:
        if inner_fc is None:raise ValueError('the inner unsampled region requires an explicit inner_fc')
        if not np.isfinite(inner_fc) or inner_fc<0:raise ValueError('inner_fc must be finite and nonnegative')
        volumes=np.r_[0.,volumes];values=np.r_[inner_fc,values]
    integrals=np.r_[0.,np.cumsum(np.diff(volumes)*(values[1:]**2+values[:-1]**2)/2)]
    volume_rms=np.sqrt(np.divide(integrals,volumes,out=np.zeros_like(volumes),where=volumes>0))
    volume_rms[volumes==0]=values[volumes==0]
    return {'actual_volumes_m3':volumes,'surface_fC_flux_shell_rms':values,
            'volume_fC_rms':volume_rms,'inner_fc':supplied_inner,
            'quadrature':'trapezoid in actual enclosed volume; requires nested solved branch'}
