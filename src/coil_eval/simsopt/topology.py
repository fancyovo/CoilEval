"""Angular covering checks relative to the source magnetic axis."""
from __future__ import annotations
import numpy as np
from .state import _hermite


def _winding(angle, axis, period=2*np.pi):
    phase=angle*(2*np.pi/period)
    return np.angle(np.exp(1j*(np.roll(phase,-1,axis=axis)-phase))).sum(axis=axis)/(2*np.pi)


def angular_covering(xyz, axis_samples, nfp):
    """Check the single-cover convention used by alpha+nu surface seeds.

    Poloidal angle is clockwise about the source axis. One poloidal parameter
    circuit has degree +1 and a toroidal field period has degree 0 in that
    angle. These sampled checks supplement residual and nondegeneracy checks.
    """
    xyz=np.asarray(xyz,dtype=float);axis=np.asarray(axis_samples,dtype=float)
    if (xyz.ndim!=3 or xyz.shape[-1]!=3 or min(xyz.shape[:2])<3 or
        axis.ndim!=2 or axis.shape[1]!=4 or len(axis)<2 or
        not np.isfinite(axis).all() or not np.isfinite(xyz).all()):
        raise ValueError('covering checks require finite surface points and source axis samples')
    phi=np.arctan2(xyz[...,1],xyz[...,0])
    radius=np.hypot(xyz[...,0],xyz[...,1])
    ar=_hermite(phi,axis[:,0],axis[:,2],nfp)
    az=_hermite(phi,axis[:,1],axis[:,3],nfp)
    distance=np.hypot(radius-ar,xyz[...,2]-az)
    theta=-np.arctan2(xyz[...,2]-az,radius-ar)
    degrees={'theta_poloidal':_winding(theta,1),
             'theta_toroidal':_winding(theta,0),
             'phi_poloidal':_winding(phi,1),
             'phi_toroidal':_winding(phi,0,2*np.pi/nfp)}
    expected={'theta_poloidal':1.,'theta_toroidal':0.,'phi_poloidal':0.,'phi_toroidal':1.}
    matches={k:bool(np.allclose(v,expected[k],rtol=0.,atol=1e-7)) for k,v in degrees.items()}
    separated=bool(np.min(distance)>1e-10)
    return dict(checked=True,valid_convention=all(matches.values()) and separated,
        source_axis_separated=separated,minimum_axis_distance_m=float(np.min(distance)),
        degrees={k:dict(minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()))
                 for k,v in degrees.items()},checks=matches,
        scope='sampled angular covering relative to the source magnetic axis')
