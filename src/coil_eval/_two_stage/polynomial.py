"""Factor the scalar basis into toroidal amplitudes and radial ray powers."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def _fixed_ray_basis(degree, mtor, nfp, phi_dtype, phi_bytes, theta_dtype, theta_bytes):
    """Cache only coefficient-independent grid factors, preserving input dtype."""
    phis = np.frombuffer(phi_bytes, dtype=np.dtype(phi_dtype))
    theta = np.frombuffer(theta_bytes, dtype=np.dtype(theta_dtype))
    monomials = [(a,total-a) for total in range(2,degree+1) for a in range(total,-1,-1)]
    index = {pair:i for i,pair in enumerate(monomials)}
    phase = phis[:,None]*(nfp*np.arange(1,mtor+1))
    trig = np.column_stack((np.ones(len(phis)),np.stack((np.cos(phase),np.sin(phase)),
                                                    axis=-1).reshape(len(phis),-1)))
    cosine,sine = np.cos(theta),np.sin(theta)
    angular = np.asarray([cosine**a*sine**b for a,b in monomials])
    selections = [[i for i,(a,b) in enumerate(monomials) if a+b==total]
                  for total in range(2,degree+1)]
    trig.setflags(write=False);angular.setflags(write=False)
    return monomials,index,trig,angular,selections


def ray_polynomials(model, phis, theta):
    degree=max([2]+[int(mode.a+mode.b) for mode in model.modes])
    mtor=max([0]+[int(mode.m) for mode in model.modes])
    phis,theta=np.asarray(phis),np.asarray(theta)
    monomials,index,trig,angular,selections=_fixed_ray_basis(degree,mtor,int(model.nfp),
        phis.dtype.str,phis.tobytes(),theta.dtype.str,theta.tobytes())
    coefficients=np.zeros((len(monomials),2*mtor+1))
    coefficients[index[2,0],0]=1.
    for coefficient,mode in zip(model.coeffs,model.modes):
        if mode.kind not in ('cos','sin'):
            raise ValueError('mode.kind must be cos or sin')
        if mode.m==0 and mode.kind=='sin':
            continue
        column=0 if mode.m==0 else 2*int(mode.m)-1+(mode.kind=='sin')
        coefficients[index[int(mode.a),int(mode.b)],column]+=float(coefficient)
    amplitude=trig@coefficients.T
    result=np.zeros((len(phis),len(theta),degree+1))
    for total,select in zip(range(2,degree+1),selections):
        result[:,:,total]=amplitude[:,select]@angular[select,:]
    return result
