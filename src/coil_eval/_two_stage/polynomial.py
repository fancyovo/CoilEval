"""Factor the scalar basis into toroidal amplitudes and radial ray powers."""
import numpy as np


def ray_polynomials(model, phis, theta):
    degree=max([2]+[int(mode.a+mode.b) for mode in model.modes])
    mtor=max([0]+[int(mode.m) for mode in model.modes])
    monomials=[(a,total-a) for total in range(2,degree+1) for a in range(total,-1,-1)]
    index={pair:i for i,pair in enumerate(monomials)}
    coefficients=np.zeros((len(monomials),2*mtor+1))
    coefficients[index[2,0],0]=1.
    for coefficient,mode in zip(model.coeffs,model.modes):
        if mode.kind not in ('cos','sin'):
            raise ValueError('mode.kind must be cos or sin')
        if mode.m==0 and mode.kind=='sin':
            continue
        column=0 if mode.m==0 else 2*int(mode.m)-1+(mode.kind=='sin')
        coefficients[index[int(mode.a),int(mode.b)],column]+=float(coefficient)
    phase=np.asarray(phis)[:,None]*(int(model.nfp)*np.arange(1,mtor+1))
    trig=np.column_stack((np.ones(len(phis)),np.stack((np.cos(phase),np.sin(phase)),axis=-1).reshape(len(phis),-1)))
    amplitude=trig@coefficients.T
    cosine,sine=np.cos(theta),np.sin(theta)
    angular=np.asarray([cosine**a*sine**b for a,b in monomials])
    result=np.zeros((len(phis),len(theta),degree+1))
    for total in range(2,degree+1):
        select=[i for i,(a,b) in enumerate(monomials) if a+b==total]
        result[:,:,total]=amplitude[:,select]@angular[select,:]
    return result
