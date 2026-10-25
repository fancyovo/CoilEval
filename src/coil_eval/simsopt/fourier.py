"""Uniform-grid nu projection and its independent direct-sum reference."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

TWOPI = 2*np.pi


@lru_cache(maxsize=16)
def modes(mpol, ntor):
    return np.asarray([(0,n) for n in range(1,ntor+1)] +
                      [(m,n) for m in range(1,mpol+1) for n in range(-ntor,ntor+1)],dtype=int)


@dataclass
class ToroidalCorrection:
    modes: np.ndarray
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    iota: float
    nfp: int
    diagnostics: dict

    def evaluate(self, theta, phi, *, device='cpu'):
        theta,phi=np.broadcast_arrays(theta,phi)
        shape=theta.shape
        mm,nn=self.modes.T
        if device=='cpu':
            phase=TWOPI*(theta.ravel()[:,None]*mm-self.nfp*phi.ravel()[:,None]*nn)
            cosine,sine=np.cos(phase),np.sin(phase)
            nu=cosine@self.cos_coeffs+sine@self.sin_coeffs
            derivative=-sine*self.cos_coeffs+cosine*self.sin_coeffs
            dt=derivative@(TWOPI*mm)
            dp=derivative@(-TWOPI*self.nfp*nn)
        else:
            import torch
            def tensor(value):return torch.as_tensor(value,device=device,dtype=torch.float64)
            m,n,c,s=map(tensor,(mm,nn,self.cos_coeffs,self.sin_coeffs))
            phase=TWOPI*(tensor(theta.ravel())[:,None]*m-self.nfp*tensor(phi.ravel())[:,None]*n)
            co,si=torch.cos(phase),torch.sin(phase)
            derivative=-si*c+co*s
            nu=(co@c+si@s).cpu().numpy()
            dt=(derivative@(TWOPI*m)).cpu().numpy()
            dp=(derivative@(-TWOPI*self.nfp*n)).cpu().numpy()
        return tuple(x.reshape(shape) for x in (nu,dt,dp,dp+self.iota*dt))


def invert_toroidal_map(correction, alpha, phi_b, *, device='cpu', method='resident',
                        maxiter=20, tol=1e-13):
    """Invert phi_B=phi+nu at fixed alpha=theta-iota*phi.

    The resident CUDA path retains the Newton state and Fourier coefficients
    on the device. The reference path evaluates the original full derivatives.
    Both use the same double precision Newton equations and stopping rule.
    """
    if method not in ('resident','reference'):
        raise ValueError("inversion method must be 'resident' or 'reference'")
    alpha, phi_b = np.broadcast_arrays(alpha, phi_b)
    if method=='reference' or device=='cpu':
        old=phi_b.copy()
        for iteration in range(1,maxiter+1):
            theta=alpha+correction.iota*old
            nu,_,_,along=correction.evaluate(theta,old,device=device)
            residual=old+nu-phi_b
            if np.min(1+along)<=0:raise ValueError('noninvertible toroidal correction')
            old-=residual/(1+along)
            if np.max(abs(residual))<tol:break
        nu,_,_,along=correction.evaluate(alpha+correction.iota*old,old,device=device)
        inversion=float(np.max(abs(old+nu-phi_b)))
        minimum,maximum=float(np.min(1+along)),float(np.max(1+along))
    else:
        import torch
        def tensor(value):return torch.as_tensor(value,device=device,dtype=torch.float64)
        m,n=correction.modes.T
        frequency=tensor(TWOPI*(m*correction.iota-n*correction.nfp))
        base=tensor(alpha.ravel())[:,None]*tensor(TWOPI*m)
        c,s=tensor(correction.cos_coeffs),tensor(correction.sin_coeffs)
        target=tensor(phi_b.ravel());old=target.clone()
        def values(phi):
            phase=base+phi[:,None]*frequency
            co,si=torch.cos(phase),torch.sin(phase)
            return co@c+si@s,(-si*c+co*s)@frequency
        for iteration in range(1,maxiter+1):
            nu,along=values(old);residual=old+nu-target;jacobian=1+along
            error,minimum=torch.stack((residual.abs().max(),jacobian.min())).cpu().numpy()
            if minimum<=0:raise ValueError('noninvertible toroidal correction')
            old-=residual/jacobian
            if error<tol:break
        nu,along=values(old)
        inversion,minimum,maximum=torch.stack(((old+nu-target).abs().max(),
                                               (1+along).min(),(1+along).max())).cpu().numpy()
        old=old.cpu().numpy().reshape(phi_b.shape)
    if inversion>1e-10:raise ValueError('toroidal coordinate inversion did not converge')
    return old,{'newton_iterations':iteration,'inversion_residual_max_turns':float(inversion),
                'mapping_jacobian_min':float(minimum),'mapping_jacobian_max':float(maximum),
                'inversion_method':method if device!='cpu' else 'reference'}


def fit_correction(target, *, iota, nfp, order=12, regularization=0.0,
                   resonance_tolerance=1e-10, method='fft'):
    """Fit D nu=target on the unshifted one-period uniform tensor grid.

    ``method='direct'`` preserves the independent mode-by-mode projection.
    Both methods keep the same resonance omission and regularization.
    """
    target=np.asarray(target,dtype=float)
    if target.ndim!=2 or min(target.shape)<=2*order or not np.isfinite(target).all():
        raise ValueError('nu projection needs a finite grid with both sizes >2*order')
    if regularization<0 or nfp<1:
        raise ValueError('invalid nu regularization or field periods')
    mode=modes(int(order),int(order))
    mm,nn=mode.T
    centered=target-target.mean()
    if method=='fft':
        coefficients=np.fft.fft2(centered)/centered.size
        selected=coefficients[(-nn)%target.shape[0], mm%target.shape[1]]
        tc,ts=2*selected.real,-2*selected.imag
    elif method=='direct':
        phi=np.arange(target.shape[0])[:,None]/(nfp*target.shape[0])
        theta=np.arange(target.shape[1])[None,:]/target.shape[1]
        tc,ts=[],[]
        for m,n in mode:
            phase=TWOPI*(m*theta-n*nfp*phi)
            tc.append(2*np.mean(centered*np.cos(phase)))
            ts.append(2*np.mean(centered*np.sin(phase)))
        tc,ts=np.asarray(tc),np.asarray(ts)
    else:
        raise ValueError("nu method must be 'fft' or 'direct'")
    divisor=TWOPI*(mm*iota-nn*nfp)
    active=abs(divisor)>resonance_tolerance
    weight=1+mm*mm+(nn*nfp)**2
    denominator=divisor**2+regularization*weight**2
    cc=np.divide(-divisor*ts,denominator,out=np.zeros_like(ts),where=active)
    ss=np.divide(divisor*tc,denominator,out=np.zeros_like(tc),where=active)
    result=ToroidalCorrection(mode,cc,ss,float(iota),int(nfp),
        {'method':method,'skipped_resonance_count':int(np.sum(~active)),
         'min_mode_divisor':float(np.min(abs(divisor))),'target_mean':float(target.mean())})
    return result
