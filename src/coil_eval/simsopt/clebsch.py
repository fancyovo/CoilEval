"""Clebsch alpha data, factored evaluation and optional dense QR fitting."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import math
import time
import numpy as np
from .._alpha_basis import radial_value


@lru_cache(maxsize=16)
def build_modes(radial_order, poloidal_order, toroidal_order):
    return tuple((l,m,n,kind) for m in range(min(radial_order,poloidal_order)+1)
                 for n in (range(1,toroidal_order+1) if m==0 else range(-toroidal_order,toroidal_order+1))
                 for l in range(m,radial_order+1,2) for kind in (0,1))


def radial(rho,l,m,basis='zernike'):
    return radial_value(np.asarray(rho),int(l),int(m),basis)


@dataclass
class AlphaFit:
    modes: np.ndarray
    lambda_coefficients: np.ndarray
    iota_coefficients: np.ndarray
    nfp: int
    diagnostics: dict=field(default_factory=dict)
    radial_basis: str='zernike'

    def __post_init__(self):
        if self.radial_basis not in ('zernike','flux_legendre'):
            raise ValueError('alpha radial basis must be zernike or flux_legendre')

    @classmethod
    def from_snapshot(cls,snapshot):
        mode=np.column_stack([snapshot['alpha_mode_'+k] for k in ('l','m','n','kind')]).astype(int)
        if 'alpha_coefficients' in snapshot:
            coefficients=np.asarray(snapshot['alpha_coefficients'])[:len(mode)]
        else:
            coefficients=(np.asarray(snapshot['alpha_center_scaled_solution'])/
                          np.asarray(snapshot['alpha_scale']))[:len(mode)]
        basis=snapshot.get('config',{}).get('alpha_radial_basis','zernike')
        if 'alpha_radial_basis_mode' in snapshot:
            captured=int(np.asarray(snapshot['alpha_radial_basis_mode']).reshape(()))
            if captured not in (0,1) or ('zernike','flux_legendre')[captured]!=basis:
                raise ValueError('captured alpha basis differs from snapshot configuration')
        return cls(mode,coefficients.copy(),np.asarray(snapshot['alpha_iota_coefficients']).copy(),
                   int(np.asarray(snapshot['nfp']).reshape(())),{'origin':'native_capture',
                   'theta_convention':'negative geometric polar angle',
                   'rho_convention':'sqrt(signed_flux / signed_edge_flux)'},radial_basis=basis)

    @classmethod
    def read(cls,path,*,nfp):
        with np.load(path,allow_pickle=False) as z:
            kind=np.asarray(z['mode_kind'])
            if kind.dtype.kind in ('U','S','O'):kind=(kind.astype(str)=='sin').astype(int)
            mode=np.column_stack([z['mode_'+k] for k in ('l','m','n')]+[kind])
            basis=str(np.asarray(z['radial_basis']).item()) if 'radial_basis' in z else 'zernike'
            return cls(mode,z['lambda_coeffs'].copy(),z['iota_coeffs'].copy(),nfp,
                       {'origin':'saved_alpha_fit'},radial_basis=basis)

    def iota(self,rho):
        rho=np.asarray(rho)
        return sum(c*(rho*rho)**i for i,c in enumerate(self.iota_coefficients))

    def evaluate(self,rho,theta,phi,*,device='cpu'):
        """Contract radial and angular factors rather than repeating each mode."""
        rho,theta,phi=np.broadcast_arrays(rho,theta,phi)
        shape=rho.shape
        if not len(self.modes):return tuple(np.zeros(shape) for _ in range(3))
        radial_keys=sorted(set((int(l),int(m)) for l,m,n,k in self.modes))
        phase_keys=sorted(set((int(m),int(n)) for l,m,n,k in self.modes))
        ri={key:i for i,key in enumerate(radial_keys)};pi={key:i for i,key in enumerate(phase_keys)}
        weights=np.zeros((2,len(radial_keys),len(phase_keys)))
        for coefficient,(l,m,n,kind) in zip(self.lambda_coefficients,self.modes):
            weights[int(kind),ri[int(l),int(m)],pi[int(m),int(n)]]+=coefficient
        mm,nn=np.asarray(phase_keys).T
        result=[np.empty(rho.size) for _ in range(3)]
        for start in range(0,rho.size,16384):
            end=min(start+16384,rho.size)
            rr=np.stack([radial(rho.ravel()[start:end],l,m,self.radial_basis) for l,m in radial_keys],axis=1)
            if device=='cpu':
                cc,ss=rr@weights[0],rr@weights[1]
                phase=theta.ravel()[start:end,None]*mm-self.nfp*phi.ravel()[start:end,None]*nn
                co,si=np.cos(phase),np.sin(phase)
                derivative=-cc*si+ss*co
                values=((cc*co+ss*si).sum(1),(derivative*mm).sum(1),(-self.nfp*derivative*nn).sum(1))
            else:
                import torch
                def tensor(v):return torch.as_tensor(v,device=device,dtype=torch.float64)
                r,wc,ws,m,n=map(tensor,(rr,weights[0],weights[1],mm,nn))
                cc,ss=r@wc,r@ws
                phase=tensor(theta.ravel()[start:end])[:,None]*m-self.nfp*tensor(phi.ravel()[start:end])[:,None]*n
                co,si=torch.cos(phase),torch.sin(phase)
                derivative=-cc*si+ss*co
                values=[v.cpu().numpy() for v in ((cc*co+ss*si).sum(1),(derivative*m).sum(1),(-self.nfp*derivative*n).sum(1))]
            for output,value in zip(result,values):output[start:end]=value
        return tuple(value.reshape(shape) for value in result)


def fit_alpha(coordinates,B,*,nfp,orders=(12,12,16),iota_degree=0,device='cuda',precision='fp32',
              radial_basis='zernike'):
    """Reference dense alpha QR with caller-owned sampling/flux coordinates.

    Coordinates use cylindrical vector components, clockwise theta and radians.
    The fit keeps the allocator resident for a persistent evaluation worker.
    """
    import torch
    start=time.perf_counter()
    dtype={'fp32':torch.float32,'fp64':torch.float64}[precision]
    def tensor(v):return torch.as_tensor(v,device=device,dtype=dtype)
    mode=np.asarray(build_modes(*map(int,orders)))
    rho,theta,phi=[tensor(coordinates[k]) for k in ('rho','theta','phi')]
    ct,cp=[tensor(coordinates[k]) for k in ('cross_theta','cross_phi')]
    B=tensor(B)
    rhs=(B-ct).reshape(-1,1)
    design=torch.empty((rhs.numel(),len(mode)+iota_degree+1),device=device,dtype=dtype)
    radial_cache={};phase_cache={}
    for j,(l,m,n,kind) in enumerate(mode):
        key=int(l),int(m)
        if key not in radial_cache:
            radial_cache[key]=radial_value(rho,*key,radial_basis)
        key=int(m),int(n)
        if key not in phase_cache:
            phase=m*theta-n*nfp*phi;phase_cache[key]=(torch.cos(phase),torch.sin(phase))
        co,si=phase_cache[key];rr=radial_cache[int(l),int(m)]
        dt,dp=((-m*rr*si,n*nfp*rr*si) if kind==0 else (m*rr*co,-n*nfp*rr*co))
        design[:,j]=(dt[:,None]*ct+dp[:,None]*cp).reshape(-1)
    for power in range(iota_degree+1):
        design[:,len(mode)+power]=(-(rho**(2*power))[:,None]*cp).reshape(-1)
    scales=torch.linalg.vector_norm(design,dim=0).clamp_min(1e-30)
    design/=scales
    solution=torch.linalg.lstsq(design,rhs,driver='gels').solution[:,0]
    residual=design@solution-rhs[:,0]
    coefficients=(solution/scales).detach().cpu().numpy()
    diagnostics={'origin':'dense_qr','point_count':len(rho),'columns':design.shape[1],
                 'residual_rms':float(torch.sqrt(torch.mean(residual**2))),
                 'seconds':time.perf_counter()-start,'precision':precision}
    return AlphaFit(mode,coefficients[:len(mode)],coefficients[len(mode):],int(nfp),diagnostics,
                    radial_basis=radial_basis)
