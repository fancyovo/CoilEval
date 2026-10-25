"""NumPy representation of the scalar basis and periodic magnetic axis."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PsiMode:
    a: int
    b: int
    m: int
    kind: str


def build_modes(degree=10, mtor=12):
    return [PsiMode(a,total-a,m,kind)
            for total in range(2,int(degree)+1) for a in range(total,-1,-1)
            for m in range(int(mtor)+1) for kind in (("cos",) if m==0 else ("cos","sin"))
            if not (a==2 and total-a==0 and m==0)]


class PsiModel:
    def __init__(self, coefficients, axis_samples, *, nfp=4, a=.15, degree=10, mtor=12):
        self.coeffs=np.ascontiguousarray(coefficients,dtype=np.float64)
        self.axis_samples=np.ascontiguousarray(axis_samples,dtype=np.float64)
        self.modes=build_modes(degree,mtor)
        self.nfp,self.a=int(nfp),float(a)
        if self.coeffs.shape!=(len(self.modes),) or not np.isfinite(self.coeffs).all():
            raise ValueError('scalar coefficients must match the complete finite basis')
        if self.axis_samples.ndim!=2 or self.axis_samples.shape[1]!=4 or len(self.axis_samples)<2 or not np.isfinite(self.axis_samples).all():
            raise ValueError('axis samples require finite [sample,4] R,Z,R_phi,Z_phi')
        if self.nfp<1 or self.nfp!=nfp or not np.isfinite(self.a) or self.a<=0:
            raise ValueError('positive integer nfp and positive scalar scale required')

    def axis_at(self, phi):
        phi=np.asarray(phi,dtype=np.float64)
        n=len(self.axis_samples);period=2*np.pi/self.nfp;h=period/n
        position=np.remainder(phi,period)/h
        i0=np.floor(position).astype(np.int64).clip(0,n-1);i1=(i0+1)%n;t=position-i0
        def interpolate(column):
            y0,y1=self.axis_samples[i0,column],self.axis_samples[i1,column]
            d0,d1=self.axis_samples[i0,column+2],self.axis_samples[i1,column+2]
            value=((2*t-3)*t*t+1)*y0+((t-2)*t+1)*t*h*d0+(-2*t+3)*t*t*y1+(t-1)*t*t*h*d1
            derivative=((6*t-6)*t*y0+(-6*t+6)*t*y1)/h+((3*t-4)*t+1)*d0+(3*t-2)*t*d1
            return value,derivative
        R,Rp=interpolate(0);Z,Zp=interpolate(1)
        return R,Z,Rp,Zp
