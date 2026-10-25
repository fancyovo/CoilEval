"""Radial polynomial conventions shared by reverse and surface interfaces."""
from functools import lru_cache
import math


@lru_cache(maxsize=256)
def radial_polynomial(l, m, basis='zernike'):
    """Return (power, coefficient) pairs for one Clebsch radial mode.

    Flux Legendre uses P_((l-m)/2)(2*rho**2-1), retaining the existing mode
    labels and number of columns while allowing finite angular corrections
    toward the magnetic axis.
    """
    l,m=int(l),int(m)
    if l<0 or m<0 or m>l or (l-m)%2:
        raise ValueError('radial modes require 0 <= m <= l with even l-m')
    if basis=='zernike':
        return tuple((l-2*k,(-1)**k*math.factorial(l-k)/
            (math.factorial(k)*math.factorial((l+m)//2-k)*math.factorial((l-m)//2-k)))
            for k in range((l-m)//2+1))
    if basis=='flux_legendre':
        order=(l-m)//2
        return tuple((2*j,(-1)**(order-j)*math.comb(order,j)*math.comb(order+j,j))
                     for j in range(order+1))
    raise ValueError('alpha_radial_basis must be zernike or flux_legendre')
