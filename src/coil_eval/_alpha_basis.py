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


def radial_value(rho, l, m, basis='zernike'):
    """Evaluate a radial mode on NumPy arrays or tensors, preserving gradients."""
    coefficients = radial_polynomial(l, m, basis)  # Validate the mode and basis.
    if basis == 'zernike':
        return sum(c * rho**power for power, c in coefficients)
    order = (int(l) - int(m)) // 2
    previous = rho * 0 + 1
    if order == 0:
        return previous
    x = 2 * rho * rho - 1
    current = x
    for j in range(2, order + 1):
        previous, current = current, ((2*j-1)*x*current - (j-1)*previous) / j
    return current
