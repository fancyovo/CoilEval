"""Vacuum Boozer toroidal correction from the tangential magnetic potential."""
from numbers import Integral

import torch


def fit_nu_potential(A_theta, A_phi, G, nfp, order):
    """Fit ``d_theta nu=A_theta/G``, ``d_phi nu=A_phi/G-1`` in radians.

    ``A_theta=B dot x_theta`` and ``A_phi=B dot x_phi`` are sampled uniformly
    in a straight-field-line poloidal angle and the original toroidal angle,
    with shape ``(nphi,ntheta)``. Phi covers one field period; theta one turn.
    The vacuum assumption is zero net poloidal covariant field, I=0. The
    physical, signed G remains a live scalar. Returned coefficients have the
    same ``(m,k,cosine,sine)`` format as ``boozer.fourier_values``.

    Curl-free B pulls back to the closed surface one-form
    ``A_theta*dtheta + A_phi*dphi = G*dphi + d(chi)``. Thus nu=chi/G up to a
    constant. Each nonzero Fourier coefficient is determined by its full
    gradient, with denominator ``m*m+k*k`` rather than ``k+iota*m``. Resonant
    modes are retained. The constant gauge is zero; angular orders outside
    the retained band are omitted and reported. The Jacobian and invertibility
    of the resulting coordinate map must still be checked by its caller.

    On an approximate/nonclosed sampled one-form this is its Fourier gradient
    least-squares projection. The residual diagnostics remain meaningful; a
    small along-field transport residual alone would not establish this fit.
    """
    for value, name in ((nfp, 'nfp'), (order, 'order')):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    nfp, order = int(nfp), int(order)
    if (not isinstance(A_theta, torch.Tensor) or not isinstance(A_phi, torch.Tensor)
            or A_theta.dtype not in (torch.float32, torch.float64)
            or A_phi.dtype != A_theta.dtype or A_phi.device != A_theta.device
            or A_theta.ndim != 2 or A_phi.shape != A_theta.shape):
        raise ValueError('covariant fields require matching real 2D torch tensors')
    nphi, ntheta = A_theta.shape
    if (min(nphi, ntheta) < 5 or nphi % 2 == 0 or ntheta % 2 == 0
            or min(nphi, ntheta) <= 2*order):
        raise ValueError('odd angular grids must resolve all retained nu modes')
    if not bool(torch.isfinite(A_theta).all() & torch.isfinite(A_phi).all()):
        raise ValueError('covariant fields must be finite')
    G = torch.as_tensor(G, dtype=A_theta.dtype, device=A_theta.device)
    if G.numel() != 1 or not bool(torch.isfinite(G).all()) or bool((G == 0).any()):
        raise ValueError('G must be a finite nonzero scalar')
    G = G.reshape(())
    target_t, target_p = A_theta/G, A_phi/G-1
    kt = torch.fft.fftfreq(ntheta, d=1/ntheta, device=G.device, dtype=G.dtype)
    kp = torch.fft.fftfreq(nphi, d=1/nphi, device=G.device, dtype=G.dtype)
    k, m = torch.meshgrid(kp*nfp, kt, indexing='ij')
    st = torch.fft.fft2(target_t, norm='forward')
    sp = torch.fft.fft2(target_p, norm='forward')
    nonzero = (m != 0) | (k != 0)
    band = (m.abs() <= order) & ((k/nfp).abs() <= order)
    active = band & nonzero
    denominator = torch.where(nonzero, m*m+k*k, torch.ones_like(m))
    coefficients = torch.where(active, -1j*(m*st+k*sp)/denominator,
                               torch.zeros_like(st))
    reconstructed_t = torch.fft.ifft2(1j*m*coefficients, norm='forward').real
    reconstructed_p = torch.fft.ifft2(1j*k*coefficients, norm='forward').real
    curl = torch.fft.ifft2(1j*(k*st-m*sp), norm='forward').real
    half = band & ((m > 0) | ((m == 0) & (k > 0)))
    data = (m[half], k[half], 2*coefficients.real[half], -2*coefficients.imag[half])
    diagnostics = dict(
        nu_potential_derivative_residual_rms=(
            (reconstructed_t-target_t).square()+
            (reconstructed_p-target_p).square()).mean().sqrt(),
        nu_potential_target_derivative_rms=(target_t.square()+target_p.square()).mean().sqrt(),
        nu_potential_curl_rms=curl.square().mean().sqrt(),
        nu_potential_I_over_G=target_t.mean(),
        nu_potential_toroidal_mean_error=target_p.mean(),
        nu_potential_omitted_derivative_l2=(
            st[~band].abs().square().sum()+sp[~band].abs().square().sum()).sqrt(),
        nu_potential_retained_mode_count=active.sum())
    return data, diagnostics
