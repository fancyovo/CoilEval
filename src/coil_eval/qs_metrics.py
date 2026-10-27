"""Fourier and triple-product QS metrics on uniform Boozer angle grids.

The last two axes are ``(phi_B, theta_B)``. Angles are radians, covering one
field period and one poloidal turn without repeated endpoints. This module
accepts NumPy arrays or PyTorch tensors; importing it does not import PyTorch.
"""
from __future__ import annotations

from operator import index
import sys

import numpy as np


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if positive and result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _conventions(nfp, helicity):
    nfp = _integer(nfp, "nfp", positive=True)
    try:
        if len(helicity) != 2:
            raise ValueError
        M, N = (_integer(v, "helicity component") for v in helicity)
    except (TypeError, ValueError) as exc:
        raise ValueError("helicity must contain two integers (M, N)") from exc
    if M == 0 and N == 0:
        raise ValueError("helicity (0, 0) does not specify a symmetry")
    return nfp, M, N


def _frequency_arrays(nphi, ntheta, nfp):
    # FFT uses exp(+i*(m*theta + k*nfp*phi)); the full-torus mode n is -k*nfp.
    k = np.rint(np.fft.fftfreq(nphi) * nphi).astype(np.int64)
    m = np.rint(np.fft.fftfreq(ntheta) * ntheta).astype(np.int64)
    return np.broadcast_arrays(m[None, :], (nfp * k)[:, None])


def boozer_spectrum_metrics(B, *, nfp, helicity):
    """Compute Fourier f_B and spectrum diagnostics with one forward FFT.

    B is a positive field-magnitude NumPy array or Torch tensor of shape
    ``(...,nphi,ntheta)``. Odd angular grids (at least five samples each) cover
    one field period in phi and one poloidal turn in theta, without repeated
    endpoints. Helicity ``(M,N)`` uses the full-torus toroidal mode number N.
    The returned f_B is the RMS asymmetric Fourier energy fraction. This
    operation needs no covariant field components, angular derivatives or
    normalization length. Torch inputs retain their local parameter gradients.
    """
    nfp, M, N = _conventions(nfp, helicity)
    torch = sys.modules.get('torch')
    tensor = torch is not None and isinstance(B, torch.Tensor)
    if tensor:
        if B.dtype not in (torch.float32, torch.float64):
            raise ValueError('B tensor must have float32 or float64 dtype')
        xp = torch
    else:
        B = np.asarray(B)
        if B.dtype.kind not in 'fiu':
            raise ValueError('B must be a real numeric array')
        B = B.astype(np.result_type(B.dtype,np.float64),copy=False)
        xp = np
    if B.ndim < 2:
        raise ValueError('B needs angular axes (nphi, ntheta)')
    nphi, ntheta = B.shape[-2:]
    if min(nphi,ntheta) < 5 or nphi % 2 == 0 or ntheta % 2 == 0:
        raise ValueError('B needs odd angular grid sizes >= 5')
    if not bool(xp.isfinite(B).all()) or not bool((B > 0).all()):
        raise ValueError('B must contain finite positive field magnitudes')
    shape = B.shape[:-2]
    m, k = _frequency_arrays(nphi,ntheta,nfp)
    asymmetric = m*N+k*M != 0
    tail = ((np.abs(m) >= (ntheta-1)/3) | (np.abs(k)/nfp >= (nphi-1)/3))
    if tensor:
        asymmetric = torch.as_tensor(asymmetric,device=B.device)
        tail = torch.as_tensor(tail,device=B.device)
        dc = torch.zeros((nphi,ntheta),device=B.device,dtype=B.dtype)
        norm = lambda value: torch.linalg.vector_norm(value.reshape(*shape,-1),dim=-1)
    else:
        dc = np.zeros((nphi,ntheta),dtype=B.dtype)
        norm = lambda value: np.linalg.norm(value.reshape(*shape,-1),axis=-1)
    dc[0,0] = 1
    offset = B[...,:1,:1]
    spectrum = xp.fft.fft2(B-offset,norm='forward')+offset*dc
    total = norm(spectrum)
    f_b = norm(spectrum*asymmetric)/total
    return {'f_B':f_b,'f_B_squared':f_b*f_b,'spectrum':spectrum,
            'spectral_tail_fraction':(norm(spectrum*tail)/total)**2}


def boozer_qs_metrics(B, *, nfp, helicity, iota, G, major_radius, I=0.0):
    """Evaluate dimensionless ``f_B`` and ``f_T`` on each supplied surface.

    Parameters
    ----------
    B : numpy.ndarray or torch.Tensor
        Positive field magnitude, in tesla, with shape ``(..., nphi, ntheta)``.
        Both angular sizes must be odd and at least five, avoiding the ambiguous
        first derivative of an even-grid Nyquist mode. The grids start at zero
        with spacings ``2*pi/(nfp*nphi)`` and ``2*pi/ntheta``. A constant angular
        origin shift changes Fourier phases but leaves both metrics invariant.
    nfp : int
        Number of field periods.
    helicity : (int, int)
        Symmetry ``M*theta_B - N*phi_B``; N is the full-torus mode number.
    iota, G, I, major_radius : scalar or array-like
        Broadcastable to the leading surface dimensions of B. G and I are
        covariant components for *radian* angles, in tesla metres, and
        ``major_radius`` is the explicitly selected normalization length in
        metres. I defaults to zero for vacuum fields. ``G + iota*I`` must be
        nonzero and the normalization length positive. Torch parameters retain
        their parameter derivatives when B is a tensor.

    Returns
    -------
    dict
        ``f_B`` is sqrt(non-symmetric Fourier energy / all Fourier energy).
        ``f_T`` is the flux-shell RMS of ``R**2*T/B**4``, where
        ``T = (grad(psi) x grad(B)) . grad(B_vector . grad(B))`` and psi is
        toroidal flux divided by 2*pi. Flux-shell weights are ``abs(J)`` with
        ``J=(G+iota*I)/B**2``. ``f_B_squared`` and ``f_T_squared`` are useful
        smooth scalar objectives. Their leading dimensions match those of B.
        ``f_T_raw`` (tesla**4/metre**2), ``f_T_normalized``, ``jacobian`` and the complex
        ``spectrum`` retain the angular grid. ``spectral_tail_fraction`` reports
        the fraction of total Fourier energy in the outer third of either
        resolved frequency range, without rejecting a surface.

    Notes
    -----
    All resolved modes are retained. The triple product uses spectral first
    and second derivatives of B and evaluates the resulting products pointwise.
    Algebraic cancellation of derivatives of J avoids differentiating an
    aliased sampled product. Grid refinement is still required to assess
    unresolved modes and quadrature of nonlinear products. These calculations
    assume a valid Boozer parameterization; they do not establish its quality.
    The Torch norm chooses the zero subgradient at an exactly zero residual.
    """
    nfp, M, N = _conventions(nfp, helicity)
    torch = sys.modules.get("torch")
    tensor = torch is not None and isinstance(B, torch.Tensor)
    if tensor:
        if B.dtype not in (torch.float32, torch.float64):
            raise ValueError("B tensor must have float32 or float64 dtype")
        xp = torch
    else:
        B = np.asarray(B)
        if B.dtype.kind not in "fiu":
            raise ValueError("B must be a real numeric array")
        B = B.astype(np.result_type(B.dtype, np.float64), copy=False)
        xp = np
    if B.ndim < 2:
        raise ValueError("B needs angular axes (nphi, ntheta)")
    nphi, ntheta = B.shape[-2:]
    if min(nphi, ntheta) < 5 or nphi % 2 == 0 or ntheta % 2 == 0:
        raise ValueError("B needs odd angular grid sizes >= 5")
    if not bool(xp.isfinite(B).all()) or not bool((B > 0).all()):
        raise ValueError("B must contain finite positive field magnitudes")
    shape = B.shape[:-2]

    def parameter(value, name, *, positive=False):
        if tensor:
            value = torch.as_tensor(value, dtype=B.dtype, device=B.device)
        else:
            value = np.asarray(value, dtype=B.dtype)
        try:
            value = xp.broadcast_to(value, shape)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must broadcast to surface shape {shape}") from exc
        if not bool(xp.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        if positive and not bool((value > 0).all()):
            raise ValueError(f"{name} must be positive")
        return value[..., None, None]

    iota = parameter(iota, "iota")
    G = parameter(G, "G")
    I = parameter(I, "I")
    radius = parameter(major_radius, "major_radius", positive=True)
    H = G + iota * I
    if not bool(xp.isfinite(H).all()) or bool((H == 0).any()):
        raise ValueError("G + iota*I must be finite and nonzero")

    m, k = _frequency_arrays(nphi, ntheta, nfp)
    asymmetric = m * N + k * M != 0
    tail = ((np.abs(m) >= (ntheta - 1) / 3)
            | (np.abs(k) / nfp >= (nphi - 1) / 3))
    if tensor:
        asymmetric = torch.as_tensor(asymmetric, device=B.device)
        tail = torch.as_tensor(tail, device=B.device)
        m = torch.tensor(m.copy(), dtype=B.dtype, device=B.device)
        k = torch.tensor(k.copy(), dtype=B.dtype, device=B.device)
        # Subtract a constant before the FFT and restore only its DC mode.
        # A spatially uniform field then has exactly zero residual instead of
        # a tiny FFT leakage with an arbitrary norm gradient at the origin.
        offset = B[..., :1, :1]
        dc = torch.zeros((nphi, ntheta), dtype=B.dtype, device=B.device)
        dc[0, 0] = 1
        spectrum = torch.fft.fft2(B-offset, norm="forward") + offset*dc
        factors = torch.stack((1j*m, 1j*k, -m*m, -m*k, -k*k))
        derivatives = torch.fft.ifft2(spectrum[..., None, :, :] * factors,
                                     norm="forward").real
        norm = lambda value: torch.linalg.vector_norm(value.reshape(*shape, -1), dim=-1)
        sum_angles = lambda value: value.sum(dim=(-2, -1))
    else:
        offset = B[..., :1, :1]
        dc = np.zeros((nphi, ntheta), dtype=B.dtype)
        dc[0, 0] = 1
        spectrum = np.fft.fft2(B-offset, norm="forward") + offset*dc
        factors = np.stack((1j*m, 1j*k, -m*m, -m*k, -k*k))
        derivatives = np.fft.ifft2(spectrum[..., None, :, :] * factors,
                                  norm="forward").real
        norm = lambda value: np.linalg.norm(value.reshape(*shape, -1), axis=-1)
        sum_angles = lambda value: value.sum(axis=(-2, -1))
    bt, bp, btt, btp, bpp = (derivatives[..., j, :, :] for j in range(5))
    bracket = bt * (bpp + iota*btp) - bp * (btp + iota*btt)
    jacobian = H / (B*B)
    raw = bracket / (jacobian*jacobian)
    normalized = radius*radius * bracket / (H*H)
    weights = xp.abs(jacobian)
    total_spectrum_norm = norm(spectrum)
    f_b = norm(spectrum * asymmetric) / total_spectrum_norm
    f_t = norm(normalized * xp.sqrt(weights)) / xp.sqrt(sum_angles(weights))
    return {"f_B": f_b, "f_T": f_t,
            "f_B_squared": f_b*f_b, "f_T_squared": f_t*f_t,
            "f_T_raw": raw, "f_T_normalized": normalized,
            "jacobian": jacobian, "spectrum": spectrum,
            "spectral_tail_fraction": (norm(spectrum * tail) / total_spectrum_norm)**2}


def boozer_qs_metrics_field(B_vector, dBdx, x_theta, x_phi, *, nfp,
                           helicity, iota, G, major_radius, I=0.0):
    """Evaluate QS from Cartesian field jets and live Boozer surface tangents.

    This alternative to :func:`boozer_qs_metrics` evaluates the physical scalar
    ``q = B_vector dot grad(|B|)`` from the analytic field Jacobian. Only q is
    differentiated spectrally, once in each angular direction. It avoids
    second angular differences of the sampled field magnitude.

    ``B_vector``, ``x_theta`` and ``x_phi`` have shape
    ``(..., nphi, ntheta, 3)``; ``dBdx[...,i,j]`` is ``d B_i / d x_j``.
    Tangents differentiate the *Boozer angles in radians*, not the original
    geometric or straight-field-line angles. Remaining arguments, grid rules,
    normalizations and return names follow :func:`boozer_qs_metrics`.

    The contraction is ``T=(B_theta*q_phi-B_phi*q_theta)/J`` where
    ``J=(G+iota*I)/|B|**2``. It therefore still assumes a valid Boozer surface.
    On an approximate coordinate fit it need not equal the B-only expression;
    report coordinate residuals and angular grid convergence separately.

    Torch inputs retain derivatives through all field jets and tangents. A
    first parameter gradient requires the field operator's VJP for its spatial
    Jacobian output, including the contraction of second spatial derivatives.
    The native SegmentField operator provides that first-order VJP.
    """
    nfp, M, N = _conventions(nfp, helicity)
    torch = sys.modules.get("torch")
    tensor = torch is not None and isinstance(B_vector, torch.Tensor)
    if tensor:
        if B_vector.dtype not in (torch.float32, torch.float64):
            raise ValueError("B_vector tensor must have float32 or float64 dtype")
        xp = torch
    else:
        B_vector = np.asarray(B_vector)
        if B_vector.dtype.kind not in "fiu":
            raise ValueError("B_vector must be a real numeric array")
        B_vector = B_vector.astype(np.result_type(B_vector.dtype, np.float64), copy=False)
        xp = np
    if B_vector.ndim < 3 or B_vector.shape[-1] != 3:
        raise ValueError("B_vector needs angular axes and three Cartesian components")
    nphi, ntheta = B_vector.shape[-3:-1]
    if min(nphi, ntheta) < 5 or nphi % 2 == 0 or ntheta % 2 == 0:
        raise ValueError("field arrays need odd angular grid sizes >= 5")
    shape = B_vector.shape[:-3]

    def array(value, name, wanted):
        if tensor:
            value = torch.as_tensor(value, dtype=B_vector.dtype, device=B_vector.device)
        else:
            value = np.asarray(value)
            if value.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a real numeric array")
            value = value.astype(B_vector.dtype, copy=False)
        if tuple(value.shape) != tuple(wanted) or not bool(xp.isfinite(value).all()):
            raise ValueError(f"{name} must be finite with shape {tuple(wanted)}")
        return value

    field = array(B_vector, "B_vector", B_vector.shape)
    jac = array(dBdx, "dBdx", (*B_vector.shape, 3))
    xt = array(x_theta, "x_theta", B_vector.shape)
    xp_ = array(x_phi, "x_phi", B_vector.shape)
    if tensor:
        B = torch.linalg.vector_norm(field, dim=-1)
        sum_last = lambda value: value.sum(dim=-1)
        norm = lambda value: torch.linalg.vector_norm(value.reshape(*shape, -1), dim=-1)
        sum_angles = lambda value: value.sum(dim=(-2, -1))
        area = torch.linalg.vector_norm(torch.linalg.cross(xt, xp_), dim=-1)
    else:
        B = np.linalg.norm(field, axis=-1)
        sum_last = lambda value: value.sum(axis=-1)
        norm = lambda value: np.linalg.norm(value.reshape(*shape, -1), axis=-1)
        sum_angles = lambda value: value.sum(axis=(-2, -1))
        area = np.linalg.norm(np.cross(xt, xp_), axis=-1)
    if not bool((B > 0).all()) or not bool((area > 0).all()):
        raise ValueError("field magnitude and surface tangent area must be positive")

    def parameter(value, name, *, positive=False):
        value = (torch.as_tensor(value, dtype=B.dtype, device=B.device) if tensor
                 else np.asarray(value, dtype=B.dtype))
        try:
            value = xp.broadcast_to(value, shape)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must broadcast to surface shape {shape}") from exc
        if not bool(xp.isfinite(value).all()) or (positive and not bool((value > 0).all())):
            raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
        return value[..., None, None]

    iota = parameter(iota, "iota")
    G, I = parameter(G, "G"), parameter(I, "I")
    radius = parameter(major_radius, "major_radius", positive=True)
    H = G + iota*I
    if not bool(xp.isfinite(H).all()) or bool((H == 0).any()):
        raise ValueError("G + iota*I must be finite and nonzero")

    grad_B = xp.einsum("...ij,...i->...j", jac, field) / B[..., None]
    bt, bp = sum_last(grad_B*xt), sum_last(grad_B*xp_)
    q = sum_last(field*grad_B)
    m, k = _frequency_arrays(nphi, ntheta, nfp)
    asymmetric = m*N + k*M != 0
    tail = ((np.abs(m) >= (ntheta-1)/3) | (np.abs(k)/nfp >= (nphi-1)/3))
    if tensor:
        asymmetric = torch.as_tensor(asymmetric, device=B.device)
        tail = torch.as_tensor(tail, device=B.device)
        factors = torch.as_tensor(np.stack((1j*m, 1j*k)), device=B.device)
        factors = factors.to(torch.complex128 if B.dtype == torch.float64 else torch.complex64)
        dc = torch.zeros((nphi, ntheta), dtype=B.dtype, device=B.device)
    else:
        factors = np.stack((1j*m, 1j*k))
        dc = np.zeros((nphi, ntheta), dtype=B.dtype)
    dc[0, 0] = 1
    offset = B[..., :1, :1]
    spectrum = xp.fft.fft2(B-offset, norm="forward") + offset*dc
    q_spectrum = xp.fft.fft2(q-q[..., :1, :1], norm="forward")
    q_derivatives = xp.fft.ifft2(q_spectrum[..., None, :, :]*factors, norm="forward").real
    qt, qp = q_derivatives[..., 0, :, :], q_derivatives[..., 1, :, :]
    jacobian = H/(B*B)
    raw = (bt*qp-bp*qt)/jacobian
    normalized = radius*radius*raw/(B**4)
    weights = xp.abs(jacobian)
    total_norm = norm(spectrum)
    f_b = norm(spectrum*asymmetric)/total_norm
    f_t = norm(normalized*xp.sqrt(weights))/xp.sqrt(sum_angles(weights))
    return {"f_B": f_b, "f_T": f_t, "f_B_squared": f_b*f_b, "f_T_squared": f_t*f_t,
        "f_T_raw": raw, "f_T_normalized": normalized, "jacobian": jacobian,
        "spectrum": spectrum, "spectral_tail_fraction": (norm(spectrum*tail)/total_norm)**2,
        "B_magnitude": B, "B_theta": bt, "B_phi": bp, "parallel_derivative": q,
        "derivative_method": "analytic_field_jacobian_and_first_spectral_q"}
