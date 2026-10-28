"""Differentiable magnetic axis, psi fit and target-volume level."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Mapping

import torch


def monomials(degree: int = 10):
    return [(a, total - a) for total in range(2, degree + 1)
            for a in range(total, -1, -1)]


@lru_cache(maxsize=32)
def _monomial_layout(degree, device, dtype):
    """Small, coefficient-independent indices; never cache a live graph."""
    modes = monomials(degree)
    # A value-only call may warm this cache under inference_mode. Cached
    # constants must still be usable by a later differentiable call.
    with torch.inference_mode(False):
        ax = torch.tensor([a for a, _ in modes], device=device, dtype=torch.long)
        bz = torch.tensor([b for _, b in modes], device=device, dtype=torch.long)
        previous_x = torch.tensor([max(a-1, 0) for a, _ in modes], device=device, dtype=torch.long)
        previous_y = torch.tensor([max(b-1, 0) for _, b in modes], device=device, dtype=torch.long)
        groups = tuple(torch.tensor([j for j, (a, b) in enumerate(modes) if a+b == k],
                                   device=device, dtype=torch.long) for k in range(2, degree+1))
        return ax, bz, previous_x, previous_y, ax.to(dtype), bz.to(dtype), groups


def _monomial_features(X, Y, degree, scale=None):
    """Cartesian features and their derivatives using shared integer powers.

    Storage remains rows times monomials, independent of the toroidal order.
    Scalar powers and arithmetic order match the per-monomial expressions.
    A tensor scale broadcasts over the point dimensions, before the feature
    dimension is appended (including independent per-query batch scales).
    """
    ax, bz, previous_x, previous_y, nx, ny, _ = _monomial_layout(degree, X.device, X.dtype)
    px = torch.stack([X**k for k in range(degree+1)], -1)
    py = torch.stack([Y**k for k in range(degree+1)], -1)
    x = px.index_select(-1, ax)
    y = py.index_select(-1, bz)
    polynomial = x*y
    derivative_x = (nx*px.index_select(-1, previous_x))*y
    derivative_y = (ny*x)*py.index_select(-1, previous_y)
    if scale is not None:
        denominator = scale[..., None] if isinstance(scale, torch.Tensor) else scale
        derivative_x = derivative_x/denominator
        derivative_y = derivative_y/denominator
    return polynomial, derivative_x, derivative_y


@lru_cache(maxsize=32)
def _ray_denominators(degree, scale, device, dtype):
    with torch.inference_mode(False):
        return torch.tensor([scale**(a+b) for a, b in monomials(degree)], device=device, dtype=dtype)


def native_modes(degree: int = 10, mtor: int = 12):
    return [(a, b, m, kind) for a, b in monomials(degree)
            for m in range(mtor + 1) for kind in range(1 if m == 0 else 2)
            if not (a == 2 and b == 0 and m == 0)]


def _trig(phi, mtor, nfp, derivative=False):
    phase = phi[..., None] * (nfp * torch.arange(1, mtor + 1,
                             device=phi.device, dtype=phi.dtype))
    if derivative:
        frequency = nfp * torch.arange(1, mtor + 1, device=phi.device, dtype=phi.dtype)
        channels = torch.stack((-frequency * phase.sin(), frequency * phase.cos()), -1)
        leading = torch.zeros_like(phi[..., None])
    else:
        channels = torch.stack((phase.cos(), phase.sin()), -1)
        leading = torch.ones_like(phi[..., None])
    return torch.cat((leading, channels.flatten(-2)), -1)


def _coefficient_matrix(coefficients, degree, mtor, fixed_quadratic=0.):
    expected = len(monomials(degree)) * (2 * mtor + 1) - 1
    if coefficients.shape[-1] != expected:
        raise ValueError(f'expected full psi coefficient count {expected}, got {coefficients.shape}')
    fixed = coefficients.new_full((*coefficients.shape[:-1], 1), fixed_quadratic)
    return torch.cat((fixed, coefficients), -1).reshape(
        *coefficients.shape[:-1], len(monomials(degree)), 2 * mtor + 1)


def _tensor_product_column_norms(features, PR, PZ, P, T, Tp):
    """Live data-column norms without the full row-by-mode matrix.

    Accumulate the expanded quadratic form in FP64 to avoid cancellation of
    radial/vertical contributions, then use the operator's solve precision.
    Leading batch dimensions broadcast through the factored GEMMs.
    """
    cr,cz,cp=[value.double() for value in features]
    pr,pz,p,t,tp=[value.double() for value in (PR,PZ,P,T,Tp)]
    u2=pr.square().mT@cr.square()+pz.square().mT@cz.square()+2*(pr*pz).mT@(cr*cz)
    uw=(pr*p).mT@(cr*cp)+(pz*p).mT@(cz*cp)
    w2=p.square().mT@cp.square()
    square=u2@t.square()+2*uw@(t*tp)+w2@tp.square()
    return square.flatten(-2)[...,1:].clamp_min(1e-30).sqrt().to(features[0].dtype)


def _section_column_norms(features, PR, PZ, P, T, Tp):
    """Section-factorized norms, retaining weighted and moving-point terms."""
    cr,cz,cp=[value.double() for value in features]
    pr,pz,p,t,tp=[value.double() for value in (PR,PZ,P,T,Tp)]
    # The O(rows*monomials) factors already bound operator memory. Direct
    # squaring here also avoids cancellation within cr*PR+cz*PZ.
    u=cr[...,None]*pr+cz[...,None]*pz
    w=cp[...,None]*p
    u2=u.square().sum(-2);uw=(u*w).sum(-2);w2=w.square().sum(-2)
    square=u2.mT@t.square()+2*uw.mT@(t*tp)+w2.mT@tp.square()
    return square.flatten(-2)[...,1:].clamp_min(1e-30).sqrt().to(features[0].dtype)


def _frozen_fit_tensor(value, *, device, dtype):
    # Captured NumPy factors can be read-only; copy them instead of exposing
    # their storage through as_tensor. Existing device tensors need no copy.
    if isinstance(value,torch.Tensor):return value.to(device=device,dtype=dtype).detach()
    return torch.tensor(value,device=device,dtype=dtype).detach()


def periodic_hermite(phi, values, derivatives, nfp):
    """Periodic native Hermite interpolation, including derivative of interpolant."""
    phi = torch.as_tensor(phi, device=values.device, dtype=values.dtype)
    period = 2 * math.pi / nfp
    count = values.shape[-1]
    position = torch.remainder(phi, period) * count / period
    i0 = torch.floor(position).long().clamp_max(count - 1)
    i1 = (i0 + 1) % count
    t = position - i0
    h = period / count
    y0, y1 = values[..., i0], values[..., i1]
    d0, d1 = derivatives[..., i0], derivatives[..., i1]
    value = ((2*t - 3)*t*t + 1)*y0 + ((t - 2)*t + 1)*t*h*d0
    value = value + (-2*t + 3)*t*t*y1 + (t - 1)*t*t*h*d1
    deriv = ((6*t - 6)*t*y0 + (-6*t + 6)*t*y1)/h
    deriv = deriv + ((3*t - 4)*t + 1)*d0 + (3*t - 2)*t*d1
    return value, deriv


def cylindrical_rhs(field: Callable, rz, phi):
    phi = torch.as_tensor(phi, device=rz.device, dtype=rz.dtype)
    c, s = phi.cos(), phi.sin()
    R, Z = rz.unbind(-1)
    xyz = torch.stack((R*c, R*s, Z), -1)
    B = field(xyz.reshape(-1, 3), gradient=False).reshape(xyz.shape).to(rz.dtype)
    br = B[..., 0]*c + B[..., 1]*s
    bp = -B[..., 0]*s + B[..., 1]*c
    floor = torch.where(bp < 0, -torch.ones_like(bp), torch.ones_like(bp))*1e-14
    denominator = torch.where(bp.abs() > 1e-14, bp, floor)
    return torch.stack((R*br/denominator, R*B[..., 2]/denominator), -1)


def implicit_axis_start(trace: Callable, captured_start, nfp: int, steps: int,
                        *, condition_limit=1e10):
    """Attach the periodic-root derivative to a native-selected axis root.

    y=P(y,x), so dy/dx=(I-P_y)^-1 P_x.  The captured root's value is preserved;
    it is only a branch seed, not a frozen physical variable.  Two reverse
    contractions obtain the small 2x2 monodromy matrix without differences.
    The returned diagnostics distinguish root conditioning from root residual.
    """
    y = captured_start.detach().reshape(1, 2).requires_grad_(True)
    end = trace(y, 0., 2*math.pi/nfp, steps, return_samples=False)
    jac = torch.stack([torch.autograd.grad(end[0, k], y, retain_graph=True)[0][0]
                       for k in range(2)])
    matrix = torch.eye(2, device=y.device, dtype=y.dtype) - jac.detach()
    condition = torch.linalg.cond(matrix)
    if not bool(torch.isfinite(condition) & (condition < condition_limit)):
        raise ValueError('periodic magnetic-axis root has singular/ill-conditioned monodromy')
    correction = torch.linalg.solve(matrix, (end-end.detach())[0])
    attached = y.detach()[0] + correction
    return attached, dict(monodromy=jac.detach(), condition=condition.detach(),
                          root_residual=torch.linalg.vector_norm(end-y).detach())


def differentiable_axis(field: Callable, trace: Callable, captured_start, *,
                        nfp=4, trace_steps=240, sample_count=240):
    start, diagnostics = implicit_axis_start(trace, captured_start, nfp, trace_steps)
    substeps = (trace_steps + sample_count - 1)//sample_count
    actual_steps = sample_count*substeps
    states = trace(start[None], 0., 2*math.pi/nfp, actual_steps, return_samples=True)
    if states.shape != (actual_steps+1, 1, 2):
        raise ValueError('trace(return_samples=True) must include every step and the initial point')
    rz = states[:-1:substeps, 0]
    phi = torch.arange(sample_count, device=rz.device, dtype=rz.dtype)*(2*math.pi/(nfp*sample_count))
    rhs = cylindrical_rhs(field, rz, phi)
    return torch.cat((rz, rhs), -1), diagnostics


class _RadialRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, polynomial, level, radius_scale, maximum_scale, iterations, tolerance, library=None):
        # Newton itself need not be taped: its converged simple-root derivative
        # is explicit.  FP64 evaluation also avoids spurious FP32 residual guards.
        p = polynomial.double()
        level64 = level.double()
        shape = torch.broadcast_shapes(p.shape[:-1], level64.shape)
        p = p.expand(*shape, p.shape[-1])
        level64 = level64.expand(shape)
        if (p.is_cuda and getattr(library, 'has_radial_root', False) and
                3 <= p.shape[-1] <= 33 and
                isinstance(radius_scale, (int, float)) and isinstance(maximum_scale, (int, float))):
            r, rp, derivative = library.radial_root(p, level64, radius_scale,
                                                   maximum_scale, iterations, tolerance)
            ctx.save_for_backward(rp, derivative)
            ctx.poly_shape, ctx.level_shape = polynomial.shape, level.shape
            ctx.poly_dtype, ctx.level_dtype = polynomial.dtype, level.dtype
            return r.to(polynomial.dtype)
        quadratic = p[..., 2]
        if not bool((quadratic > 0).all()):
            raise ValueError('radial root has nonpositive quadratic coefficient')
        minimum = 1e-12*radius_scale
        maximum = maximum_scale*radius_scale
        r = torch.sqrt(level64.clamp_min(0)/quadratic).clamp(minimum, maximum)
        powers = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
        for _ in range(iterations):
            rp = r[..., None]**powers
            value = (p*rp).sum(-1)
            derivative = (p[..., 1:]*powers[1:]*r[..., None]**(powers[1:]-1)).sum(-1)
            denom = torch.where(derivative.abs() > 1e-14, derivative,
                                torch.where(derivative < 0, -1e-14, 1e-14))
            limit = .45*r.abs().clamp_min(1e-8*radius_scale)
            step = ((value-level64)/denom).clamp(-limit, limit)
            r = torch.where((value-level64).abs() <= tolerance, r,
                            (r-step).clamp(minimum, maximum))
        rp = r[..., None]**powers
        derivative = (p[..., 1:]*powers[1:]*r[..., None]**(powers[1:]-1)).sum(-1)
        residual = (p*rp).sum(-1)-level64
        if not bool(torch.isfinite(r).all() & (r > 0).all() &
                    (residual.abs() < max(1e-8, 100*tolerance)).all() &
                    (derivative.abs() > 1e-12).all()):
            raise ValueError('radial root is unavailable, inaccurate, or singular')
        ctx.save_for_backward(rp, derivative)
        ctx.poly_shape, ctx.level_shape = polynomial.shape, level.shape
        ctx.poly_dtype, ctx.level_dtype = polynomial.dtype, level.dtype
        return r.to(polynomial.dtype)

    @staticmethod
    def backward(ctx, adj_r):
        powers, derivative = ctx.saved_tensors
        q = adj_r.double()/derivative
        dp = (-q[..., None]*powers).sum_to_size(ctx.poly_shape).to(ctx.poly_dtype)
        dl = q.sum_to_size(ctx.level_shape).to(ctx.level_dtype)
        return dp, dl, *(None for _ in range(len(ctx.needs_input_grad)-2))


class PsiGeometry:
    """Full native psi basis on a differentiable Hermite magnetic axis."""
    def __init__(self, axis_samples, coefficients, nfp=4, a=.15, degree=10, mtor=12,
                 *, maximum_radius_scale=1., root_iterations=20, root_tolerance=1e-12,
                 root_library=None):
        self.axis_samples = axis_samples
        self.coefficients = coefficients
        self.nfp, self.a, self.degree, self.mtor = nfp, a, degree, mtor
        self.maximum_radius_scale = maximum_radius_scale
        self.root_iterations, self.root_tolerance = root_iterations, root_tolerance
        self.root_library = root_library
        self._matrix = _coefficient_matrix(coefficients, degree, mtor, fixed_quadratic=1.)
        self._monomials = monomials(degree)

    def axis_eval(self, phi):
        R, Rp = periodic_hermite(phi, self.axis_samples[:, 0], self.axis_samples[:, 2], self.nfp)
        Z, Zp = periodic_hermite(phi, self.axis_samples[:, 1], self.axis_samples[:, 3], self.nfp)
        return R, Z, Rp, Zp

    def psi_eval(self, R, Z, phi):
        R, Z, phi = torch.broadcast_tensors(R, Z, torch.as_tensor(phi, device=R.device, dtype=R.dtype))
        axis_R, axis_Z, axis_Rp, axis_Zp = self.axis_eval(phi)
        X, Y = (R-axis_R)/self.a, (Z-axis_Z)/self.a
        polynomial, derivative_x, derivative_y = _monomial_features(X, Y, self.degree, self.a)
        matrix = self._matrix.to(phi.dtype)
        amplitude = _trig(phi, self.mtor, self.nfp) @ matrix.T
        amplitude_phi = _trig(phi, self.mtor, self.nfp, derivative=True) @ matrix.T
        value = (polynomial*amplitude).sum(-1)
        gR = (derivative_x*amplitude).sum(-1)
        gZ = (derivative_y*amplitude).sum(-1)
        gphi = (polynomial*amplitude_phi).sum(-1)-axis_Rp*gR-axis_Zp*gZ
        return value, gR, gZ, gphi

    def rays(self, phi, theta):
        phi = torch.as_tensor(phi, device=self.coefficients.device, dtype=self.coefficients.dtype)
        theta = torch.as_tensor(theta, device=phi.device, dtype=phi.dtype)
        phi, theta = torch.broadcast_tensors(phi, theta)
        amplitude = _trig(phi, self.mtor, self.nfp) @ self._matrix.T
        ax, bz, _, _, _, _, groups = _monomial_layout(self.degree, theta.device, theta.dtype)
        cosine, sine = theta.cos(), theta.sin()
        cp = torch.stack([cosine**k for k in range(self.degree+1)], -1)
        sp = torch.stack([sine**k for k in range(self.degree+1)], -1)
        denominator = (torch.stack([self.a**(a+b) for a, b in self._monomials])
                       if isinstance(self.a, torch.Tensor) else
                       _ray_denominators(self.degree, self.a, theta.device, theta.dtype))
        angular = cp.index_select(-1, ax)*sp.index_select(-1, bz)/denominator
        terms = amplitude*angular
        return torch.stack([torch.zeros_like(phi), torch.zeros_like(phi),
                            *(terms.index_select(-1, group).sum(-1) for group in groups)], -1)

    def solve_rays(self, polynomial, level):
        level = torch.as_tensor(level, device=polynomial.device, dtype=polynomial.dtype)
        return _RadialRoot.apply(polynomial, level, self.a, self.maximum_radius_scale,
                                 self.root_iterations, self.root_tolerance, self.root_library)


class TensorProductPsiOperator:
    """Exact native basis factorization without a huge dense per-query matrix.

    The native row order is (radial offset, vertical offset, phi).  Its modes
    are 63 Cartesian monomials times 25 toroidal functions minus fixed X^2.
    Production dimensions are unchanged.  The factorization changes only GEMM
    association/reduction order, which requires native numerical acceptance.
    """
    def __init__(self, field, axis_samples, *, nfp=4, a=.15, degree=10, mtor=12,
                 radial_grid=48, vertical_grid=48, phi_grid=48, rho_min=.002,
                 dtype=torch.float32):
        self.degree, self.mtor, self.nfp, self.a = degree, mtor, nfp, a
        device = axis_samples.device
        # Selection in FP64 matches native host offsets; indices are independent
        # of coil geometry and of axis movement.
        Roffset = torch.linspace(-a, a, radial_grid, device=device, dtype=torch.float64)
        Zoffset = torch.linspace(-a, a, vertical_grid, device=device, dtype=torch.float64)
        rr, zz = torch.meshgrid(Roffset, Zoffset, indexing='ij')
        selected = (torch.hypot(rr,zz) >= rho_min) & (torch.hypot(rr,zz) <= a)
        X, Y = (rr[selected]/a).to(dtype), (zz[selected]/a).to(dtype)
        phi = (torch.arange(phi_grid, device=device, dtype=torch.float64)*
               (2*math.pi/(nfp*phi_grid))).to(dtype)
        self.X, self.Y, self.phi = X, Y, phi
        self.monomials = monomials(degree)
        self.P, self.PR, self.PZ = _monomial_features(X, Y, degree, a)
        self.T = _trig(phi, mtor, nfp)
        self.Tp = _trig(phi, mtor, nfp, derivative=True)
        ax = axis_samples.to(dtype)
        ra, rap = periodic_hermite(phi, ax[:,0], ax[:,2], nfp)
        za, zap = periodic_hermite(phi, ax[:,1], ax[:,3], nfp)
        R = ra[None,:] + a*X[:,None]
        Z = za[None,:] + a*Y[:,None]
        self.points = torch.stack((R*phi.cos(), R*phi.sin(), Z), -1)
        B = field(self.points.reshape(-1,3), gradient=False).reshape(self.points.shape).to(dtype)
        cp = (-B[...,0]*phi.sin()+B[...,1]*phi.cos())/R
        cr = B[...,0]*phi.cos()+B[...,1]*phi.sin()-cp*rap
        cz = B[...,2]-cp*zap
        self.features = cr, cz, cp
        self.rhs = -2*X[:,None]*cr/a
        self.coefficient_count = len(self.monomials)*(2*mtor+1)-1

    def apply(self, coefficients):
        C = _coefficient_matrix(coefficients, self.degree, self.mtor)
        cr, cz, cp = self.features
        return cr*(self.PR@C@self.T.T) + cz*(self.PZ@C@self.T.T) + cp*(self.P@C@self.Tp.T)

    def transpose(self, rows):
        cr, cz, cp = self.features
        C = self.PR.T@(cr*rows)@self.T + self.PZ.T@(cz*rows)@self.T + self.P.T@(cp*rows)@self.Tp
        return C.reshape(-1)[1:]

    def column_norms(self):
        return _tensor_product_column_norms(self.features,self.PR,self.PZ,self.P,self.T,self.Tp)


def four_step_pcgls(operator, center_coefficients, center_scale, center_upper,
                    *, ridge=1e-6, iterations=4, live_column_scale=False):
    """Keep center coordinates/preconditioner fixed and differentiate live ridge.

    Formal native QR normalizes each endpoint with D=column_norms(A), so its physical
    penalty is ridge*||D*c||². In center coordinates z=D0*c the ridge rows are
    sqrt(ridge)*(D/D0)*z. ``live_column_scale=False`` retains the historical
    frozen-ridge nearby continuation, which remains the single-stage default.
    """
    dtype, device = operator.rhs.dtype, operator.rhs.device
    c0, scale, upper = [_frozen_fit_tensor(v, device=device, dtype=dtype)
                        for v in (center_coefficients, center_scale, center_upper)]
    if not bool((scale > 0).all()):
        raise ValueError('psi center normalization must be positive')
    sqrt_ridge = math.sqrt(ridge)
    live_scale=operator.column_norms() if live_column_scale else scale
    ratio=live_scale/scale
    operator.column_scales=live_scale.detach()
    operator.column_scale_ratio=ratio.detach()
    def triangular(v, transpose=False):
        matrix = upper.T if transpose else upper
        return torch.linalg.solve_triangular(matrix, v[:,None], upper=not transpose)[:,0]
    z = c0*scale
    residual = operator.rhs-operator.apply(z/scale)
    ridge_residual = -sqrt_ridge*ratio*z
    def normal(r, rr):
        return triangular(operator.transpose(r)/scale+sqrt_ridge*ratio*rr, transpose=True)
    n = normal(residual, ridge_residual)
    direction, gamma = n, (n*n).sum()
    for iteration in range(iterations):
        transformed = triangular(direction)
        image = operator.apply(transformed/scale)
        ridge_image = sqrt_ridge*ratio*transformed
        delta = (image*image).sum()+(ridge_image*ridge_image).sum()
        alpha = gamma/delta.clamp_min(1e-30)
        z = z+alpha*transformed
        residual = residual-alpha*image
        ridge_residual = ridge_residual-alpha*ridge_image
        if iteration+1 == iterations:
            break
        n = normal(residual, ridge_residual)
        next_gamma = (n*n).sum()
        direction = n+(next_gamma/gamma.clamp_min(1e-30))*direction
        gamma = next_gamma
    return z/scale, torch.sqrt((residual*residual).mean())


def predict_target_level(geometry: PsiGeometry, center: Mapping):
    """Native center discrete volume predictor, including the moving-axis term."""
    reference = geometry.coefficients
    def frozen(name):
        return torch.as_tensor(center[name], device=reference.device, dtype=torch.float64).detach()
    weights = frozen('target_adjoint_weights')
    axis_weights = frozen('target_axis_weights')
    R, _, _, _ = geometry.axis_eval(frozen('target_phi'))
    delta = (weights*(reference.double()-frozen('psi_coefficients'))).sum()
    delta = delta+(axis_weights*(R.double()-frozen('target_axis_R'))).sum()
    return frozen('target_level')-(delta+frozen('target_adjoint_residual'))/frozen('target_dVds')


@dataclass
class FrontResult:
    geometry: PsiGeometry
    level: torch.Tensor
    psi_train_rms: torch.Tensor
    axis_diagnostics: dict
    operator: TensorProductPsiOperator


def build_front(field, trace, center: Mapping, *, config=None, dtype=torch.float32):
    """Compose the full front; field and trace must close over live coil tensors."""
    cfg = dict(center.get('config', {}))
    cfg.update(config or {})
    axis_center = center['axis_samples']
    if not isinstance(axis_center, torch.Tensor):
        raise TypeError('center axis_samples must be a device tensor; convert the capture explicitly')
    nfp, a = int(cfg.get('nfp',4)), float(cfg['psi_a'])
    axis, diagnostics = differentiable_axis(field, trace, axis_center[0,:2], nfp=nfp,
        trace_steps=int(cfg.get('axis_trace_steps',240)), sample_count=int(cfg.get('axis_sample_count',240)))
    degree, mtor = int(cfg.get('psi_poly_degree',10)), int(cfg.get('psi_m_tor',12))
    op = TensorProductPsiOperator(field, axis, nfp=nfp, a=a, degree=degree, mtor=mtor,
        radial_grid=int(cfg.get('psi_n_r',48)), vertical_grid=int(cfg.get('psi_n_z',48)),
        phi_grid=int(cfg.get('psi_n_phi',48)), rho_min=float(cfg.get('psi_rho_min',.002)), dtype=dtype)
    coefficients, rms = four_step_pcgls(op, center['psi_coefficients'], center['psi_scale'],
        center['psi_upper'], ridge=float(cfg.get('psi_ridge',1e-6)), iterations=4,
        live_column_scale=bool(cfg.get('psi_live_column_scale',False)))
    diagnostics['psi_column_scale_relative_max']=(op.column_scale_ratio-1).abs().max().detach()
    geometry = PsiGeometry(axis, coefficients, nfp, a, degree, mtor,
        maximum_radius_scale=float(cfg.get('surface_max_radius_scale',1.)),
        root_iterations=int(cfg.get('surface_newton_iters',20)),
        root_tolerance=float(cfg.get('surface_newton_tolerance',1e-12)),
        root_library=getattr(field, 'library', None))
    level = predict_target_level(geometry, center)
    return FrontResult(geometry, level, rms, diagnostics, op)
