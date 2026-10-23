"""Fused native RK4 and its discrete reverse, including moving-point response.

Forward stores only stage points and tiny local RHS derivatives. Reverse first
propagates all state cotangents through RK4, then contracts the resulting field
cotangents with coil segments in one existing field VJP call. No numerical
differences, frozen axis coordinates, or point-by-segment tapes are used.
"""
from __future__ import annotations

import ctypes
import torch


def available(library):
    return hasattr(library.lib, 'fc_trace_forward_f32_f64')


def _bind(library):
    if getattr(library, '_trace_bound', False):
        return
    pointer, integer, stream = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    for field_precision in ('f32','f64'):
        for state_precision in ('f32','f64'):
            suffix = field_precision+'_'+state_precision
            forward = getattr(library.lib, 'fc_trace_forward_'+suffix)
            forward.argtypes = [pointer]*9+[integer]*3+[stream]
            forward.restype = integer
            reverse = getattr(library.lib, 'fc_trace_reverse_'+suffix)
            reverse.argtypes = [pointer]*7+[integer]*3+[stream]
            reverse.restype = integer
    library._trace_bound = True


def _call(library, name, tensors, integers, field_dtype, state_reference):
    _bind(library)
    suffix = {torch.float32:'f32',torch.float64:'f64'}[field_dtype]+'_'+{
        torch.float32:'f32',torch.float64:'f64'}[state_reference.dtype]
    with torch.cuda.device(state_reference.device):
        stream = torch.cuda.current_stream(state_reference.device).cuda_stream
        code = getattr(library.lib, name+'_'+suffix)(
            *[t.data_ptr() for t in tensors], *integers, stream)
    if code:
        raise RuntimeError(f'{name}: {library.lib.fc_error_string(code).decode()}')


class _Trace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, elements, starts, phi0, phi1, steps, library, return_samples):
        positions,elements,starts,phi0,phi1 = [t.contiguous() for t in
                                              (positions,elements,starts,phi0,phi1)]
        if starts.ndim != 2 or starts.shape[1] != 2 or not len(starts):
            raise ValueError('trace starts must be nonempty [lines,2]')
        if positions.ndim != 2 or positions.shape[1] != 3 or elements.shape != positions.shape:
            raise ValueError('trace segments must be matching [segments,3] arrays')
        if positions.dtype != elements.dtype or not all(t.is_cuda and t.device == starts.device
                for t in (positions,elements,starts,phi0,phi1)):
            raise ValueError('trace inputs must share a CUDA device and segment dtype')
        if phi0.shape != (len(starts),) or phi1.shape != phi0.shape or phi0.dtype != starts.dtype or phi1.dtype != starts.dtype:
            raise ValueError('phi bounds must have one value per line in state precision')
        lines = len(starts)
        states = torch.empty((steps+1,lines,2), device=starts.device,dtype=starts.dtype)
        stage_xyz = torch.empty((steps*4*lines,3),device=positions.device,dtype=positions.dtype)
        jacobian = torch.empty((steps*4*lines,2,2),device=starts.device,dtype=starts.dtype)
        field_derivative = torch.empty((steps*4*lines,2,3),device=starts.device,dtype=starts.dtype)
        _call(library,'fc_trace_forward',(positions,elements,starts,phi0,phi1,
              states,stage_xyz,jacobian,field_derivative),(len(positions),lines,steps),positions.dtype,starts)
        ctx.save_for_backward(positions,elements,phi0,phi1,stage_xyz,jacobian,field_derivative)
        ctx.steps,ctx.library,ctx.return_samples = steps,library,return_samples
        return states if return_samples else states[-1]

    @staticmethod
    def backward(ctx, adj_states):
        positions,elements,phi0,phi1,stage_xyz,jacobian,field_derivative = ctx.saved_tensors
        lines = len(phi0)
        adj_starts = torch.empty((lines,2),device=phi0.device,dtype=phi0.dtype)
        adj_B = torch.empty_like(stage_xyz)
        _call(ctx.library,'fc_trace_reverse',(jacobian,field_derivative,phi0,phi1,
              adj_states.contiguous(),adj_starts,adj_B),(lines,ctx.steps,int(ctx.return_samples)),
              positions.dtype,phi0)
        dp,dw = (torch.empty_like(positions),torch.empty_like(elements)) if any(ctx.needs_input_grad[:2]) else (None,None)
        if dp is not None:
            # J already contains point response. Passing dq=None prevents
            # counting that dependency twice; only explicit coil response remains.
            ctx.library.call('field_vjp',(positions,elements,stage_xyz,adj_B,None,dp,dw,None),
                             len(positions),len(stage_xyz),stage_xyz)
        return dp,dw,adj_starts,None,None,None,None,None


def trace(field, rz, phi0, phi1, steps, return_samples=False):
    """Drop-in trace callback for a SegmentField; phi bounds are fixed grids."""
    if not isinstance(steps,int) or steps < 1:
        raise ValueError('positive integer RK4 step count required')
    if rz.dtype not in (torch.float32,torch.float64):
        raise ValueError('trace state precision must be FP32 or FP64')
    bounds = [torch.broadcast_to(torch.as_tensor(phi,device=rz.device,dtype=rz.dtype).detach(),
                                (len(rz),)).contiguous() for phi in (phi0,phi1)]
    return _Trace.apply(field.positions,field.elements,rz,*bounds,steps,field.library,return_samples)


def local_rhs_derivatives(rz, phi, B, G):
    """Torch audit oracle for native J=F_y and M=F_B, including floor branches.

    This helper uses analytic spatial G and is not used by the production
    fused path. It permits CPU checks of moving-point and denominator logic.
    """
    phi=torch.as_tensor(phi,device=rz.device,dtype=rz.dtype)
    B,G=B.to(rz.dtype),G.to(rz.dtype)
    R=rz[...,0];c,s=phi.cos(),phi.sin()
    br=B[...,0]*c+B[...,1]*s;bp=-B[...,0]*s+B[...,1]*c
    active=bp.abs()>1e-14
    denominator=torch.where(active,bp,torch.where(bp<0,-torch.ones_like(bp),torch.ones_like(bp))*1e-14)
    bp_gradient=torch.stack((-s,c,torch.zeros_like(s)),-1)
    br_gradient=torch.stack((c,s,torch.zeros_like(s)),-1)
    bz_gradient=torch.stack((torch.zeros_like(s),torch.zeros_like(s),torch.ones_like(s)),-1)
    M0=R[...,None]*br_gradient/denominator[...,None]-torch.where(active,R*br/denominator.square(),0)[...,None]*bp_gradient
    M1=R[...,None]*bz_gradient/denominator[...,None]-torch.where(active,R*B[...,2]/denominator.square(),0)[...,None]*bp_gradient
    M=torch.stack((M0,M1),-2)
    qR=torch.stack((c,s,torch.zeros_like(s)),-1)
    B_R=(G@qR[...,None])[...,0]
    B_Z=G[...,2]
    explicit=torch.stack((br/denominator,B[...,2]/denominator),-1)
    J_R=(M@B_R[...,None])[...,0]+explicit
    J_Z=(M@B_Z[...,None])[...,0]
    J=torch.stack((J_R,J_Z),-1)
    rhs=torch.stack((R*br/denominator,R*B[...,2]/denominator),-1)
    return rhs,J,M
