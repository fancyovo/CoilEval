"""Independent-geometry CUDA field and trace operators."""
from __future__ import annotations

import ctypes
import math
from operator import index
import torch


def validate_batch_geometry(positions,elements,points=None,*,require_cuda=False):
    if positions.ndim != 3 or positions.shape[-1] != 3 or min(positions.shape[:2]) <= 0:
        raise ValueError('segments require nonempty [query,segment,3] shape')
    if elements.shape != positions.shape or elements.dtype != positions.dtype or elements.device != positions.device:
        raise ValueError('segment positions/elements must have identical shapes, dtype and device')
    if positions.dtype not in (torch.float32,torch.float64):
        raise ValueError('FP32 or FP64 segment storage required')
    if require_cuda and not positions.is_cuda:
        raise ValueError('batch native primitives require CUDA tensors')
    if points is not None:
        if points.ndim != 3 or points.shape[0] != positions.shape[0] or points.shape[-1] != 3:
            raise ValueError('points require [query,point,3] with exactly matching query count')
        if points.device != positions.device:
            raise ValueError('points and segments must share device')
    return positions.shape[0],positions.shape[1],None if points is None else points.shape[1]


def broadcast_trace_bounds(phi,starts):
    """Explicit 2-D bounds avoid confusing per-query and per-line vectors."""
    if starts.ndim != 3 or starts.shape[-1] != 2 or min(starts.shape[:2]) <= 0:
        raise ValueError('batch trace starts require nonempty [query,line,2]')
    value=torch.as_tensor(phi,device=starts.device,dtype=starts.dtype).detach()
    if value.ndim == 1:
        raise ValueError('batch phi bounds must be scalar or 2-D: [query,1], [1,line], or [query,line]')
    if value.ndim > 2:
        raise ValueError('batch phi bounds must be scalar or 2-D')
    try:return torch.broadcast_to(value,starts.shape[:2]).contiguous()
    except RuntimeError as e:raise ValueError('phi bounds do not broadcast to [query,line]') from e


def _bind(library):
    if getattr(library,'_batch_field_bound',False):return
    ptr,integer,stream=ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64
    for dtype in ('f32','f64'):
        for name,count in (('field',5),('potential',4),('field_vjp',8),('potential_vjp',7)):
            fn=getattr(library.lib,'fc_batch_'+name+'_'+dtype)
            fn.argtypes=[ptr]*count+[integer]*3+[stream];fn.restype=integer
    for fd in ('f32','f64'):
        for sd in ('f32','f64'):
            for name,count in (('forward',9),('reverse',7)):
                fn=getattr(library.lib,'fc_batch_trace_'+name+'_'+fd+'_'+sd)
                fn.argtypes=[ptr]*count+[integer]*4+[stream];fn.restype=integer
    library._batch_field_bound=True


def _call(library,name,tensors,integers,reference,state_reference=None):
    _bind(library)
    dt={torch.float32:'f32',torch.float64:'f64'}
    suffix=dt[reference.dtype]+('' if state_reference is None else '_'+dt[state_reference.dtype])
    with torch.cuda.device(reference.device):
        stream=torch.cuda.current_stream(reference.device).cuda_stream
        code=getattr(library.lib,'fc_batch_'+name+'_'+suffix)(
            *[None if t is None else t.data_ptr() for t in tensors],*integers,stream)
    if code:raise RuntimeError(f'batch {name}: {library.lib.fc_error_string(code).decode()}')


class _BatchField(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,w,q,library,gradient,potential):
        p,w,q=[v.contiguous() for v in (p,w,q)]
        nq,ns,np_=validate_batch_geometry(p,w,q,require_cuda=True)
        value=torch.empty_like(q)
        grad=torch.empty((*q.shape[:-1],3,3),device=q.device,dtype=q.dtype) if gradient else None
        if potential:_call(library,'potential',(p,w,q,value),(ns,np_,nq),q)
        else:_call(library,'field',(p,w,q,value,grad),(ns,np_,nq),q)
        ctx.save_for_backward(p,w,q);ctx.library=library;ctx.potential=potential;ctx.gradient=gradient
        ctx.set_materialize_grads(False)
        return (value,grad) if gradient else value

    @staticmethod
    def backward(ctx,av,ag=None):
        p,w,q=ctx.saved_tensors;nq,ns,np_=p.shape[0],p.shape[1],q.shape[1]
        dp,dw=(torch.empty_like(p),torch.empty_like(w)) if any(ctx.needs_input_grad[:2]) else (None,None)
        dq=torch.empty_like(q) if ctx.needs_input_grad[2] else None
        av=None if av is None else av.contiguous();ag=None if ag is None else ag.contiguous()
        if not np_:
            if dp is not None:dp.zero_();dw.zero_()
        elif ctx.potential:_call(ctx.library,'potential_vjp',(p,w,q,av,dp,dw,dq),(ns,np_,nq),q)
        else:_call(ctx.library,'field_vjp',(p,w,q,av,ag,dp,dw,dq),(ns,np_,nq),q)
        return dp,dw,dq,None,None,None


class _BatchTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,w,starts,phi0,phi1,steps,library,return_samples):
        p,w,starts,phi0,phi1=[t.contiguous() for t in (p,w,starts,phi0,phi1)]
        queries,segments,_=validate_batch_geometry(p,w,require_cuda=True)
        if starts.ndim!=3 or starts.shape[0]!=queries or starts.shape[-1]!=2 or starts.shape[1]<=0:
            raise ValueError('trace starts must match segment query count and have [query,line,2] shape')
        if starts.device!=p.device or starts.dtype not in (torch.float32,torch.float64):
            raise ValueError('trace states require same CUDA device and FP32/FP64 dtype')
        lines=starts.shape[1]
        states=torch.empty((steps+1,queries,lines,2),device=starts.device,dtype=starts.dtype)
        xyz=torch.empty((queries,steps*4*lines,3),device=p.device,dtype=p.dtype)
        J=torch.empty((queries,steps*4*lines,2,2),device=starts.device,dtype=starts.dtype)
        M=torch.empty((queries,steps*4*lines,2,3),device=starts.device,dtype=starts.dtype)
        _call(library,'trace_forward',(p,w,starts,phi0,phi1,states,xyz,J,M),
              (segments,lines,queries,steps),p,starts)
        ctx.save_for_backward(p,w,phi0,phi1,xyz,J,M)
        ctx.steps,ctx.library,ctx.return_samples=steps,library,return_samples
        return states if return_samples else states[-1]

    @staticmethod
    def backward(ctx,adj_states):
        p,w,phi0,phi1,xyz,J,M=ctx.saved_tensors
        queries,segments,lines=p.shape[0],p.shape[1],phi0.shape[1]
        adj_starts=torch.empty((queries,lines,2),device=phi0.device,dtype=phi0.dtype)
        adj_B=torch.empty_like(xyz)
        _call(ctx.library,'trace_reverse',(J,M,phi0,phi1,adj_states.contiguous(),adj_starts,adj_B),
              (lines,queries,ctx.steps,int(ctx.return_samples)),p,phi0)
        dp,dw=(torch.empty_like(p),torch.empty_like(w)) if any(ctx.needs_input_grad[:2]) else (None,None)
        if dp is not None:
            _call(ctx.library,'field_vjp',(p,w,xyz,adj_B,None,dp,dw,None),
                  (segments,ctx.steps*4*lines,queries),p)
        return dp,dw,adj_starts,None,None,None,None,None


def coil_segments_batch(tokens,nfp=4,count=256):
    """Vectorized native quadrature with a leading independent-query dimension."""
    if tokens.ndim!=3 or tokens.shape[0]<=0 or tokens.shape[1]<=0:
        raise ValueError('tokens require [query,coil,3*ncoeff+1]')
    if isinstance(count,bool) or isinstance(nfp,bool):
        raise ValueError('positive integer nfp and segment count required')
    try:
        count,nfp=index(count),index(nfp)
    except TypeError as error:
        raise ValueError('positive integer nfp and segment count required') from error
    if count<1 or nfp<1:
        raise ValueError('positive integer nfp and segment count required')
    width=tokens.shape[-1];coeff=(width-1)//3
    if coeff<1 or width!=3*coeff+1 or coeff%2!=1:
        raise ValueError('three odd Fourier coefficient blocks and current required')
    order=(coeff-1)//2
    t=(torch.arange(count,device=tokens.device,dtype=tokens.dtype)+.5)/count
    modes=torch.arange(1,order+1,device=tokens.device,dtype=tokens.dtype)
    phase=2*math.pi*t[:,None]*modes
    basis=torch.cat((torch.ones_like(t[:,None]),torch.stack((phase.sin(),phase.cos()),-1).flatten(1)),1)
    derivative=torch.cat((torch.zeros_like(t[:,None]),torch.stack((2*math.pi*modes*phase.cos(),-2*math.pi*modes*phase.sin()),-1).flatten(1)),1)
    c=tokens[...,:-1].reshape(*tokens.shape[:2],3,coeff)
    xyz=(c@basis.T).transpose(-1,-2);dl=(c@derivative.T).transpose(-1,-2)/count
    positions,elements=[],[]
    for k in range(nfp):
        angle=2*math.pi*k/nfp
        rotation=tokens.new_tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
        for mirror in (False,True):
            transform=rotation@torch.diag(tokens.new_tensor([1,-1,-1] if mirror else [1,1,1]))
            positions.append((xyz@transform.T).flatten(1,2))
            elements.append(((dl@transform.T)*tokens[...,-1,None,None]*(-1 if mirror else 1)).flatten(1,2))
    return torch.cat(positions,1),torch.cat(elements,1)


class BatchSegmentField:
    def __init__(self,positions,elements,library):
        validate_batch_geometry(positions,elements)
        self.positions,self.elements,self.library=positions,elements,library

    @classmethod
    def from_tokens(cls,tokens,library,nfp=4,count=256):
        return cls(*coil_segments_batch(tokens,nfp,count),library)

    def __call__(self,points,gradient=False):
        validate_batch_geometry(self.positions,self.elements,points)
        return _BatchField.apply(self.positions,self.elements,points.to(self.positions.dtype),self.library,gradient,False)

    def potential(self,points):
        validate_batch_geometry(self.positions,self.elements,points)
        return _BatchField.apply(self.positions,self.elements,points.to(self.positions.dtype),self.library,False,True)

    def trace(self,starts,phi0,phi1,steps,return_samples=False):
        if not isinstance(steps,int) or steps<=0:
            raise ValueError('positive integer RK4 steps required')
        if starts.ndim!=3 or starts.shape[-1]!=2 or min(starts.shape[:2])<=0:
            raise ValueError('trace starts require nonempty [query,line,2]')
        if starts.shape[0]!=self.positions.shape[0]:
            raise ValueError('trace and segments must have equal query counts')
        bounds=[broadcast_trace_bounds(phi,starts) for phi in (phi0,phi1)]
        return _BatchTrace.apply(self.positions,self.elements,starts,*bounds,steps,self.library,return_samples)
