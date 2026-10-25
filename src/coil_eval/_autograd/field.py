"""CUDA field, vector potential and RK4 reverse operators."""
from __future__ import annotations
import ctypes
import math
from operator import index
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint


class NativeFieldLibrary:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.lib = ctypes.CDLL(self.path)
        pointer, integer, stream = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        for suffix in ('f32', 'f64'):
            for name, count in [('field', 5), ('potential', 4), ('field_vjp', 8), ('potential_vjp', 7)]:
                fn = getattr(self.lib, f'fc_{name}_{suffix}')
                fn.argtypes = [pointer] * count + [integer, integer, stream]
                fn.restype = integer
        self.lib.fc_error_string.argtypes = [integer]
        self.lib.fc_error_string.restype = ctypes.c_char_p

    def call(self, name, tensors, ns, np, reference):
        suffix = {torch.float32:'f32', torch.float64:'f64'}[reference.dtype]
        with torch.cuda.device(reference.device):
            st = torch.cuda.current_stream(reference.device).cuda_stream
            code = getattr(self.lib, f'fc_{name}_{suffix}')(
                *[None if t is None else t.data_ptr() for t in tensors], ns, np, st)
        if code:
            raise RuntimeError(f'{name}: {self.lib.fc_error_string(code).decode()}')


class _Field(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, elements, points, library, gradient):
        positions, elements, points = [t.contiguous() for t in (positions, elements, points)]
        if not all(t.is_cuda and t.dtype == points.dtype and t.device == points.device
                   and t.ndim == 2 and t.shape[1] == 3 for t in (positions,elements,points)):
            raise ValueError('CUDA [N,3] tensors of matching dtype/device required')
        if positions.shape != elements.shape or not len(positions):
            raise ValueError('nonempty matching segment positions/elements required')
        b = torch.empty_like(points)
        g = torch.empty((len(points),3,3),dtype=points.dtype,device=points.device) if gradient else None
        library.call('field',(positions,elements,points,b,g),len(positions),len(points),points)
        ctx.save_for_backward(positions,elements,points)
        ctx.library,ctx.gradient = library,gradient
        ctx.set_materialize_grads(False)
        return (b,g) if gradient else b

    @staticmethod
    def backward(ctx, adj_b, adj_g=None):
        p,w,q=ctx.saved_tensors
        dp=torch.empty_like(p) if ctx.needs_input_grad[0] else None
        dw=torch.empty_like(w) if ctx.needs_input_grad[1] else None
        # The native segment reduction emits both arrays in one pass.
        if dp is not None or dw is not None:
            dp=torch.empty_like(p) if dp is None else dp
            dw=torch.empty_like(w) if dw is None else dw
        dq=torch.empty_like(q) if ctx.needs_input_grad[2] else None
        if not len(q):
            if dp is not None: dp.zero_();dw.zero_()
        else:
            ctx.library.call('field_vjp',(p,w,q,None if adj_b is None else adj_b.contiguous(),
                None if adj_g is None else adj_g.contiguous(),dp,dw,dq),len(p),len(q),q)
        return dp,dw,dq,None,None


class _Potential(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,w,q,library):
        p,w,q=[t.contiguous() for t in (p,w,q)]
        if not all(t.is_cuda and t.dtype == q.dtype and t.device == q.device for t in (p,w,q)):
            raise ValueError('matching CUDA tensors required')
        a=torch.empty_like(q)
        library.call('potential',(p,w,q,a),len(p),len(q),q)
        ctx.save_for_backward(p,w,q);ctx.library=library
        return a

    @staticmethod
    def backward(ctx,adj):
        p,w,q=ctx.saved_tensors
        dp,dw=(torch.empty_like(p),torch.empty_like(w)) if any(ctx.needs_input_grad[:2]) else (None,None)
        dq=torch.empty_like(q) if ctx.needs_input_grad[2] else None
        if not len(q):
            if dp is not None:dp.zero_();dw.zero_()
        else:ctx.library.call('potential_vjp',(p,w,q,adj.contiguous(),dp,dw,dq),len(p),len(q),q)
        return dp,dw,dq,None


def coil_segments(tokens, nfp=4, count=256):
    """Native midpoint Fourier quadrature, currents in amperes, all symmetries."""
    if tokens.ndim!=2 or tokens.shape[0]<1:
        raise ValueError('tokens require nonempty [coil,3*ncoeff+1] shape')
    if isinstance(count,bool) or isinstance(nfp,bool):
        raise ValueError('positive integer nfp and segment count required')
    try:
        count,nfp=index(count),index(nfp)
    except TypeError as error:
        raise ValueError('positive integer nfp and segment count required') from error
    if count<1 or nfp<1:
        raise ValueError('positive integer nfp and segment count required')
    nc,width=tokens.shape
    coeff=(width-1)//3
    if coeff<1 or width != 3*coeff+1 or coeff%2 != 1:
        raise ValueError('tokens must contain three odd Fourier coefficient blocks and current')
    order=(coeff-1)//2
    t=(torch.arange(count,device=tokens.device,dtype=tokens.dtype)+.5)/count
    modes=torch.arange(1,order+1,device=tokens.device,dtype=tokens.dtype)
    phase=2*math.pi*t[:,None]*modes
    basis=torch.cat((torch.ones_like(t[:,None]),torch.stack((phase.sin(),phase.cos()),-1).flatten(1)),1)
    derivative=torch.cat((torch.zeros_like(t[:,None]),torch.stack((2*math.pi*modes*phase.cos(),-2*math.pi*modes*phase.sin()),-1).flatten(1)),1)
    c=tokens[:,:-1].reshape(nc,3,coeff)
    xyz=(c@basis.T).transpose(1,2)
    dl=(c@derivative.T).transpose(1,2)/count
    positions,elements=[],[]
    for k in range(nfp):
        angle=2*math.pi*k/nfp
        rotation=tokens.new_tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
        for mirror in (False,True):
            transform=rotation@torch.diag(tokens.new_tensor([1,-1,-1] if mirror else [1,1,1]))
            positions.append((xyz@transform.T).flatten(0,1))
            elements.append(((dl@transform.T)*tokens[:,-1,None,None]*(-1 if mirror else 1)).flatten(0,1))
    return torch.cat(positions),torch.cat(elements)


class SegmentField:
    def __init__(self, positions, elements, library):
        self.positions,self.elements,self.library=positions,elements,library

    @classmethod
    def from_tokens(cls,tokens,library,nfp=4,count=256):
        return cls(*coil_segments(tokens,nfp,count),library)

    def __call__(self, xyz, gradient=False):
        # Match the native mixed trace: FP64 integration states may query an
        # FP32 field. The cast remains differentiable with respect to points.
        return _Field.apply(self.positions,self.elements,xyz.to(self.positions.dtype),self.library,gradient)

    def potential(self,xyz):
        return _Potential.apply(self.positions,self.elements,xyz.to(self.positions.dtype),self.library)

    def trace(self,rz,phi0,phi1,steps,return_samples=False,checkpoint_steps=8,use_fused=True):
        """Discrete RK4 reverse, recomputing bounded blocks of trajectory state.

        Prefer the validated fused native implementation when exported by the
        supplied library. The checkpointed implementation remains an audit
        reference (use_fused=False). No phi-parameter derivative
        is promised: phi0/phi1 specify the fixed integration grid.
        """
        if use_fused:
            from .trace import available, trace
            if available(self.library):
                return trace(self,rz,phi0,phi1,steps,return_samples=return_samples)
        if steps<1:raise ValueError('positive trace step count required')
        phi0=torch.as_tensor(phi0,device=rz.device,dtype=rz.dtype).detach()
        phi1=torch.as_tensor(phi1,device=rz.device,dtype=rz.dtype).detach()
        h=(phi1-phi0)/steps
        def block(p,w,start,first,count):
            local=SegmentField(p,w,self.library)
            def rhs(y,phi):
                phi=torch.broadcast_to(phi,y.shape[:-1])
                sn,cs=phi.sin(),phi.cos();r,z=y.unbind(-1)
                b=local(torch.stack((r*cs,r*sn,z),-1)).to(y.dtype)
                br=b[:,0]*cs+b[:,1]*sn;bp=-b[:,0]*sn+b[:,1]*cs
                floor=torch.where(bp<0,-torch.ones_like(bp),torch.ones_like(bp))*1e-14
                denominator=torch.where(bp.abs()>1e-14,bp,floor)
                return torch.stack((r*br/denominator,r*b[:,2]/denominator),-1)
            y=start;states=[];hs=h[...,None]
            for j in range(first,first+count):
                phi=phi0+h*j
                k1=rhs(y,phi);k2=rhs(y+hs*k1/2,phi+h/2)
                k3=rhs(y+hs*k2/2,phi+h/2);k4=rhs(y+hs*k3,phi+h)
                y=y+hs*(k1+2*k2+2*k3+k4)/6
                if return_samples:states.append(y)
            return torch.stack(states) if return_samples else y
        history=[rz[None]] if return_samples else None
        for first in range(0,steps,checkpoint_steps):
            count=min(checkpoint_steps,steps-first)
            # Bind iteration bounds; checkpoint re-evaluates this closure later.
            def operation(p,w,y,first=first,count=count):return block(p,w,y,first,count)
            if torch.is_grad_enabled() and any(t.requires_grad for t in (self.positions,self.elements,rz)):
                out=checkpoint(operation,self.positions,self.elements,rz,use_reentrant=False)
            else:out=operation(self.positions,self.elements,rz)
            if return_samples:history.append(out);rz=out[-1]
            else:rz=out
        return torch.cat(history) if return_samples else rz
