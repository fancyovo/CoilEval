"""Small CPU C++ adapter; importing it does not load any NVIDIA library."""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import numpy as np
from .._binding import _SgpuScoreConfig, SGPU_SCORE_TIMING_NAMES


def pointer(array):
    return None if array is None else np.asarray(array).ctypes.data_as(ct.c_void_p)


def resolve_cpu_library(path=None):
    if path is not None:
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    configured = os.environ.get("COIL_EVAL_CPU_LIBRARY")
    if configured:
        return resolve_cpu_library(configured)
    folder = Path(__file__).resolve().parents[1] / "_native"
    for name in ("libcoil_eval_cpu.so", "coil_eval_cpu.dll", "libcoil_eval_cpu.dylib"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise RuntimeError("CoilEval CPU backend is not installed. Install a CPU-enabled wheel, "
                       "or pass the CPU library path as library=...")


def segments(coils, count):
    """Native midpoint rule and native base/sample/period/reflection order."""
    t = (np.arange(count, dtype=np.float64) + .5) / count
    xyz = np.stack((coils.coeffs_x, coils.coeffs_y, coils.coeffs_z), 1)
    order = (xyz.shape[-1] - 1) // 2
    p = np.broadcast_to(xyz[..., 0, None], (*xyz.shape[:-1], count)).copy()
    w = np.zeros_like(p)
    for m in range(1, order + 1):
        phase = 2 * np.pi * m * t
        p += xyz[..., 2*m-1, None] * np.sin(phase) + xyz[..., 2*m, None] * np.cos(phase)
        w += 2*np.pi*m*(xyz[..., 2*m-1, None]*np.cos(phase)-xyz[..., 2*m, None]*np.sin(phase))/count
    p, w = p.transpose(0, 2, 1), w.transpose(0, 2, 1)
    ps, ws = [], []
    for period in range(coils.nfp):
        phi = 2*np.pi*period/coils.nfp
        rotation = np.array([[np.cos(phi),-np.sin(phi),0], [np.sin(phi),np.cos(phi),0], [0,0,1]])
        for reflected in (False, True):
            transform = rotation @ np.diag([1,-1,-1] if reflected else [1,1,1])
            ps.append(p @ transform.T)
            ws.append((w @ transform.T)*coils.currents_a[:,None,None]*(-1 if reflected else 1))
    return (np.ascontiguousarray(np.stack(ps,2).reshape(-1,3)),
            np.ascontiguousarray(np.stack(ws,2).reshape(-1,3)))


class CpuLibrary:
    def __init__(self, path=None):
        self.path = resolve_cpu_library(path)
        self.lib = ct.CDLL(str(self.path))
        P, I, D = ct.c_void_p, ct.c_int, ct.c_double
        signatures = {
            'sgpu_default_score_config': [P], 'ce_cpu_open': [P,P,I,I,P],
            'ce_cpu_close': [P], 'ce_cpu_field': [P,P,I,I,P,P,P],
            'ce_cpu_trace': [P,P,P,P,P,I,D,D,I,I],
            'ce_cpu_axis': [P,P,P,P,I,I,I,P,P,P,P],
            'ce_cpu_coil_metrics': [P,P,P,P,I,I,I,P],
            'ce_cpu_target': [P,P,P,I,I,I,I,D,D,D,D,P,P],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = I
        self.lib.ce_cpu_close.restype = None
        self.lib.sgpu_last_error.restype = ct.c_char_p
        omp=getattr(self.lib,'ce_cpu_openmp_enabled',None)
        if omp is not None:
            omp.argtypes=[];omp.restype=I
        self.openmp_enabled=bool(omp()) if omp is not None else None
        avx2=getattr(self.lib,'ce_cpu_avx2_enabled',None)
        if avx2 is not None:
            avx2.argtypes=[];avx2.restype=I
        self.avx2_enabled=bool(avx2()) if avx2 is not None else None
        self.axis_polish_info=getattr(self.lib,'ce_cpu_axis_polish_info',None)
        if self.axis_polish_info is not None:
            self.axis_polish_info.argtypes=[P,I];self.axis_polish_info.restype=None

    def configuration(self, config, nfp):
        cfg = _SgpuScoreConfig()
        self.check(self.lib.sgpu_default_score_config(ct.byref(cfg)))
        for key, value in config.native_options().items():
            setattr(cfg, key, value)
        cfg.target_M, cfg.target_N = config.helicity
        return cfg

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.sgpu_last_error().decode() or f"CPU native error {code}")


class CpuField:
    def __init__(self, coils, count, threads, library):
        self.library, self.coils = library, coils
        self.positions, self.elements = segments(coils, count)
        self.handle = ct.c_void_p()
        library.check(library.lib.ce_cpu_open(pointer(self.positions), pointer(self.elements),
                      len(self.positions), threads, ct.byref(self.handle)))

    def close(self):
        if self.handle.value:
            self.library.lib.ce_cpu_close(self.handle)
            self.handle = ct.c_void_p()

    def evaluate(self, xyz, *, gradient=False, potential=False, mixed=True):
        xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1,3)
        B = np.empty_like(xyz)
        G = np.empty((len(xyz),3,3)) if gradient else None
        A = np.empty_like(xyz) if potential else None
        self.library.check(self.library.lib.ce_cpu_field(self.handle, pointer(xyz), len(xyz),
                           int(mixed), pointer(B), pointer(G), pointer(A)))
        dtype = np.float32 if mixed else np.float64
        if potential:
            return A.astype(dtype)
        if gradient:
            return B.astype(dtype), G.astype(dtype)
        return B.astype(dtype)

    def __call__(self, xyz, gradient=False):
        import torch
        out = self.evaluate(xyz.detach().numpy(), gradient=gradient)
        return tuple(torch.from_numpy(a) for a in out) if gradient else torch.from_numpy(out)

    def potential(self, xyz):
        import torch
        return torch.from_numpy(self.evaluate(xyz.detach().numpy(), potential=True))

    def trace(self, rz, phi0, phi1, steps, return_samples=False):
        import torch
        host = np.ascontiguousarray(rz.detach().numpy(),dtype=np.float64)
        r,z = np.ascontiguousarray(host[:,0]),np.ascontiguousarray(host[:,1])
        if return_samples:
            history = [host.copy()]
            for k in range(steps):
                re,ze = np.empty_like(r),np.empty_like(z)
                self.library.check(self.library.lib.ce_cpu_trace(self.handle,pointer(r),pointer(z),
                    pointer(re),pointer(ze),len(r),float(phi0)+(float(phi1)-float(phi0))*k/steps,
                    float(phi0)+(float(phi1)-float(phi0))*(k+1)/steps,1,1))
                r,z=re,ze;history.append(np.column_stack((r,z)))
            return torch.from_numpy(np.stack(history)).to(rz.dtype)
        re,ze = np.empty_like(r),np.empty_like(z)
        self.library.check(self.library.lib.ce_cpu_trace(self.handle,pointer(r),pointer(z),
            pointer(re),pointer(ze),len(r),float(phi0),float(phi1),steps,1))
        return torch.from_numpy(np.column_stack((re,ze))).to(rz.dtype)

    def axis(self, cfg):
        coils=self.coils; sample=np.empty((cfg.axis_sample_count,4)); info=np.empty(8)
        timing=np.zeros(len(SGPU_SCORE_TIMING_NAMES))
        polish=self.library.axis_polish_info
        if polish is not None:polish(None,1)
        code=self.library.lib.ce_cpu_axis(self.handle,*[pointer(x) for x in
            (coils.coeffs_x,coils.coeffs_y,coils.coeffs_z)],*coils.coeffs_x.shape,coils.nfp,
            ct.byref(cfg),pointer(sample),pointer(info),pointer(timing))
        details={}
        if polish is not None:
            values=np.empty(12);polish(pointer(values),0)
            names=('attempted','accepted','newton_steps','skipped_nonelliptic','skipped_far',
                'rejected_singular','rejected_displacement_or_domain','rejected_no_improvement',
                'rejected_not_converged','rejected_topology','max_displacement_m','wall_s')
            details['axis_fp64_polish']=dict(zip(names,values))
            details['axis_fp64_polish']['used']=bool(values[1]>0)
        if code:
            if code==1:self.library.check(code)
            return None, dict(error=self.library.lib.sgpu_last_error().decode(),
                status='branch_lost' if code==3 else 'no_axis',**details), timing
        return sample, {**dict(zip(('axis_R','axis_Z','axis_residual','axis_topology_trace',
            'axis_topology_det','axis_ellipse_aspect','axis_candidate_count','axis_trace_s'),info)),**details}, timing
