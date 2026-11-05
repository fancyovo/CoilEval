"""Example2: exactly32 multiplicative perturbations, then dense analytic BFGS.

The frozen example1 geometry, target volume and fitting radius stay fixed.
Run ``screen`` independently in four shards, then ``optimize`` with their JSONs.
This example uses capture internals to retain one center for the analytic reverse;
the package version and complete settings are recorded with every run.
"""
import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from coil_eval import CoilSet,EvaluationConfig,__version__
from coil_eval._two_stage.native import TwoStageNative
from coil_eval._two_stage.boundary import MaxClosedBoundaryError
from coil_eval.torch import TensorEvaluator
from _dense_bfgs import initial_inverse,descent_direction,inverse_bfgs_update,armijo_alphas,armijo_decision

HERE=Path(__file__).resolve().parent
TARGET=np.array([.47,.1,6.,3.5,.3])
CONFIG=EvaluationConfig(.3,psi_a=math.sqrt(.06),helicity=(1,0))
SEED=2026110502
PROTOCOL='coileval-S2-example1-perturb32-BFGS-20261105-v1'


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)


def digest(value):return hashlib.sha256(np.asarray(value,'<f8').tobytes()).hexdigest()


def candidates(fraction):
    coils=CoilSet.read_json(HERE/'example1.json')
    base=np.c_[coils.coeffs_x,coils.coeffs_y,coils.coeffs_z,coils.currents_a]
    assert digest(base)=='aebf705719b77ad064624362f5d90601dc4b8a2cfbaf3f8240b427faf8e1eaf7'
    z=np.random.default_rng(SEED).standard_normal((32,4,27))
    values=np.repeat(base[None],32,axis=0)
    values[:,:,:27]*=np.exp(math.log1p(fraction)*z)
    return values,digest(z)


def bilateral(error,tolerance,budget):
    tau=.2;r=error/tolerance;base=2*np.logaddexp(0.,-1/tau)
    edge=np.log(2)+np.logaddexp(0.,-2/tau)-base
    return budget*np.maximum(np.logaddexp(0.,(r-1)/tau)+np.logaddexp(0.,(-r-1)/tau)-base,0.)/edge


def formal_objective(raw):
    if raw['status']!='ok':return dict(valid=False,status=raw['status'],loss=None)
    d=raw['diagnostics']
    qs=d['qs_target_global_error_per_helicity']
    metrics=np.array([d['iota_target_surface'],d['coil_min_intercoil_distance'],
        d['coil_curvature_max'],d['coil_length_mean'],d['surface_volume']])
    if not np.isfinite(metrics).all() or not np.isfinite(qs) or qs<0:
        return dict(valid=False,status='nonfinite_metrics',loss=None)
    ve=0. if metrics[4]>=TARGET[4]*(1-1e-4) else max(TARGET[4]-metrics[4],0.)
    terms=[bilateral(metrics[0]-TARGET[0],.005,math.log(1.05)/10),
        .16*(max(TARGET[1]-metrics[1],0.)/.005)**2,
        .45*(max(metrics[2]-TARGET[2],0.)/1.5)**2,
        bilateral(metrics[3]-TARGET[3],.05,math.log(1.05)),.125*(ve/(.1*TARGET[4]))**2]
    radius=d['surface_effective_minor_radius']/d['surface_inverse_aspect_ratio']
    radius_cost=(np.logaddexp(0.,(radius-1.02)/.01)/np.logaddexp(0.,8.))**2
    quality=d['target_quality_penalty']
    loss=math.log(qs+1e-8)+sum(terms)+quality+radius_cost
    if not np.isfinite(loss):raise ValueError('nonfinite full objective')
    return dict(valid=True,status='ok',loss=loss,qs=qs,metrics=metrics,
        condition_terms=terms,quality_penalty=quality,radius_penalty=radius_cost,radius=radius)


def tensor_objective(m,radius):
    F=torch.nn.functional;tau=.2;base=2*math.log1p(math.exp(-1/tau))
    edge=math.log(2)+math.log1p(math.exp(-2/tau))-base
    def bilateral_t(error,tolerance,budget):
        r=error/tolerance
        return budget*(F.softplus((r-1)/tau)+F.softplus((-r-1)/tau)-base).clamp_min(0)/edge
    ve=torch.where(m.surface_volume>=TARGET[4]*(1-1e-4),torch.zeros_like(m.surface_volume),
                   (TARGET[4]-m.surface_volume).clamp_min(0))
    terms=(bilateral_t(m.iota_boundary-TARGET[0],.005,math.log(1.05)/10),
        .16*((TARGET[1]-m.coil_gap_min).clamp_min(0)/.005).square(),
        .45*((m.coil_curvature_max-TARGET[2]).clamp_min(0)/1.5).square(),
        bilateral_t(m.coil_length_mean-TARGET[3],.05,math.log(1.05)),.125*(ve/(.1*TARGET[4])).square())
    quality=.0025*F.softplus((m.relative_normal_drift_p95-.01)/.0025)
    return (torch.log(m.qs_volume+1e-8)+sum(terms)+10000*quality.square()
            +(F.softplus((radius-1.02)/.01)/math.log1p(math.exp(8.))).square()).sum()


class Oracle:
    def __init__(self,library,out):
        from coil_eval._native_loader import resolve_library
        library=resolve_library(library)
        self.library=library;self.out=out;self.cache=OrderedDict();self.calls=0
        self.driver=TwoStageNative(library)
        self.tensor=None

    def formal(self,physical):
        key=digest(physical)
        if key in self.cache:return self.cache[key]
        self.calls+=1;started=time.monotonic()
        coils=CoilSet(physical[:,:9],physical[:,9:18],physical[:,18:27],physical[:,27],3)
        try:raw,snapshot=self.driver.evaluate(coils,CONFIG,capture=True,include_torsion=False)
        except MaxClosedBoundaryError as error:
            raw=dict(status='no_surface',diagnostics=dict(error_message=str(error)));snapshot=None
        value=formal_objective(raw)
        save(self.out/'evaluations'/f'{self.calls:06d}.json',dict(physical_sha256=key,
            raw=raw,value=value,seconds=time.monotonic()-started))
        self.cache[key]=(raw,snapshot,value);self.cache.move_to_end(key)
        while len(self.cache)>3:self.cache.popitem(last=False)
        return raw,snapshot,value

    def gradient(self,physical,scale):
        from coil_eval._autograd.batch_field import BatchSegmentField,coil_segments_batch
        from coil_eval._autograd.batch_front import differentiable_batch_axis
        from coil_eval._autograd.front import periodic_hermite
        raw,snapshot,formal=self.formal(physical)
        if snapshot is None:raise ValueError('gradient unavailable at invalid center')
        if self.tensor is None:self.tensor=TensorEvaluator(self.library,config=CONFIG,nfp=3)
        ev=self.tensor
        x=torch.tensor(physical[None],device=ev.device,dtype=torch.float64,requires_grad=True)
        result=ev._evaluate(x,batch=True,configs=[CONFIG],_snapshots=[snapshot])
        segments,weights=coil_segments_batch(x,nfp=3,count=256)
        field=BatchSegmentField(segments.to(ev.dtype),weights.to(ev.dtype),ev.field_library)
        starts=torch.tensor(snapshot['axis_samples'][None,0,:2],device=x.device,dtype=x.dtype)
        axis,_=differentiable_batch_axis(field,starts,nfp=3,trace_steps=240,sample_count=240)
        phi=torch.arange(8,device=x.device,dtype=x.dtype)*(2*math.pi/(3*8))
        radius=periodic_hermite(phi,axis[:,:,0],axis[:,:,2],3)[0].mean(-1)
        loss=tensor_objective(result.metrics,radius)
        gradient,=torch.autograd.grad(loss,x)
        g=gradient[0].detach().cpu().numpy()*scale
        if not np.isfinite(g).all():raise ValueError('nonfinite analytic reverse')
        return g,dict(tensor_loss=float(loss.detach()),formal_loss=formal['loss'],
            absolute_gap=abs(float(loss.detach())-formal['loss']),gradient_norm=float(np.linalg.norm(g)))

    def close(self):
        self.driver.close()
        if self.tensor is not None:self.tensor.close()


def screen(a,oracle):
    physical,zsha=candidates(a.fraction);rows=[]
    for index in range(a.shard,32,a.shards):
        _,_,value=oracle.formal(physical[index])
        rows.append(dict(index=index,physical_sha256=digest(physical[index]),value=value))
        save(a.output/'screen.json',dict(protocol=PROTOCOL,version=__version__,fraction=a.fraction,
            seed=SEED,gaussian_sha256=zsha,shard=a.shard,shards=a.shards,rows=rows,config=asdict(CONFIG)))


def optimize(a,oracle):
    physical,zsha=candidates(a.fraction);rows=[]
    for path in a.screens:
        d=json.loads(path.read_text());assert d['fraction']==a.fraction and d['gaussian_sha256']==zsha
        rows.extend(d['rows'])
    assert sorted(r['index'] for r in rows)==list(range(32)),'exactly32 distinct candidates required'
    valid=[r for r in rows if r['value']['valid']]
    if not valid:raise ValueError('no valid start among exactly32 candidates; reduce noise')
    selected=min(valid,key=lambda row:row['value']['loss']);center=physical[selected['index']].copy()
    assert digest(center)==selected['physical_sha256']
    stats=json.loads((HERE/'dataset_stats.json').read_text(encoding='utf-8-sig'))
    assert stats['sha256']=='7335a0088164ce64b9d1e763ed2f733968636e8938f887165771ecbba781012e'
    scale=np.asarray(stats['stats']['token_scale'])
    assert scale.shape==(28,) and np.all(scale>0)
    _,_,value=oracle.formal(center);initial=value;best=value;best_physical=center.copy()
    H=initial_inverse(center.size);g=None;pending=None;bound=.05;accepted=0;rejected=0;history=[]
    np.savez_compressed(a.output/'start.npz',physical_tokens=center)
    reason='iteration_budget'
    for iteration in range(1,a.max_steps+1):
        if g is None:
            g,check=oracle.gradient(center,scale);g=g.ravel()
            if pending is not None:
                H=inverse_bfgs_update(H,pending[0],g-pending[1])['inverse'];pending=None
        direction=descent_direction(H,g,bound);H=direction['inverse'];p=direction['direction'];trials=[]
        for alpha in armijo_alphas():
            trial=center+(alpha*p).reshape(center.shape)*scale
            _,_,trial_value=oracle.formal(trial)
            decision=armijo_decision(value['loss'],trial_value['loss'],direction['directional_derivative'],
                alpha,trial_valid=trial_value['valid'])
            trials.append(dict(alpha=alpha,value=trial_value,decision=decision))
            if decision['accepted']:break
        ok=decision['accepted']
        if ok:
            center=trial;value=trial_value;accepted+=1;rejected=0
            pending=(alpha*p,g.copy());g=None;bound=min(.05,bound*1.5)
            if value['loss']<best['loss']:best=value;best_physical=center.copy()
        else:
            rejected+=1;bound=max(1e-6,bound*.5);H=initial_inverse(center.size)
        history.append(dict(iteration=iteration,accepted=ok,value=value,gradient_check=check,trials=trials))
        save(a.output/'history.json',history)
        np.savez_compressed(a.output/'selected.npz',physical_tokens=best_physical)
        save(a.output/'result.json',dict(protocol=PROTOCOL,status='running',version=__version__,
            fraction=a.fraction,selected_start=selected,valid_candidates=len(valid),initial=initial,best=best,
            accepted_updates=accepted,iterations=iteration,formal_calls=oracle.calls,
            physical_array_sha256=digest(best_physical),config=asdict(CONFIG)))
        if rejected>=max(10,int(.2*iteration)):
            reason='consecutive_rejection_limit';break
    result=json.loads((a.output/'result.json').read_text())
    result.update(status='completed' if accepted>0 else 'failed_no_accepted_update',stop_reason=reason,
        convergence_claim=False,max_steps=a.max_steps,targets=TARGET.tolist(),
        optimizer='dense inverse BFGS / Powell damping / analytic one reverse / formal Armijo')
    save(a.output/'result.json',result)
    if accepted==0:raise ValueError('no accepted update; preserve run and consider smaller noise')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['screen','optimize'])
    p.add_argument('--library',help='Optional custom native library');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fraction',type=float,default=.01);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--screens',type=Path,nargs='*')
    p.add_argument('--max-steps',type=int,default=400);a=p.parse_args()
    assert 0<a.fraction<1 and 0<=a.shard<a.shards<=32
    a.output.mkdir(parents=True,exist_ok=False)
    oracle=Oracle(a.library,a.output)
    try:
        if a.mode=='screen':screen(a,oracle)
        else:optimize(a,oracle)
    except Exception as error:
        save(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:oracle.close()


if __name__=='__main__':main()
