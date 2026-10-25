"""Batched derivative-root isolation with strict scalar handling of hard cases."""
from concurrent.futures import ThreadPoolExecutor
from operator import index
import os
import numpy as np


def resolve_workers(workers=None):
    """Respect an explicit Slurm CPU allowance; one worker is the default."""
    if workers is None:
        try:
            workers=int(os.environ.get('COIL_EVAL_GEOMETRY_WORKERS','1'))
        except ValueError as error:
            raise ValueError('COIL_EVAL_GEOMETRY_WORKERS must be a positive integer') from error
    if isinstance(workers,bool):raise ValueError('geometry workers must be a positive integer')
    try:
        requested=index(workers)
    except TypeError as error:
        raise ValueError('geometry workers must be a positive integer') from error
    if requested<1:raise ValueError('geometry workers must be a positive integer')
    slurm=os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm is not None:
        try:allowance=int(slurm)
        except ValueError as error:raise ValueError('SLURM_CPUS_PER_TASK must be a positive integer') from error
        if allowance<1:raise ValueError('SLURM_CPUS_PER_TASK must be a positive integer')
    elif hasattr(os,'sched_getaffinity'):
        allowance=len(os.sched_getaffinity(0))
    else:
        allowance=os.cpu_count() or 1
    return requested,min(requested,allowance)


def _eigenvalues(matrix,workers):
    # NumPy's LAPACK loop releases the GIL. Independent chunks use separate
    # workspaces and preserve matrix order. BLAS thread counts are controlled
    # by the caller's numerical environment, independently of this pool.
    if workers==1 or len(matrix)<256:
        return np.linalg.eigvals(matrix)
    chunks=np.array_split(matrix,min(workers,len(matrix)),axis=0)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return np.concatenate(list(executor.map(np.linalg.eigvals,chunks)),axis=0)


def _values(coefficients, points):
    result=np.zeros_like(points,dtype=np.longdouble)
    for column in range(coefficients.shape[1]-1,-1,-1):
        result=result*points+coefficients[:,column,None]
    return result


def branch_limits(polynomials, maximum, scalar, *, workers=1):
    shape=polynomials.shape[:-1]
    p=np.asarray(polynomials,dtype=np.float64).reshape(-1,polynomials.shape[-1])
    maximum=np.broadcast_to(maximum,shape).reshape(-1)
    q=p[:,2:]*np.arange(2,p.shape[1])
    count,lenq=q.shape
    if lenq==1:
        invalid=(p[:,0]!=0)|(p[:,1]!=0)|(p[:,2]<=0)|~np.isfinite(p).all(axis=1)|(maximum<=0)
        for index in np.flatnonzero(invalid):
            scalar(p[index],float(maximum[index]))
        return maximum.reshape(shape).copy(),np.zeros(shape,dtype=bool),np.zeros(shape,dtype=np.int64),int(invalid.sum())
    degrees=np.argmax((q!=0)[:,::-1],axis=1)
    degrees=lenq-1-degrees
    roots=np.full((count,lenq-1),complex(np.inf,0),dtype=np.complex128)
    fallback=(p[:,0]!=0)|(p[:,1]!=0)|(p[:,2]<=0)|~np.isfinite(p).all(axis=1)|(maximum<=0)
    for degree in range(1,lenq):
        indices=np.flatnonzero((degrees==degree)&~fallback)
        if not len(indices):
            continue
        matrix=np.zeros((len(indices),degree,degree),dtype=np.float64)
        if degree>1:
            matrix[:,np.arange(1,degree),np.arange(degree-1)]=1.
        matrix[:,:,-1]=-q[indices,:degree]/q[indices,degree,None]
        # NumPy polynomial.polyroots reverses both companion axes for accuracy.
        try:
            found=_eigenvalues(matrix[:,::-1,::-1],workers)
        except np.linalg.LinAlgError:
            fallback[indices]=True
            continue
        roots[indices,:degree]=found
        fallback[indices]|=~np.isfinite(found).all(axis=1)
    real,imaginary=roots.real,abs(roots.imag)
    inside=(real>0)&(real<maximum[:,None])&np.isfinite(real)
    near=inside&(imaginary<=1e-4*np.maximum(1.,abs(real)))
    critical=inside&(imaginary<=1e-7*np.maximum(1.,abs(real)))
    ordered=np.sort(np.where(critical,real,np.inf),axis=1)
    number=critical.sum(axis=1)
    valid=np.arange(lenq-1)[None,:]<number[:,None]
    if lenq>2:
        with np.errstate(invalid='ignore'):
            close=(ordered[:,1:]-ordered[:,:-1])<=1e-9*np.maximum(1.,abs(ordered[:,1:]))
        fallback|=(close&valid[:,1:]).any(axis=1)
    qlong=q.astype(np.longdouble)
    eps=64*np.finfo(np.longdouble).eps
    # A companion root near the real axis is explicitly ambiguous even when
    # q(real(root)) is positive. The scalar resolver isolates its actual real
    # roots exactly instead of inferring an entire interval from one value.
    fallback|=(near&~critical).any(axis=1)
    rootpoints=np.where(valid,ordered,0).astype(np.longdouble)
    rootvalue=_values(qlong,rootpoints)
    rootscale=_values(abs(qlong),abs(rootpoints))
    fallback|=(valid&(abs(rootvalue)>1e-6*np.maximum(rootscale,np.finfo(float).tiny))).any(axis=1)
    columns=np.arange(lenq)[None,:]
    right=np.concatenate((ordered,maximum[:,None]),axis=1)
    right=np.where(columns<number[:,None],right,maximum[:,None])
    left=np.concatenate((np.zeros((count,1)),ordered),axis=1)
    active=columns<=number[:,None]
    midpoint=np.where(active,.5*(left+right),0).astype(np.longdouble)
    value=_values(qlong,midpoint)
    scale=_values(abs(qlong),abs(midpoint))
    fallback|=(active&(abs(value)<=eps*scale)).any(axis=1)
    signs=np.where(value>0,1,-1)
    fallback|=signs[:,0]!=1
    folds=valid&(signs[:,:-1]>0)&(signs[:,1:]<0)
    has_fold=folds.any(axis=1)
    first=folds.argmax(axis=1)
    caps=np.where(has_fold,ordered[np.arange(count),first],maximum)
    capacity_values=np.polynomial.polynomial.polyval(caps,p.T,tensor=False)
    fallback|=~np.isfinite(caps)|(caps<=0)|~np.isfinite(capacity_values)|(capacity_values<=0)
    flat=(valid&(signs[:,:-1]==signs[:,1:])&(ordered<caps[:,None])).sum(axis=1)
    # Unusual critical-root clusters and unresolved numerical signs retain the
    # scalar extended-precision policy instead of changing branch semantics.
    for index in np.flatnonzero(fallback):
        try:
            result=scalar(p[index],float(maximum[index]))
        except ValueError as error:
            if hasattr(error,'diagnostics'):
                error.diagnostics={**error.diagnostics,'flat_ray_index':int(index)}
            raise
        caps[index]=result['maximum_u']
        has_fold[index]=result['first_fold_u'] is not None
        flat[index]=result['flat_nonfold_count']
    return caps.reshape(shape),has_fold.reshape(shape),flat.reshape(shape),int(fallback.sum())
