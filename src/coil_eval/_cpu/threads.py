"""Thread limits for CPU sessions, respecting process affinity and Slurm."""
from contextlib import contextmanager
import os
from operator import index


def available_threads(requested=None):
    if requested is not None:
        if isinstance(requested, bool):
            raise ValueError("threads must be a positive integer or None")
        try:
            value = index(requested)
        except TypeError as exc:
            raise ValueError("threads must be a positive integer or None") from exc
        if value < 1:
            raise ValueError("threads must be a positive integer or None")
        return value
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit() and int(allocated) > 0:
        counts.append(int(allocated))
    return max(1, min(counts))


@contextmanager
def thread_scope(threads):
    import torch
    from threadpoolctl import threadpool_info,threadpool_limits
    previous = torch.get_num_threads()
    initial=threadpool_info()
    limits={};openblas={}
    for pool in initial:
        prefix=pool['prefix']
        limit=min(threads,pool['num_threads']) if pool['internal_api']=='openblas' else threads
        limits[prefix]=min(limits.get(prefix,limit),limit)
        if pool['internal_api']=='openblas':openblas[prefix]=limits[prefix]
    # Some OpenBLAS builds crash when a process started with a small pool
    # grows it at runtime. Formal QR uses Torch's LAPACK provider. Keep other
    # OpenBLAS pools at or below their entry size, while limiting MKL/OpenMP
    # normally. The outer scope restores the original provider state exactly.
    with threadpool_limits(limits=limits):
        try:
            torch.set_num_threads(threads)
            with threadpool_limits(limits=openblas):
                yield dict(requested_threads=threads,initial_pools=initial,
                           active_pools=threadpool_info(),torch_threads=torch.get_num_threads())
        finally:
            torch.set_num_threads(previous)
