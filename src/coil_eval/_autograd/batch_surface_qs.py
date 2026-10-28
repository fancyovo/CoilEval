"""Cooperative target-surface sampling with query-batched field sweeps.

Geometry roots and residual-controlled angular solves retain their independent
query branches. Each sampler yields when it needs the magnetic field, allowing
one native query/point grid to evaluate that sweep for every query. The same
coroutine drives scalar sampling, so batching cannot silently select different
numerical settings or omit a rejection check.
"""
from __future__ import annotations

import torch


def _run_field_steps(steps, field):
    """Drive one scalar sampler; preserve callables without a gradient keyword."""
    result = None
    try:
        while True:
            try:
                points, gradient = steps.send(result)
            except StopIteration as completed:
                return completed.value
            result = field(points, gradient=True) if gradient else field(points)
    finally:
        steps.close()


def sample_boozer_surfaces(geometries, levels, physics, field, configs, **options):
    """Return independent sample dictionaries using one field call per sweep.

    ``field`` accepts ``[query, point, 3]`` points with the native batch field
    convention. Configurations and output angular grids must be compatible;
    the surrounding evaluator already enforces this for an independent batch.
    Per-query geometry, levels, fitted angular data and field cotangents remain
    live. No fields, surface samples or solver answers are detached or reused
    between independent queries.
    """
    from .boozer import _sample_boozer_surface_steps

    count = len(geometries)
    if count == 0 or any(len(values) != count for values in (levels, physics, configs)):
        raise ValueError('Boozer batch requires matching nonempty query inputs')
    samplers = [_sample_boozer_surface_steps(geometry, level, phys, config, **options)
                for geometry, level, phys, config in zip(geometries, levels, physics, configs)]
    responses = [None] * count
    try:
        while True:
            requests, completed = [], []
            for sampler, response in zip(samplers, responses):
                try:
                    requests.append(sampler.send(response))
                except StopIteration as result:
                    completed.append(result.value)
            if completed:
                if requests:
                    raise ValueError('Boozer batch queries require identical field stages')
                return completed
            shape, gradient = requests[0][0].shape, requests[0][1]
            if any(points.shape != shape or derivative != gradient
                   for points, derivative in requests):
                raise ValueError('Boozer batch queries require matching field grids')
            points = torch.stack([request[0] for request in requests])
            if gradient:
                magnetic, jacobian = field(points, gradient=True)
                if magnetic.shape != points.shape or jacobian.shape != (*points.shape, 3):
                    raise ValueError('Boozer batch field returned an invalid derivative shape')
                responses = list(zip(magnetic.unbind(), jacobian.unbind()))
            else:
                magnetic = field(points)
                if magnetic.shape != points.shape:
                    raise ValueError('Boozer batch field returned an invalid value shape')
                responses = list(magnetic.unbind())
    finally:
        # Close suspended samplers immediately after an error, releasing their
        # live geometry/fit graphs without waiting for cyclic garbage collection.
        for sampler in samplers:
            sampler.close()
