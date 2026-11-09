"""Experimental geometric target-surface score with no volume alpha fit."""
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import torch

from ._autograd.boozer import sample_boozer_surface
from ._autograd.downstream import DownstreamConfig
from ._autograd.field import SegmentField, coil_segments
from ._autograd.front import PsiGeometry
from ._autograd.two_stage_front import build_two_stage_front
from ._two_stage.native import TwoStageNative
from .qs import SurfaceQSConfig
from .qs_metrics import boozer_qs_metrics_field, boozer_spectrum_metrics
from .torch import TensorEvaluator, pack_coils


@dataclass
class TargetSurfaceEvaluation:
    parameters: torch.Tensor
    capture_result: dict
    snapshot: dict | None
    timings: dict
    metrics: dict | None = None
    surface: dict | None = None
    surface_error: str | None = None


class TargetSurfaceEvaluator:
    """Private scalar experiment for geometric alpha at one target volume.

    ``skip_volume_alpha=False`` selects the full native capture as a control.
    Both arms use the same surface computation and differentiated front.
    ``metrics`` is absent when the native center or the surface sampler fails.
    """

    def __init__(self, library=None, field_library=None, *, config,
                 surface_qs=None, **engine_options):
        self.surface_qs = SurfaceQSConfig() if surface_qs is None else surface_qs
        if (config.psi_fit != 'two_stage' or self.surface_qs.alpha_order == 0 or
                self.surface_qs.alpha_coordinates != 'geometric'):
            raise ValueError('target-surface evaluation requires two-stage psi and geometric surface alpha')
        self.engine = TensorEvaluator(library, field_library, config=config, **engine_options)

    def close(self):
        self.engine.close()

    def evaluate(self, coils, *, derivative=False, skip_volume_alpha=True):
        engine = self.engine
        engine._check()
        if coils.nfp != engine.nfp:
            raise ValueError('coil field periods do not match evaluator nfp')
        if engine.device.type == 'cpu':
            from ._cpu.threads import thread_scope
            scope = thread_scope(engine.threads)
        else:
            scope = nullcontext()
        with scope, torch.set_grad_enabled(derivative):
            return self._evaluate(coils, derivative, skip_volume_alpha)

    def _evaluate(self, coils, derivative, skip_volume_alpha):
        engine = self.engine
        config = engine.config
        on_cpu = engine.device.type == 'cpu'
        timings = {}

        def synchronize():
            if not on_cpu:
                torch.cuda.synchronize(engine.device)

        def timed(name, fn):
            synchronize()
            start = perf_counter()
            try:
                return fn()
            finally:
                synchronize()
                timings[name] = perf_counter() - start

        def capture():
            if on_cpu:
                return engine.backend._capture(coils, config, geometry_only=skip_volume_alpha)
            mode = ('geometry_reverse' if derivative else 'geometry') if skip_volume_alpha else (
                'reverse' if derivative else 'sources')
            driver = TwoStageNative(engine.library_path, engine.device.index)
            try:
                return driver.evaluate(coils, config, capture=mode, include_torsion=False)
            finally:
                driver.close()

        synchronize()
        start = perf_counter()
        parameters = pack_coils(coils, device=engine.device).requires_grad_(derivative)
        raw, snapshot = timed('native_capture', capture)
        result = TargetSurfaceEvaluation(parameters, raw, snapshot, timings)
        if raw['status'] == 'ok' and snapshot is not None:
            options = {**snapshot['config'], 'nfp': coils.nfp}
            downstream = DownstreamConfig.from_mapping(options)
            if on_cpu:
                from ._cpu.autograd import CpuSegmentField
                field_type = CpuSegmentField
            else:
                field_type = SegmentField

            def tensor(value):
                return torch.tensor(value, dtype=torch.float64, device=engine.device)

            def front():
                if derivative:
                    # Match the tensor frontend's precision and segment count.
                    p, w = coil_segments(parameters, nfp=coils.nfp,
                                         count=config.segments_per_coil if on_cpu else 256)
                    field = field_type(p.to(engine.dtype), w.to(engine.dtype), engine.field_library)
                    response = build_two_stage_front(field, field.trace, engine._center(snapshot),
                                                     dtype=torch.float64)
                    return response.geometry, response.level
                geometry = PsiGeometry(tensor(snapshot['axis_samples']), tensor(snapshot['psi_coefficients']),
                    nfp=coils.nfp, a=options['psi_a'], degree=options['psi_poly_degree'], mtor=options['psi_m_tor'])
                return geometry, tensor(snapshot['target_level']).reshape(())

            geometry, level = timed('surface_geometry', front)

            def sample():
                p, w = coil_segments(parameters, nfp=coils.nfp,
                                     count=256 if derivative and not on_cpu else config.segments_per_coil)
                field = field_type(p, w, engine.field_library)
                physics = {'major_radius': geometry.axis_samples[:, 0].double().mean()}
                return sample_boozer_surface(geometry, level, physics, field, downstream,
                    **self.surface_qs.sampler_options(), field_derivatives=self.surface_qs.compute_f_T)

            def metrics():
                s = result.surface
                if self.surface_qs.compute_f_T:
                    values = boozer_qs_metrics_field(s['B'], s['dBdx'], s['x_theta'], s['x_phi'],
                        nfp=coils.nfp, helicity=config.helicity, iota=s['iota'], G=s['G'],
                        major_radius=s['major_radius'])
                else:
                    values = boozer_spectrum_metrics(s['B_magnitude'], nfp=coils.nfp, helicity=config.helicity)
                return {**{k: values[k] for k in ('f_B', 'f_T') if k in values}, 'iota': s['iota']}

            try:
                result.surface = timed('surface_coordinates', sample)
                result.metrics = timed('surface_metrics', metrics)
            except (ValueError, RuntimeError) as exc:
                result.surface_error = f'{type(exc).__name__}: {exc}'
        timings['total'] = perf_counter() - start
        return result
