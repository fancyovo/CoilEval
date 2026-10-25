"""Native value and center-hinted neighborhood calls without training imports.

The private binding is an unchanged copy of the frozen ctypes wrapper. This
module retains the numerical sequence of LocalFullGradientEstimator while
accepting coefficient arrays directly instead of Flow token layouts. Calls
sharing a native library must be serialized on one host thread by the public
session: the native QR and target-volume caches are thread-local scratch.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import ctypes

from ._basis import native_mode_arrays
from ._binding import (
    BatchCoilFieldGpu,
    coil_component_gradient_native,
    coil_metric_gradient_native,
    score_coils_native,
)


_GEOMETRY_DIAGNOSTICS = (
    "coil_length_mean",
    "coil_curvature_p95",
    "coil_curvature_max",
    "coil_min_intercoil_distance",
    "coil_min_axis_distance",
    "coil_high_mode_energy_fraction",
)


def _coil_arrays(coils: Any) -> tuple[np.ndarray, ...]:
    """Coerce only the array layout required by the native binding."""
    arrays = tuple(
        np.ascontiguousarray(getattr(coils, name), dtype=np.float64)
        for name in ("coeffs_x", "coeffs_y", "coeffs_z")
    )
    if arrays[0].ndim != 2 or not all(a.shape == arrays[0].shape for a in arrays):
        raise ValueError("coil coefficients must have matching (coil, coefficient) shapes")
    if not all(arrays[0].shape):
        raise ValueError("coil coefficient arrays must be nonempty")
    currents = np.ascontiguousarray(coils.currents_a, dtype=np.float64).reshape(-1)
    if currents.size != arrays[0].shape[0]:
        raise ValueError("currents must have one value per base coil")
    if not all(np.isfinite(a).all() for a in (*arrays, currents)):
        raise ValueError("coil coefficients and currents must be finite")
    return (*arrays, currents)


def _center_axis(center_raw: dict[str, Any]) -> tuple[float, float]:
    if center_raw.get("status") != "ok":
        raise RuntimeError(
            f"center evaluation is not ok: {center_raw.get('status', 'missing status')}"
        )
    diagnostics = center_raw.get("diagnostics", {})
    try:
        axis = (float(diagnostics["axis_R"]), float(diagnostics["axis_Z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("accepted center does not contain a finite magnetic-axis hint") from exc
    if not np.isfinite(axis).all():
        raise RuntimeError("accepted center does not contain a finite magnetic-axis hint")
    return axis


class NativeBackend:
    """Duck-typed backend for the standalone evaluator's public session.

    ``coils`` exposes coeffs_x/y/z, currents_a and nfp. ``config`` exposes
    native_options(), helicity, segments_per_coil, psi_iterations and
    alpha_iterations. Losses, parameter normalization and FD assembly belong
    to the public layer. This backend returns private raw native dictionaries.
    """

    def __init__(self, library: str | Path, device_id: int = 0) -> None:
        self.library = Path(library).expanduser().resolve()
        self.device_id = int(device_id)
        self._modes = native_mode_arrays()
        self._active_batch: BatchCoilFieldGpu | None = None

    def evaluate(self, coils: Any, config: Any) -> dict[str, Any]:
        if config.psi_fit == "two_stage":
            from ._two_stage.native import TwoStageNative
            return TwoStageNative(self.library,self.device_id).evaluate(coils,config)[0]
        arrays = _coil_arrays(coils)
        options = dict(config.native_options())
        options["segments_per_coil"] = int(config.segments_per_coil)
        return score_coils_native(
            self.library,
            *arrays,
            int(coils.nfp),
            device_id=self.device_id,
            target_helicity=tuple(config.helicity),
            config_overrides=options,
        )

    def nearby(
        self,
        coils: Any,
        candidates: Sequence[Any],
        center_raw: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Recompute one center and evaluate its center-hinted local neighbors.

        Geometry diagnostics use the same center analytical linearization as
        the frozen path; maximum absolute current is recomputed per candidate.
        Raw local alpha_relative_l2 and coordinate-score diagnostics retain
        their frozen-center meaning and are not advertised as local metrics.
        A numerical/domain failure propagates rather than becoming zero loss.
        """
        candidates = list(candidates)
        if not candidates:
            return []
        if config.psi_fit == "two_stage":
            from ._two_stage.native import TwoStageNative
            if (config.psi_nearby_method=="local" and config.fast and config.flux_method=="stokes"
                    and tuple(config.helicity)==(1,4) and coils.nfp==4 and coils.n_base_coils==4):
                try:
                    import torch
                except ImportError:
                    torch=None
                if torch is not None and torch.version.cuda is not None:
                    from ._two_stage.local import nearby
                    return nearby(self,coils,candidates,center_raw,config)
            driver=TwoStageNative(self.library,self.device_id)
            axis=_center_axis(center_raw)
            # Formal independent reconstruction is the initial correctness
            # oracle. Optimized local neighborhoods are measured separately.
            return [driver.evaluate(candidate,config,axis_hint=axis)[0] for candidate in candidates]
        if self._active_batch is not None:
            raise RuntimeError("native neighborhood calls must not overlap")
        total_started = perf_counter()
        center_arrays = _coil_arrays(coils)
        candidate_arrays = []
        for candidate in candidates:
            arrays = _coil_arrays(candidate)
            if int(candidate.nfp) != int(coils.nfp) or any(
                a.shape != c.shape for a, c in zip(arrays, center_arrays)
            ):
                raise ValueError("nearby candidates must match center shape and nfp")
            candidate_arrays.append(arrays)
        x, y, z, currents = (
            np.ascontiguousarray(np.stack(values), dtype=np.float64)
            for values in zip(*candidate_arrays)
        )
        query_count = len(candidates)
        axis = _center_axis(center_raw)
        options = dict(config.native_options())
        options["segments_per_coil"] = int(config.segments_per_coil)
        if float(options.get("target_volume", 0.0)) <= 0.0:
            raise ValueError("standalone local evaluation requires positive target volume")
        options.update(
            axis_hint_enabled=1,
            axis_hint_require_continuation=2,
            axis_hint_R=axis[0],
            axis_hint_Z=axis[1],
        )
        timings: dict[str, float] = {}
        local_stats: dict[str, Any] = {}
        psi_stats: dict[str, Any] = {}

        # Allocate the Python owner before invoking its initializer so finally
        # can release a native handle even if construction raises part-way.
        batch = BatchCoilFieldGpu.__new__(BatchCoilFieldGpu)
        self._active_batch = batch
        try:
            started = perf_counter()
            batch.__init__(
                self.library,
                x,
                y,
                z,
                currents,
                int(coils.nfp),
                segments_per_coil=int(config.segments_per_coil),
                device_id=self.device_id,
            )
            timings["field_create"] = perf_counter() - started

            started = perf_counter()
            capture = batch.capture_psi_center(
                *center_arrays,
                target_helicity=tuple(config.helicity),
                config_overrides=options,
            )
            timings["center_capture"] = perf_counter() - started
            captured = capture["score_result"]
            if captured.get("status") != "ok":
                raise RuntimeError(f"center capture failed with status {captured.get('status')}")

            started = perf_counter()
            refined_axis = batch.refine_axis_hint(
                np.full(query_count, axis[0], dtype=np.float64),
                np.full(query_count, axis[1], dtype=np.float64),
                trace_steps=240 if config.fast else 960,
                newton_iterations=6,
                finite_difference_step=2.0e-4,
                maximum_newton_step=0.25,
                residual_tolerance=1.0e-7,
                hint_max_distance=0.08,
            )
            timings["axis_refine"] = perf_counter() - started

            started = perf_counter()
            batch_axis = batch.trace_axis_samples(
                refined_axis["R"],
                refined_axis["Z"],
                integration_steps=240 if config.fast else 960,
                sample_count=240,
            )
            timings["axis_samples"] = perf_counter() - started

            started = perf_counter()
            psi, psi_rms, psi_stats = batch.fit_psi_pcgls(
                *batch_axis,
                *self._modes,
                capture["psi_coefficients"],
                radius_scale=float(options.get("psi_a", 0.05)),
                radial_grid=48,
                vertical_grid=48,
                phi_grid=48,
                rho_min=0.002,
                ridge=1.0e-6,
                iterations=int(config.psi_iterations),
            )
            timings["psi"] = perf_counter() - started

            started = perf_counter()
            coil_linear = coil_component_gradient_native(
                self.library, *center_arrays, int(coils.nfp)
            )
            coil_components = np.full(query_count, coil_linear["component"], dtype=np.float64)
            for values, center_values, name in zip(
                (x, y, z, currents), center_arrays, ("x", "y", "z", "current")
            ):
                delta = (values - center_values) * coil_linear["gradient"][name]
                coil_components += delta.reshape(query_count, -1).sum(axis=1)
            timings["coil_linear"] = perf_counter() - started

            started = perf_counter()
            set_flux = getattr(batch.lib, "sgpu_set_flux_method", None)
            if set_flux is None and config.flux_method != "area":
                raise RuntimeError("this native library does not support Stokes flux")
            if set_flux is not None:
                set_flux.argtypes = [ctypes.c_int]
                set_flux.restype = ctypes.c_int
                batch._check(set_flux(int(config.flux_method == "stokes")))
            local_results, local_stats = batch.score_local_batch(
                currents,
                *batch_axis,
                refined_axis["residual"],
                refined_axis["topology_trace"],
                refined_axis["topology_det"],
                psi,
                psi_rms,
                coil_components,
                capture,
                surface_theta_count=int(options.get("surface_theta_count", 128)),
                alpha_iterations=int(config.alpha_iterations),
            )
            timings["local_score"] = perf_counter() - started

            started = perf_counter()
            self._add_geometry(local_results, center_arrays, (x, y, z, currents), int(coils.nfp))
            timings["geometry_metrics"] = perf_counter() - started
        finally:
            set_flux = getattr(getattr(batch, "lib", None), "sgpu_set_flux_method", None)
            if set_flux is not None:
                set_flux(0)
            started = perf_counter()
            self.close()
            timings["cleanup"] = perf_counter() - started

        timing_record = {
            "scope": "shared_batch",
            "query_count": query_count,
            "stages_s": timings,
            "total_s": perf_counter() - total_started,
            "psi_stats": psi_stats,
            "native_local_stats": local_stats,
        }
        for result in local_results:
            result["local_timing"] = dict(timing_record)
        return local_results

    def _add_geometry(
        self,
        results: list[dict[str, Any]],
        center_arrays: tuple[np.ndarray, ...],
        candidate_arrays: tuple[np.ndarray, ...],
        nfp: int,
    ) -> None:
        metrics = coil_metric_gradient_native(self.library, *center_arrays, nfp)
        delta = np.stack(
            [values - center for values, center in zip(candidate_arrays[:3], center_arrays[:3])],
            axis=1,
        )
        linear = np.einsum("mcak,ecak->em", metrics["derivatives"], delta)
        metric_names = (
            "length_mean", "curvature_p95", "curvature_max",
            "min_intercoil_distance", "min_axis_distance", "high_mode_fraction",
        )
        for index, result in enumerate(results):
            diagnostics = result.setdefault("diagnostics", {})
            for metric, name in enumerate(_GEOMETRY_DIAGNOSTICS):
                diagnostics[name] = float(metrics["values"][metric_names[metric]] + linear[index, metric])
            diagnostics["coil_current_abs_max_a"] = float(np.max(np.abs(candidate_arrays[3][index])))

    def close(self) -> None:
        """Release an active batch and clear exposed native QR scratch."""
        batch, self._active_batch = self._active_batch, None
        if batch is None:
            return
        try:
            batch.close()
        finally:
            # A failed library load has no cache to clear. If a library did
            # load, attempt the native cleanup even when field destruction fails.
            if getattr(batch, "lib", None) is not None:
                batch.clear_psi_preconditioner()
