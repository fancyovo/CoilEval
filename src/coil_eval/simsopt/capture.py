"""Public owned source records for two-stage alpha+nu initialization."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import EvaluationConfig
from ..evaluator import EvaluationError, _decode_result
from ..types import CoilSet, EvaluationResult
from .clebsch import AlphaFit
from .state import SourcePsi


@dataclass(frozen=True)
class SourceCapture:
    """One evaluated target volume and its owned source-psi/alpha records."""
    result: EvaluationResult
    source: SourcePsi | None
    alpha: AlphaFit | None

    @property
    def valid(self):
        return self.result.valid and self.source is not None and self.alpha is not None


def capture_sources(coils, configs, *, library=None, device_id=0):
    """Capture one coil geometry at one or more target volumes.

    A single EvaluationConfig or an iterable is accepted; the result is always
    a list in input order. Configurations use ``psi_fit='two_stage'`` and may
    differ only in target_volume. One axis/S1/domain/S2 fit is shared. Source
    arrays are owned host data and remain valid after this function returns.
    """
    if not isinstance(coils, CoilSet):
        raise TypeError("coils must be a CoilSet")
    configs = [configs] if isinstance(configs, EvaluationConfig) else list(configs)
    for config in configs:
        if not isinstance(config, EvaluationConfig):
            raise TypeError("configs must contain EvaluationConfig objects")
        if config.psi_fit != "two_stage":
            raise ValueError("capture_sources requires psi_fit='two_stage'")
    if not configs:
        return []
    reference = configs[0].to_dict()
    for config in configs[1:]:
        current = config.to_dict()
        if any(reference.get(key) != current.get(key) for key in reference.keys() | current.keys()
               if key != "target_volume"):
            raise ValueError("capture_sources configurations may differ only in target_volume")
    from .._native_loader import resolve_library
    from .._binding import GpuError
    from .._two_stage.native import TwoStageNative
    driver = TwoStageNative(resolve_library(library), device_id=device_id)
    try:
        try:
            rows = driver.evaluate_many(coils, configs, capture=True)
        except GpuError as exc:
            raise EvaluationError(f"native source capture failed: {exc}") from exc
    finally:
        driver.close()
    if len(rows) != len(configs):
        raise EvaluationError("native source capture returned an incomplete result list")
    captures = []
    for config, (raw, snapshot) in zip(configs, rows):
        result = _decode_result(raw, config)
        source = alpha = None
        if result.valid and snapshot is not None:
            try:
                source = SourcePsi.from_snapshot(snapshot)
                alpha = AlphaFit.from_snapshot(snapshot)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                source = alpha = None
                result = EvaluationResult("invalid_source_capture", result.metrics,
                    {**result.diagnostics, "source_capture_error": str(exc)}, result.timings, result.mode)
        captures.append(SourceCapture(result, source, alpha))
    return captures
