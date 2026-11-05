"""CPU-only validation of the public tensor evaluator's supported settings."""
from __future__ import annotations

from .config import EvaluationConfig


# These settings select shared downstream operators within one tensor batch.
# Target volumes and fitting radii are independent per query. The second-fit
# samplers/row counts are also independent and use masked section padding.
_BATCH_SHARED_SETTINGS = (
    "psi_fit", "psi_target_refit", "alpha_radial_weighting", "alpha_weight_sigma", "alpha_count_floor", "alpha_radial_basis",
)


def validate_tensor_config(config):
    """Validate one config without loading PyTorch or a native library."""
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    if not config.fast or config.flux_method != "stokes":
        raise ValueError('tensor evaluation requires fast=True and flux_method="stokes"')
    if (config.segments_per_coil, config.psi_iterations, config.alpha_iterations) != (256, 4, 4):
        raise ValueError("tensor evaluation requires 256 segments, four fit iterations")


def validate_tensor_batch_configs(configs, *, count=None):
    """Return validated configs; reject unsupported mixes before GPU capture.

    Each query may use its own target volume, psi radius and second-fit sampling
    settings. One batch shares its fit method and alpha weighting parameters.
    """
    configs = tuple(configs)
    if not configs:
        raise ValueError("a tensor batch requires at least one configuration")
    if count is not None and len(configs) != count:
        raise ValueError("provide one configuration per geometry")
    for config in configs:
        validate_tensor_config(config)
    for name in _BATCH_SHARED_SETTINGS:
        if any(getattr(config, name, 'zernike') != getattr(configs[0], name, 'zernike') for config in configs[1:]):
            raise ValueError(f"tensor batches require shared {name}; group compatible configurations")
    return configs
