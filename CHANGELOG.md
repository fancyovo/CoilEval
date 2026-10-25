# Changelog

## 0.4.0a1

- Added opt-in two-stage psi fitting with Cartesian domain sampling and experimental weighted polar quadrature.
- Added two-stage local reverse and finite-difference gradient paths, including batched local endpoints.
- Exposed optional `psi_fit_rms` in scalar metrics for psi training-equation residual objectives, in T/m.
- Added optional experimental smooth radial-density weights for two-stage alpha fitting.
- Extended physical parameter tensors to higher Fourier orders, including order 16.
- Added shared multi-volume source capture, a callable GPU coil field, alpha+nu initialization and standard Simsopt LS/Newton integration.
- Added native-normalized surface f_C RMS diagnostics and volume aggregation helpers.
- Added examples for two-stage evaluation and separate GPU preparation/CPU-only process pools, with bilingual API documentation.

## 0.3.0a3

- Pinned CUDA runtime dependencies to a validated CUDA 12.8 set, fixing cuSPARSE and nvJitLink symbol mismatches during installation from package indexes.

## 0.3.0a2

- Added Linux x86-64 CUDA 12.8 binary wheels with multiple GPU architectures.
- Evaluators automatically load the packaged native backend and runtime dependencies.
- Source builds use CMake through the Python build backend; explicit library paths remain supported.
- Updated installation guides and examples for compiler-free wheel installation.

## 0.3.0a1

- Added optional PyTorch metrics, local reverse gradients and independent-geometry batch evaluation. Physical parameters and custom tensor objectives use a separate additive API.

## 0.2.0

- Added explicit Stokes boundary flux for fast local evaluations. Area quadrature remains the default.

## 0.1.0

- Added opt-in fast integration settings and serialized result configuration.

## 0.0.0

- Coil and metric records with explicit physical units.
- Target-volume GPU evaluation and centered local gradients.
- Modular native sources, examples and bilingual documentation.
