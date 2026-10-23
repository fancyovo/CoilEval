# Changelog

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
