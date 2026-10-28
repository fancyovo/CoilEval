# Changelog

## 0.8.0 (unreleased)

- Overlap independent two-stage domain preparation and CPU engineering derivatives with GPU evaluation.
- Batch optional target-surface magnetic sweeps across independent queries.
- Defer stage timing to CUDA events to avoid per-stage device barriers.
- Preserve detailed native capture errors and report unsupported concurrent solver captures explicitly.
- Add execution controls, independent-query regression and GPU parity benchmarks.

## 0.7.0 (unreleased)

- Add CPU evaluation and local reverse gradients, with explicit or adaptive thread limits.
- Add CPU-only magnetic-axis grid selection while retaining the original default and fallback search.
- Reduce repeated boundary-check work and tensor feature launches.
- Add an optional native radial-root operator with compatibility fallback for older libraries.
- Provide CPU and GPU installation extras, bilingual guides and a CPU example.

## 0.6.0 (unreleased)

- Add optional target-surface f_B values, local reverse gradients and independent batches.
- Refine the requested surface angle and fit vacuum-potential Boozer coordinates, with configurable accuracy presets.
- Add optional f_T using analytic field derivatives and flux-shell averaging.
- Reuse native psi/alpha captures for value-only calls and share magnetic-axis preparation across independent queries.
- Bound Fourier workspaces and accelerate large target-angle fits with mixed-precision preconditioning, FP64 residual checks and a direct-solve fallback.
- Add independent Simsopt metric references, coordinate diagnostics and bilingual examples.


## 0.5.1 (unreleased)

- Evaluate flux-Legendre alpha modes with a stable three-term recurrence in native, tensor and Simsopt initialization paths, reducing cancellation at higher radial orders without changing the finite basis.

- Reject a Simsopt surface when a geometric-section candidate is confirmed as a self-intersection on the original Fourier surface. Inconclusive searches retain an explicit quality warning; a negative sampled search is not a global embedding certificate.
- Check target-surface radial capacity on at least the psi fitting toroidal grid before accepting a target volume. Flux integration and local volume derivatives retain their configured quadrature. This sampled guard can reject surfaces previously accepted on the sparse flux grid.
- Collapse toroidal polynomial coefficients once per section to reduce repeated work in the capacity guard.

## 0.5.0a1

- Accelerated coil geometry metrics while retaining them in complete value-and-gradient evaluation.
- Kept GPU triangular solves independent per query and bounded temporary memory for larger batches.
- Preserved formal volume-point selection in the local gradient and documented its discrete sampling boundaries.
- Validated tensor Stokes flux against the formal center and added differentiable area integration for queries exceeding the configured tolerance.
- Added optional `flux_legendre` alpha fitting for two-stage evaluations and matching source initialization.
- Shared magnetic-axis preparation across independent tensor batches and repeated two-stage fits across volumes of the same geometry.
- Reduced source-capture work and made torsion diagnostics optional during source preparation.
- Preserved the source magnetic axis in saved initializers and standard surfaces, and added angular-covering acceptance checks.
- Aligned native extraction and tensor devices in alpha+nu initialization.
- Resolved ambiguous radial derivative roots during two-stage boundary construction.
- Clarified metric averaging regions, batch configuration requirements and forward-versus-gradient timing.

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
