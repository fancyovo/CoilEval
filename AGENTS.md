# Contributor instructions

Read README.md, docs/api.md and CHANGELOG.md before changing public behavior.

- Keep the Python API backward compatible within the 0.x series. Add options
  with defaults that preserve existing calls. Preserve parameter order and units.
- Keep numerical settings explicit in EvaluationConfig. Separate metrics from
  application-specific objectives and parameter normalization.
- Organize native code by field and evaluation stage. Compile implementation
  files separately; keep complete functions together and state ownership clear.
- Keep optional dependencies behind their corresponding entry points.
- Update English and Chinese documentation together. Document installation,
  inputs, outputs and runnable examples in direct, user-oriented language.
- Add focused regression tests for behavior changes and preserve failure status.
  Test independent batches as well as neighborhoods when either is affected.
- Run pytest, source-layout checks and distribution builds before tagging.
  Changes to native code also require a CUDA build and GPU regression.
- Binary wheels target CUDA 12.8 and Linux x86-64 with glibc 2.28+. Keep
  runtime dependencies, architecture targets and installation docs aligned.
  Test installed wheels with nvcc absent from PATH and automatic library loading.
  Preserve explicit native-library paths and immutable earlier release tags.
- Pin NVIDIA runtime distributions as a validated set. Individual component
  version numbers do not identify the CUDA Toolkit release; validate dependency
  resolution against the package index as well as the offline build wheelhouse.
- Future versions use `major.minor.patch` without an `a` suffix. New features
  increment the second (minor) component and reset patch to zero. Every released
  bug fix, including a small or obvious correction, increments the third (patch)
  component. Do not reuse a version for changed implementation.
- Preserve historical version numbers, release artifacts and tags, including
  existing `a` versions. This policy does not renumber earlier releases.
- Follow ascending release versions. Maintain CHANGELOG.md and immutable tags.
  Publishing requires the repository owner's approval.
