# CoilEval

[简体中文](README.zh-CN.md) · [Install](#installation) · [Example1: evaluate](#example1) · [Example2: optimize](#example2) · [Accuracy and speed](#validation)

CoilEval takes stellarator coil shapes, currents, field-period counts and a target
volume, then estimates magnetic surfaces and computes quasisymmetry and coil
geometry metrics on CPU or NVIDIA GPU. It also provides local parameter gradients.
Coil shapes are Fourier coefficients: sine and cosine terms describing each closed
curve. The target volume selects the enclosed volume of the surface to evaluate.

A local gradient describes how the result changes for small parameter changes
along the selected solution branch. The magnetic axis can move along that branch.
Rotational transform ι is the average poloidal turns per toroidal turn of a field line.

**Version 0.12.0.** The evaluator estimates magnetic surfaces as level sets of a
scalar function `s`. By default it fits `s` first in a circular sampling region
and then in a region bounded by a level set from the first fit.
Set `psi_fit="near_axis_ellipse"` to fit once in a region predicted
from the field close to the magnetic axis. This option supports values and local
gradients on CPU and GPU. `psi_target_refit=True` selects the optional third fit
after the two-fit procedure. See [configuration and limitations](docs/api.md).

<a id="installation"></a>
## Install

Linux x86-64, glibc 2.28+ and Python 3.10+ are supported. A wheel (`.whl`) is a
precompiled Python installation package. See the [source build instructions](docs/installation.md)
to build a wheel from this checkout. Run the examples from the root of the
checkout or extracted source archive.

GPU: install the supplied wheel and CUDA 12.8 runtime dependencies. Example2 also
needs CUDA-enabled PyTorch:

```sh
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
python -m pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
```

For CPU use, run both commands in the same Python environment. The first selects
the CPU build of PyTorch, which CoilEval uses for linear algebra. The second
installs CoilEval with SciPy and threadpoolctl.

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

The PyTorch commands above use the verified builds for Python 3.10–3.13. GPU
operation needs a compatible NVIDIA driver. See the [installation guide](docs/installation.md)
for build selection and environment checks.

<a id="example1"></a>
## Example1: read coils and evaluate

From the source archive root, run:

```sh
python examples/example1.py
```

[example1.py](examples/example1.py) directly shows `CoilSet` for the input,
`EvaluationConfig` for the target, and `Evaluator` for computing metrics.
Defaults are CPU, eight threads and 0.3 m³ target volume. Edit `COILS`,
`TARGET_VOLUME`, `FITTING_RADIUS`, `DEVICE` and the other settings at the top.
The fitting radius is in metres; `helicity=(1,0)` selects quasi-axisymmetry,
meaning field strength independent of the toroidal surface angle.

The script prints status, volume/outer-region QS residuals, boundary rotational
transform, normal drift, coil length/gap/curvature, current and elapsed times.
It saves `runs/example1/result.json`. A valid result still needs its accuracy
diagnostics to be assessed.

<a id="example2"></a>
## Example2: screen 32 starts and optimize in one run

The default uses a GPU with CUDA-enabled PyTorch and SciPy. After installation,
run one script:

```sh
python -m pip install scipy
python examples/example2.py
```

[example2.py](examples/example2.py) keeps `N_STARTS`, `NOISE`, `MAX_STEPS`, device
and parameter scales at the top. Edit `objective(metrics)` to try other goals or
weights.
Each Example1 coefficient is perturbed as `c*exp(log(1.01)*z)` with independent
standard-normal z; screening currents stay fixed. The valid candidate with the
lowest objective starts optimization of coefficients and currents.

The objective combines volume QS, rotational transform, coil gap, curvature,
length and normal drift. SciPy BFGS updates a curvature approximation from gradients
provided through the public `TensorEvaluator` and PyTorch interfaces.
**Each screened candidate and each optimizer iteration prints a line** containing
the objective, QS, iota, coil metrics and elapsed time. Final output includes the
stopping reason, final metrics and total time, plus
`runs/example2/optimized_coils.json` and `history.csv`. The stopping reason is
the status returned by SciPy. The output also identifies invalid trial points.

Edit and rerun either script. Change `OUTPUT` to retain multiple runs.
To run either script with a single ellipse fit, set `PSI_FIT="near_axis_ellipse"`
and `TARGET_VOLUME=0.15`. The bundled coils reach 0.3 m³ with two fits. Their
one-fit scalar turns back before that volume, so the ellipse example uses 0.15 m³.
See [all examples](examples/README.md) for batches, gradients, surface metrics and
Simsopt.

<a id="validation"></a>
## Accuracy and speed

On this page, **quasisymmetry (QS)** means that magnetic-field strength depends on one
helical angle in suitable surface coordinates. **QA** is quasi-axisymmetry and **QH**
is quasi-helical symmetry. `nfp` is the number of repeated field periods around the
device; `helicity=(M,N)` selects the angle `Mθ−Nφ`, with poloidal θ around a surface
cross-section and toroidal φ around the device. **Rotational transform ι (iota)** is
the average number of poloidal turns per toroidal turn of a field line. **RMS** is
the square root of a weighted mean of squares; **P90/P95** are the 90th/95th
percentiles. `f_B` measures field-strength Fourier content outside the requested
symmetry; `f_C` is a differential quasisymmetry residual; `f_T` is a triple-product
residual involving field derivatives. All have ideal value zero; comparisons
require matching averaging regions and normalization.


The fixed two-fit calibration contains **4096** tasks drawn from 1024 inputs in the QUASR coil-configuration database at four noise scales.
CoilEval 0.11.0 produced **2127** complete scores, **1944** matches to saved Simsopt
references from an independent surface-optimization library, and **817** pairs meeting the original reference-quality rules.
All large errors remain. The 1981 completed references are reused unchanged;
tasks without references contribute to success counts only.

![Four CoilEval–Simsopt comparisons, each n=1944, with empirical P90 error bands](docs/assets/preview_s2_20261105/calibration_p90.png)

QS bands use `exp(±P90(abs(ln(prediction/reference))))`; iota bands use
`±P90(abs(prediction-reference))`. They describe empirical error coverage.
The three QS panels use factors 1.986, 1.368 and 1.157 for the 1944 paired samples. The table separately reports P90 relative error.

| Population | Pairs | f_B P90 relative error | f_C P90 relative error | f_T P90 relative error | ι P90 absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| All paired | 1944 | 93.86% | 34.67% | 15.20% | 0.01270 |
| Reference-qualified | 817 | 87.04% | 30.17% | 12.17% | 0.01015 |

The 0.12.0 gradient check used 49 coil and volume combinations, including cases
where only one method can find the requested surface. Both methods returned a
gradient on 36. The median ellipse-to-two-fit objective gradient norm ratio was
1.002 and the median cosine was 0.997. The [gradient validation](docs/torch.md#gradient-validation)
reports the individual metrics, the lower tail and checks against perturbed evaluations.

Each cell gives **two fits → one ellipse fit** in seconds per sample. Each timing uses the median of three complete calls after one cold call and one additional warmup. Batch entries divide whole-batch wall time by batch size. Value + gradient includes the complete evaluation, objective calculation and one backward pass. The surface rows use the high-accuracy f_T calculation.

| Interface and metric | Work | CPU 8 threads | CPU 16 threads | CPU 32 threads | RTX 5090 | A100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single query / volume f_C | Value | 22.353 → 19.199 | 24.148 → 19.617 | 18.027 → 16.990 | 1.769 → 1.521 | 1.921 → 1.595 |
| Single query / volume f_C | Value + gradient | 39.172 → 34.515 | 31.516 → 26.080 | 23.293 → 21.726 | 2.212 → 1.925 | 2.586 → 2.197 |
| Batch 1 / volume f_C | Value | 25.339 → 20.703 | 48.013 → 31.804 | 15.847 → 15.355 | 1.775 → 1.524 | 1.922 → 1.599 |
| Batch 1 / volume f_C | Value + gradient | 41.875 → 36.780 | 55.598 → 41.358 | 22.055 → 21.204 | 2.211 → 1.919 | 2.405 → 2.060 |
| Batch 2 / volume f_C | Value | 32.756 → 29.227 | 56.263 → 44.451 | 23.778 → 20.620 | 1.471 → 1.302 | 1.664 → 1.348 |
| Batch 2 / volume f_C | Value + gradient | 47.013 → 43.238 | 36.030 → 33.577 | 26.830 → 29.572 | 1.595 → 1.655 | 1.821 → 1.712 |
| Batch 4 / volume f_C | Value | 33.415 → 30.437 | 25.906 → 21.746 | 19.715 → 18.361 | 1.350 → 1.190 | 1.472 → 1.207 |
| Batch 4 / volume f_C | Value + gradient | 49.788 → 46.488 | 66.042 → 51.706 | 25.651 → 26.425 | 1.436 → 1.504 | 1.639 → 1.561 |
| Batch 8 / volume f_C | Value | 31.428 → 27.965 | 32.487 → 22.986 | 23.650 → 20.764 | 1.292 → 1.115 | 1.419 → 1.139 |
| Batch 8 / volume f_C | Value + gradient | 48.061 → 41.240 | 32.256 → 28.933 | 34.473 → 29.357 | 1.475 → 1.699 | 1.692 → 1.607 |
| Single query / surface f_T | Value | 35.818 → 32.389 | 27.493 → 24.841 | 26.115 → 21.671 | 2.052 → 1.766 | 2.092 → 1.754 |
| Single query / surface f_T | Value + gradient | 56.809 → 51.793 | 74.775 → 53.558 | 34.375 → 30.751 | 2.594 → 2.332 | 2.612 → 2.243 |

Both methods use the same four coil sets and eight fixed coil/volume combinations. GPU timings wait for device completion and use eight host threads. Timings use scheduler allocations on shared nodes. The [timing records](docs/assets/ellipse_012/timings.json) include every warm repeat, cold and warmup times, and hardware details. Eight 32-thread two-fit entries reuse 2026-11-06 measurements with matching inputs, configuration, CPU model, native library and warmup procedure. See the [validation record](docs/preview_validation.md) for the historical two-fit calibration, reference qualifications and numerical limits.

## Scope and further documentation

Errors can come from finite-order surface fits, regularization bias, angular
discretization and amplification near resonances. Even smooth Poincare sections
(field-line intersections with fixed cross-sections) can accompany large metric
errors. The historical preview examples retain force-residual and reference-surface quality limits;
see the [validation report](docs/preview_validation.md).

- [API, metrics and configuration](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md), [CPU operation](docs/cpu.md)
- [Surface quasisymmetry metrics](docs/qs.md), [Simsopt independent surface solves](docs/simsopt.md)
- [All examples](examples/README.md), [versions](docs/versions.md), [changelog](CHANGELOG.md), [development](docs/development.md)

License: [MIT](LICENSE).
