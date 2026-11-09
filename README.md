# CoilEval

[简体中文](README.zh-CN.md) · [Install](#installation) · [Example1: evaluate](#example1) · [Example2: optimize](#example2) · [Accuracy and speed](#validation)

CoilEval takes stellarator coil shapes, currents, field-period counts and a target
volume, then estimates magnetic surfaces and computes quasisymmetry and coil
geometry metrics on CPU or NVIDIA GPU. It also provides local parameter gradients.
Coil shapes are Fourier coefficients: sine and cosine terms describing each closed
curve. The target volume selects the enclosed volume of the surface to evaluate.

A local gradient describes small parameter changes while retaining the selected axis and sampling branches. Rotational transform ι is the average poloidal turns per toroidal turn of a field line.

**Version 0.11.2, development candidate.** By default, a scalar surface-label function
is fitted first in an initial region and then again in the closed region selected
from that fit. An optional third fit is enabled by `psi_target_refit=True`; it is
off by default.

<a id="installation"></a>
## Install

Linux x86-64, glibc 2.28+ and Python 3.10+ are supported. A wheel (`.whl`) is a
precompiled Python installation package. The separate source archive contains
`examples/`. **Extract that archive and run examples from its root directory.**

GPU: install the supplied wheel and CUDA 12.8 runtime dependencies. Example2 also
needs CUDA-enabled PyTorch:

```sh
python -m pip install "./coil_eval-0.11.2-py3-none-manylinux_2_28_x86_64.whl[gpu]"
python -m pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
```

For CPU use, run both commands in the same Python environment. The first selects
the CPU build of PyTorch, which CoilEval uses for linear algebra. The second
installs CoilEval with SciPy and threadpoolctl.

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.11.2-py3-none-manylinux_2_28_x86_64.whl[cpu]"
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
weights. There are no command-line options, shard files or fixed input hashes.
Each Example1 coefficient is perturbed as `c*exp(log(1.01)*z)` with independent
standard-normal z; screening currents stay fixed. The valid candidate with the
lowest objective starts optimization of coefficients and currents.

The objective combines volume QS, rotational transform, coil gap, curvature,
length and normal drift. SciPy BFGS updates a curvature approximation from gradients
provided through the public `TensorEvaluator` and PyTorch interfaces.
**Each screened candidate and each optimizer iteration prints a line** containing
the objective, QS, iota, coil metrics and elapsed time. Final output includes the
stopping reason, final metrics and total time, plus
`runs/example2/optimized_coils.json` and `history.csv`. Invalid trials are identified;
an optimizer stop or iteration limit is separate from physical certification.

Edit and rerun either script; change `OUTPUT` to retain multiple runs.
See [all examples](examples/README.md) for batches, gradients, surface metrics and
Simsopt. The simpler 0.11.1 teaching objective differs from the old preview
reproduction experiment; its older trajectory is labeled historical validation.

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


The fixed cohort contains **4096** tasks: 1024 inputs from the QUASR coil-configuration database at four noise scales.
The default two-fit procedure produces **2127** complete scores, **1944** matches to saved Simsopt
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

Every entry below is **seconds per sample**: the median of three complete calls after initialization and an additional warmup, divided by batch size. Gradient mode includes one complete objective
reverse. Batch amortization measures throughput; scalar rows give single-query latency.

| Interface / metric | Mode | CPU 8 threads | CPU 16 threads | CPU 32 threads | RTX 5090 | A100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single query / volume f_C | value | 22.922 | 18.195 | 15.938 | 1.775 | 1.911 |
| single query / volume f_C | value + gradient | 40.000 | 28.481 | 21.838 | 2.168 | 2.397 |
| batch 1 / volume f_C | value | 22.843 | 20.695 | 15.847 | 1.785 | 1.915 |
| batch 1 / volume f_C | value + gradient | 42.364 | 29.731 | 22.055 | 2.159 | 2.410 |
| batch 2 / volume f_C | value | 31.186 | 29.588 | 20.676 | 1.502 | 1.606 |
| batch 2 / volume f_C | value + gradient | 50.412 | 38.103 | 26.830 | 1.572 | 1.823 |
| batch 4 / volume f_C | value | 30.449 | 25.305 | 19.715 | 1.347 | 1.453 |
| batch 4 / volume f_C | value + gradient | 47.093 | 33.554 | 25.651 | 1.418 | 1.633 |
| batch 8 / volume f_C | value | 30.669 | 24.617 | 19.702 | 1.296 | 1.396 |
| batch 8 / volume f_C | value + gradient | 47.288 | 49.192 | 34.473 | 1.375 | 1.547 |
| single query / surface f_T | value | 36.701 | 28.010 | 26.115 | 2.024 | 2.057 |
| single query / surface f_T | value + gradient | 56.714 | 40.632 | 34.375 | 2.585 | 2.584 |

CPU8 and GPU measurements date from 2026-11-05; CPU16/32 were measured on 2026-11-06 with matching inputs, numerical configuration and libraries. All CPU columns use logical threads on Xeon Gold 6530; GPU tests retain eight host threads. Four coil sets with Fourier expansions through order 16 form eight fixed queries.
Tests use scheduler-assigned CPUs without exclusive whole-node allocation. Additional threads do not guarantee a speedup for every workload; all measured results are retained. Volume f_C and optional surface f_T are separate
workloads. Cold calls, whole-batch latency, hardware and hashes remain in the
[raw timing evidence](docs/assets/preview_s2_20261105/timings.json). See the [validation record](docs/preview_validation.md)
for cohort definitions, failures, reference qualifications and numerical limitations.


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
