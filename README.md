# CoilEval

[简体中文](README.zh-CN.md) · [Install](#installation) · [Example1: evaluate](#example1) · [Example2: optimize](#example2) · [Accuracy and speed](#validation)

CoilEval takes stellarator coil shapes, currents, field-period counts and a target
volume, then estimates magnetic surfaces and computes quasisymmetry and coil
geometry metrics on CPU or NVIDIA GPU. It also provides local parameter gradients.
Coil shapes are Fourier coefficients: sine and cosine terms describing each closed
curve. The target volume selects the enclosed volume of the surface to evaluate.

A local gradient describes small parameter changes while retaining the selected axis and sampling branches. Rotational transform ι is the average poloidal turns per toroidal turn of a field line.

**Version 0.11.0, preview candidate.** By default, a scalar surface-label function
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
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

CPU: run both commands in the same Python environment. The first explicitly selects
CPU PyTorch; the name `[cpu]` alone does not make pip select that Torch distribution.
CPU value evaluation also uses Torch linear algebra.

```sh
python -m pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

GPU operation needs a compatible NVIDIA driver; CPU operation does not need CUDA.
See the [installation guide](docs/installation.md) for environments and source builds.

<a id="example1"></a>
## Example1: evaluate the supplied coils

This example uses `examples/preview/example1.json` at volume 0.3 m³. `psi_a` is the
fitting radius around the magnetic axis, in metres. `helicity=(1,0)` requests
quasi-axisymmetry: field strength independent of the toroidal surface angle.

```python
import math
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/preview/example1.json")
config = EvaluationConfig(target_volume=0.3, psi_a=math.sqrt(0.06), helicity=(1, 0))
with Evaluator(config=config, device="cpu", threads=8) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print("Volume quasisymmetry residual:", result.metrics.qs_volume)
        print("Boundary rotational transform:", result.metrics.iota_boundary)
    else:
        print("Evaluation failed:", result.status)
```

For GPU, use `Evaluator(config=config, device="cuda")`.
`valid` means the numerical procedure returned a usable result; it does not certify
global magnetic surfaces or sufficient accuracy.

<a id="example2"></a>
## Example2: screen 32 perturbed starts and optimize the best

**Run `examples/preview/perturb_optimize.py` with an NVIDIA GPU and the GPU/PyTorch
installation above.** From the extracted source root:

```sh
python examples/preview/perturb_optimize.py screen --output runs/example2-screen
python examples/preview/perturb_optimize.py optimize --screens runs/example2-screen/screen.json --output runs/example2-optimize
```

The first command makes 32 candidates using `c*exp(log(1.01)*z)` for each Example1
coefficient, with independent standard-normal z and unchanged screening currents.
The second selects the valid candidate with the **smallest complete objective**,
then optimizes coefficients and currents using BFGS, a quasi-Newton method that
updates its search direction using gradients. The objective includes quasisymmetry,
rotational transform, coil gap, curvature, length, volume and surface quality.

`screen.json` retains every candidate status. The optimization directory contains
`history.json` for iterations, `selected.npz` for the best coil parameters and
`result.json` for results and stopping reasons. Output directories must be new.
See the [complete Example2 guide](examples/preview/README.md) for the fixed seed,
targets, multiple GPUs and smaller perturbations.

The tested 1% perturbation produced 25/32 valid candidates and 116 accepted updates;
the volume quasisymmetry residual fell from 0.01786 to 0.003897. Consecutive rejected
trials stopped optimization; convergence was not established. These commands run
CoilEval optimization. Separate Simsopt surface solves and DESC equilibrium checks
performed for release validation are documented in the [validation report](docs/preview_validation.md).

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
errors. Both examples retain force-residual and reference-surface quality limits;
see the [validation report](docs/preview_validation.md).

- [API, metrics and configuration](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md), [CPU operation](docs/cpu.md)
- [Surface quasisymmetry metrics](docs/qs.md), [Simsopt independent surface solves](docs/simsopt.md)
- [All examples](examples/README.md), [versions](docs/versions.md), [changelog](CHANGELOG.md), [development](docs/development.md)

License: [MIT](LICENSE).
