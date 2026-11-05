# CoilEval

[简体中文](README.zh-CN.md)

CoilEval evaluates stellarator Fourier coils on NVIDIA GPUs and CPUs. Supply
coefficients, currents, field periods and a target volume to obtain quasi-symmetry,
rotational transform, surface quality and coil geometry metrics. Compose a scalar
objective and compute its local parameter gradient.

The **0.11.0 preview candidate** defaults to validated S2: two scalar fits,
fast/Stokes, polar quadrature, smooth volume-alpha weights and flux-Legendre
volume-alpha basis. Angle-inverse and first-radial-branch repairs are included.
S3 remains optional through `psi_target_refit=True`, disabled by default.

## Installation and quick start

Use Linux x86-64 with glibc 2.28+ and Python 3.10+. Install the candidate wheel:

```sh
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

The GPU extra uses the validated CUDA 12.8 runtime set. The driver must support
CUDA 12.8 and the GPU. For CPU, install CPU PyTorch and the wheel's `[cpu]` extra.
See [installation](docs/installation.md) for dependencies, architectures and source builds.

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator(config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

For CPU use `Evaluator(config=config, device="cpu", threads=8)`.
QA requires `helicity=(1,0)`; standard QH uses `(1,coils.nfp)`. The compatibility
helicity default remains `(1,4)`. Tensor inputs carry no field-period metadata;
pass `nfp=coils.nfp` to `TensorEvaluator`.

Install CUDA-enabled PyTorch before running the autograd and target-surface QS examples.
From the extracted source archive:

```sh
python examples/evaluate.py
python examples/autograd.py --batch
python examples/qs_metrics.py --gradient --metric f_B
```

`qs_volume` and `qs_outer` are differential f_C RMS quantities averaged over
their volume and outer radial bin. Target-surface Fourier f_B, triple-product f_T,
Simsopt and DESC references have explicitly different measures and normalizations;
see [QS](docs/qs.md) and [reference definitions](docs/simsopt.md).

## S2 accuracy and speed

The fixed cohort contains **4096** tasks: 1024 QUASR inputs at four noise scales.
Repaired S2 produces **2127** complete scores, **1944** matches to frozen Simsopt
references, and **817** pairs meeting the original reference-quality rules.
All large errors remain. The 1981 completed references are reused unchanged;
tasks without references contribute to success counts only.

![CoilEval S2 versus Simsopt QS and iota, with measured P90 bands for each population](docs/assets/preview_s2_20261105/calibration_p90.png)

QS bands use `exp(±P90(abs(ln(prediction/reference))))`; iota bands use
`±P90(abs(prediction-reference))`. They describe empirical error coverage.
All-paired QS factors are 1.986, 1.368 and 1.157; reference-qualified factors are
1.926, 1.304 and 1.122. The table separately reports P90 relative error.

| Population | Pairs | f_B P90 relative error | f_C P90 relative error | f_T P90 relative error | ι P90 absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| All paired | 1944 | 93.86% | 34.67% | 15.20% | 0.01270 |
| Reference-qualified | 817 | 87.04% | 30.17% | 12.17% | 0.01015 |

Every entry below is **seconds per sample**: the median of three warm complete
calls, divided by batch size. Gradient mode includes one complete objective
reverse. Batch amortization measures throughput; scalar rows give single-query latency.

| Interface / metric | Mode | CPU 8 threads | RTX 5090 | A100 |
| --- | --- | ---: | ---: | ---: |
| scalar / volume f_C | value | 22.922 | 1.775 | 1.911 |
| scalar / volume f_C | value + gradient | 40.000 | 2.168 | 2.397 |
| batch 1 / volume f_C | value | 22.843 | 1.785 | 1.915 |
| batch 1 / volume f_C | value + gradient | 42.364 | 2.159 | 2.410 |
| batch 2 / volume f_C | value | 31.186 | 1.502 | 1.606 |
| batch 2 / volume f_C | value + gradient | 50.412 | 1.572 | 1.823 |
| batch 4 / volume f_C | value | 30.449 | 1.347 | 1.453 |
| batch 4 / volume f_C | value + gradient | 47.093 | 1.418 | 1.633 |
| batch 8 / volume f_C | value | 30.669 | 1.296 | 1.396 |
| batch 8 / volume f_C | value + gradient | 47.288 | 1.375 | 1.547 |
| scalar / surface f_T | value | 36.701 | 2.024 | 2.057 |
| scalar / surface f_T | value + gradient | 56.714 | 2.585 | 2.584 |

These are the 2026-11-05 measurements of the same S2 recipe and native library:
four original order-16 coil sets, eight fixed queries, and eight assigned logical
CPUs per hardware allocation. Volume f_C and optional surface f_T are separate
workloads. Cold calls, whole-batch latency, hardware and hashes remain in the
[raw timing evidence](docs/assets/preview_s2_20261105/timings.json). See the [validation record](docs/preview_validation.md)
for cohort definitions, failures, reference qualifications and numerical limitations.

## Examples and limitations

Example1 reruns the original QA coils at V=0.3 m³ and a=√0.06 m through fresh
S2 source→alpha→nu→Simsopt→DESC. The workflow completes, DESC reaches gtol and
the final surface is nested. Residual forces remain visible; equilibrium certification
is false. Example2 independently perturbs example1 coefficients as
`c*exp(log(1.01)*z)`, keeps screening currents unchanged, selects the best full
objective among 32 candidates, and enters analytic-gradient BFGS. At 1% noise,
25/32 score successfully; 116 updates reduce the objective from −3.6904 to
−5.4891. The final full reference workflow completes. See the
[validation record](docs/preview_validation.md) for residual-force limits and stopping reasons.

The fixed 4096 S3 comparison scores 1821 inputs, losing 314 S2 successes and
recovering 8; S3 remains optional. Remaining S2 errors include the full-volume
scalar-fit objective and ridge bias, angular-equation discretization, resonant
error amplification and coordinate representation limits. Smooth Poincaré
contours alone do not guarantee accurate small-QS estimates. Finite diagnostics
do not certify global magnetic topology.

## Documentation

- [API and defaults](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md)
- [CPU](docs/cpu.md) and [Simsopt](docs/simsopt.md)
- [Examples](examples/README.md), [versions](docs/versions.md), [changelog](CHANGELOG.md)
- [Development](docs/development.md)

License: [MIT](LICENSE).
