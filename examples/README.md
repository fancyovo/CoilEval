# Running the examples

[简体中文](README.zh-CN.md)

Run these commands from the extracted source archive, after [installing CoilEval](../docs/installation.md).
The wheel supplies the library; the source archive also supplies these scripts and input files.

## Start here

- **Example 1: evaluate fixed coils.** The [README commands](../README.md#example1) load `preview/example1.json` and score a target volume of 0.3 m³ on CPU or GPU.
- **Example 2: perturb, select and optimize.** The [screening and optimization commands](preview/README.md) generate 32 approximately 1% multiplicative perturbations of example 1 and optimize the best valid candidate. This script requires a GPU and CUDA-enabled PyTorch.

## Other interface examples

Quasisymmetry (QS) means that field strength depends on one helical surface angle.
`f_B` measures Fourier content outside that symmetry, `f_C` is a differential residual, and `f_T` a field-derivative triple-product residual; each has ideal value zero.
The default first fits a surface label in an initial region, then refits it inside a selected closed region. These examples use that procedure.

| Script | Purpose and command |
| --- | --- |
| [evaluate.py](evaluate.py) | Score [coils.json](coils.json): `python examples/evaluate.py` |
| [cpu.py](cpu.py) | CPU values: `python examples/cpu.py --threads 8` |
| [custom_loss.py](custom_loss.py) | Custom scalar objective and parameter gradient: `python examples/custom_loss.py` |
| [autograd.py](autograd.py) | PyTorch derivatives: `python examples/autograd.py`; add `--batch` for independent inputs |
| [qs_metrics.py](qs_metrics.py) | Target-surface f_B: `python examples/qs_metrics.py`; add `--gradient --metric f_T` for f_T and its gradient |
| [two_stage.py](two_stage.py) | Explicit two-fit example: `python examples/two_stage.py --output runs/metrics.json` |

GPU scripts automatically load the installed backend; `--library /path/to/libcoil_eval_native.so` selects a custom library.
See the [CPU guide](../docs/cpu.md) for CPU dependencies and thread limits. PyTorch examples also require PyTorch for the chosen device.
Use each script’s `--help` for input options. The retained `--two-stage` compatibility flag is redundant in 0.11.0.
`autograd.py --batch` reports seconds per sample including one full backward pass; the first call includes initialization.

## Independent surface validation

[simsopt_async.py](simsopt_async.py) prepares a surface and angle-coordinate initializer on GPU, then uses the independent Simsopt library to optimize the surface on CPU.
Preparation needs CUDA PyTorch and Simsopt. CPU solve processes each use one linear-algebra/parallelism thread (BLAS/OpenMP).

```sh
python examples/simsopt_async.py prepare --directory runs/case1 --volumes 0.01 0.03 0.05
python examples/simsopt_async.py solve --directory runs/case1 --workers 4
```

Preparation saves completed tasks and failures. CPU solves can overlap preparation of other geometries; keep `--workers` within the CPU allocation.
Results include f_C RMS (square root of mean square) over a thin magnetic-flux shell, reported separately from volume averages.
Preparation optionally accepts `--alpha-fold-policy damp --alpha-min-jacobian 0.05` to reduce an angular correction until the map is monotone; the default rejects folded maps.
This changes the initializer; each final surface still undergoes independent solver acceptance. See the [Simsopt guide](../docs/simsopt.md).
