# Example 1 and example 2

[简体中文](README.zh-CN.md)

Both examples start from the fixed coils in [example1.json](example1.json): four base coils with Fourier expansions through order 4 and `nfp=3` repeated field periods.
Rotations and stellarator reflection symmetry generate the full device.
The target is quasi-axisymmetry (QA): field strength is independent of the toroidal angle in suitable surface coordinates, selected by `helicity=(1,0)`.
The target volume is 0.3 m³ and initial fitting radius √0.06 m. Version 0.11.0 first fits a surface label, then refits it inside a selected closed region.

## Example 1: evaluate fixed coils

Install the CPU or GPU dependencies from the [installation guide](../../docs/installation.md).
From the extracted source archive, run the [complete Python example in the README](../../README.md#example1).
The [validation report](../../docs/preview_validation.md) describes its values, independent references and limitations.

## Example 2: screen 32 perturbations, then optimize

This script needs a GPU, CUDA-enabled PyTorch and CoilEval’s GPU dependencies. Run from the source archive root:

```sh
python examples/preview/perturb_optimize.py screen --output runs/example2-screen
python examples/preview/perturb_optimize.py optimize --screens runs/example2-screen/screen.json --output runs/example2-optimize
```

The first command generates and scores 32 candidates; the second chooses the valid candidate with the lowest complete objective and starts optimization.
Optional `--library /path/to/libcoil_eval_native.so` selects a custom library. Use fresh output directories to retain earlier runs.

Each geometric Fourier coefficient c independently becomes `c * exp(log1p(0.01) * z)`, where z is standard Gaussian.
The log perturbation therefore has standard deviation ln(1.01), preserving signs and exact zeros. The fixed seed is 2026110502.
All 32 candidates start from the original example 1. Screening keeps currents, volume and fitting radius fixed.
Optimization may vary all 108 geometric coefficients and four currents (112 parameters); bundled statistics provide fixed parameter-normalization scales.

The objective combines log volume f_C (a differential quasisymmetry residual) with rotational-transform, coil-gap, curvature, length, volume, normal-drift and axis-radius costs.
Rotational transform ι is the average poloidal turns per toroidal turn of a field line; normal drift measures departure from the fitted surface.
The five condition targets are ι=0.47, minimum gap 0.1 m, maximum curvature 6 m⁻¹, mean length 3.5 m and volume 0.3 m³.

BFGS is a quasi-Newton method that updates a curvature approximation from gradients; this example stores its full inverse matrix.
Powell damping stabilizes updates and Armijo backtracking shortens trial steps until the objective decreases sufficiently.
One backward pass supplies each gradient; trial points receive complete evaluations.
The default budget is 400 outer iterations, with an additional consecutive-rejection limit. Stopping does not establish a global optimum or physical certification.

`screen.json` retains all candidate statuses. `history.json` and `evaluations/` retain optimization and trial history;
`result.json` gives the stopping reason, and `start.npz`/`selected.npz` store initial/selected parameters.
Validation at 1% noise gave 25/32 valid candidates and 116 accepted optimization updates, stopping after consecutive rejections. No noise reduction was needed.

If a numerical attempt in a working environment cannot complete, retry `screen --fraction 0.005`, then 0.0025, and so on in new directories, retaining the same seed and Gaussian draws.
Keep volume and radius fixed and retain failures. Correct environment or schema errors first; judge each new run by its actual status.

For four allocated GPUs, run `screen --shards 4 --shard I` for I=0,1,2,3 with distinct output directories and one assigned GPU each.
Pass all four `screen.json` paths to `optimize --screens`; all 32 distinct indices are required. Optimization itself is one dependent iteration chain.

Release validation also uses Simsopt to optimize a reference surface and DESC to solve a magnetohydrodynamic equilibrium; these are independent physics libraries.
Successful scoring/optimization is reported separately from reference residuals, convergence and certification.
The original physical-array SHA256 is `aebf705719b77ad064624362f5d90601dc4b8a2cfbaf3f8240b427faf8e1eaf7` for input identity checks.

When reducing noise, pass the same `--fraction` to both `screen` and `optimize`, for example `--fraction 0.005`.
