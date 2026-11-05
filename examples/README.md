# Examples

[evaluate.py](evaluate.py) loads [coils.json](coils.json) and evaluates its metrics.
[custom_loss.py](custom_loss.py) composes a scalar objective and returns its full physical gradient.
The scripts load the installed wheel's backend automatically. Pass
`--library /path/to/libcoil_eval_native.so` to select a custom backend.
Use `--help` for options.

[cpu.py](cpu.py) evaluates coils on CPU. Install the `cpu` extra and CPU PyTorch
as described in the [CPU guide](../docs/cpu.md), then run
`python examples/cpu.py --threads 8`. S2 is the default; use `--axis-grid 32` to select a smaller primary magnetic-axis seed grid.

[autograd.py](autograd.py) evaluates a GPU PyTorch objective and its gradient. Add `--batch` for independent geometries. Install CUDA-enabled PyTorch and the `gpu` extra to run it.

[qs_metrics.py](qs_metrics.py) evaluates target-surface f_B. Use `--gradient`
for a local parameter gradient, `--metric f_T` to include the triple-product
metric, and `--accuracy high|balanced|fast` to select the angular resolution.
It uses the CUDA-enabled PyTorch setup in the [installation guide](../docs/installation.md).

[two_stage.py](two_stage.py) demonstrates the default S2 fit. All evaluation
examples use polar quadrature, smooth alpha weights and flux-Legendre alpha.
Use `--coils`, `--volume`, `--radius` and `--output` to select your inputs.
`--two-stage` remains an accepted compatibility flag in CPU/autograd examples.
`autograd.py --batch` reports seconds per sample, including one full objective
reverse; the first call includes cold initialization.

```sh
python examples/two_stage.py --output output/metrics.json
python examples/autograd.py --batch
```

[Preview examples](preview/README.md) include the frozen QA example1 input and
example2's reproducible 1% multiplicative perturbations,32-candidate screening,
and dense inverse BFGS with analytic gradients. Their fixed target is V=0.3m³.

[simsopt_async.py](simsopt_async.py) separates GPU source capture and alpha+nu
preparation from CPU-only standard Simsopt solves. Install Simsopt and the
CUDA-enabled PyTorch dependency for preparation. The solve phase uses spawned
processes with one BLAS/OpenMP thread each and no visible GPU.

```sh
python examples/simsopt_async.py prepare --directory output/case1 --volumes 0.01 0.03 0.05
python examples/simsopt_async.py solve --directory output/case1 --workers 4
```

Preparation exits after saving ready tasks and failures. CPU jobs can solve
completed directories while another GPU job prepares a new geometry. Match
`--workers` to the CPU allocation. Accepted surfaces receive flux-shell f_C RMS
diagnostics; the saved native result also retains volume and outer-bin QS RMS.
See [the Simsopt API](../docs/simsopt.md) for these averaging conventions.

Add `--alpha-fold-policy damp --alpha-min-jacobian 0.05` to `prepare` to use
coordinate damping for folded alpha maps. The default policy is `reject`.
The solve phase still evaluates each resulting standard surface independently.
