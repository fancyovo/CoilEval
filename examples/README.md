# Examples

[evaluate.py](evaluate.py) loads [coils.json](coils.json) and evaluates its metrics.
[custom_loss.py](custom_loss.py) composes a scalar objective and returns its full physical gradient.
The scripts load the installed wheel's backend automatically. Pass
`--library /path/to/libcoil_eval_native.so` to select a custom backend.
Use `--help` for options.

[autograd.py](autograd.py) evaluates a PyTorch objective and its gradient. Add `--batch` for independent geometries. Install the `torch` extra to run it.

[two_stage.py](two_stage.py) selects the optional two-stage psi fit. Set
`--coils`, `--volume`, `--radius` and `--output` to use your own inputs.
`--sampling polar_quadrature` selects the experimental weighted sampler.

```sh
python examples/two_stage.py --output output/metrics.json
python examples/autograd.py --two-stage --batch
```

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
