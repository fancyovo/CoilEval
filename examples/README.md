# Running and editing the examples

[简体中文](README.zh-CN.md)

Install the library using the [installation guide](../docs/installation.md), then run scripts from the extracted source root.
Each example is self-contained: **edit its settings at the top and run `python examples/name.py`**.
No command-line options or preceding script are required. The default input is the bundled [example1.json](preview/example1.json).

| Script | Main objects and purpose |
| --- | --- |
| [example1.py](example1.py) | `CoilSet`, `EvaluationConfig`, `Evaluator`: input, metrics, timing and saved result; CPU default |
| [example2.py](example2.py) | `TensorEvaluator` + SciPy BFGS: 32 starts then optimization, with live candidate/step output; GPU default |
| [evaluate.py](evaluate.py) | Minimal GPU value evaluation |
| [cpu.py](cpu.py) | CPU values; edit `THREADS` at the top |
| [autograd.py](autograd.py) | `pack_coils`, a Torch objective and automatic gradients |
| [batch.py](batch.py) | `evaluate_batch`: two independent inputs and gradients, seconds/sample |
| [custom_loss.py](custom_loss.py) | NumPy objective and local finite differences with `loss_and_grad` |
| [qs_metrics.py](qs_metrics.py) | `SurfaceQSConfig`, `SurfaceQSEvaluator`: target-surface f_B and f_T |
| [two_stage.py](two_stage.py) | `evaluate_many`: three target volumes with the default two-fit procedure |
| [simsopt_async.py](simsopt_async.py) | GPU seed preparation and concurrent CPU surface solves in one run |

For example:

```sh
python examples/example1.py
python examples/example2.py
```

These are independent entry points: Example2 reads the original coils and does not consume Example1 output.
Every script prints results and elapsed time. First calls include initialization; the timing scope is labeled and differs from warmed benchmarks.
Rerunning updates the same output files; change `OUTPUT` to retain older runs.

To try the single ellipse fit in either script, set `PSI_FIT="near_axis_ellipse"`
and `TARGET_VOLUME=0.15`. This region is predicted from the magnetic field near
the axis. The fitted scalar for these coils turns back before 0.3 m³ with this
option. See the [API guide](../docs/api.md) for the available fit methods.

## Editing Example2

`N_STARTS=32`, `NOISE=0.01` and `SEED=2026110502` control screening. Perturbations are `c*exp(log1p(NOISE)*z)` with standard-normal z;
coefficient signs and exact zeros are preserved, and screening currents stay fixed. `MAX_STEPS=100` is the iteration budget.
Edit targets and weights directly in `objective(metrics)`: defaults are iota=0.47, minimum gap 0.10 m, maximum curvature 6 m⁻¹ and mean length 3.5 m.
`EvaluationConfig` fixes volume at 0.3 m³. `GEOMETRY_SCALE` and `CURRENT_SCALE` convert optimizer coordinates to metres and amperes.
Optimization changes coefficients and currents. If no screened candidate is valid, reduce `NOISE` and rerun.

QS is a quasisymmetry residual with ideal value zero; rotational transform iota is the average poloidal turns per toroidal field-line turn.
Normal drift measures how much field lines leave the fitted surface. Surface f_B uses field-strength Fourier content; f_T uses a field-derivative triple product.
BFGS updates a curvature approximation from gradients; see the [SciPy interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html).
`history.csv` stores objective, QS, iota and time for each accepted iterate; `optimized_coils.json` can be read with `CoilSet.read_json`.

## Example2 runs with CoilEval 0.12.0

Both runs on RTX 5090 completed screening and saved `optimized_coils.json`
and `history.csv`. SciPy's `success` flag reports whether its convergence
criteria were met. The default run stopped with precision loss, and the ellipse
run reached its 100-iteration limit.

| Fit | Target volume | BFGS iterations | Initial → final objective | Final volume QS | Final iota | Total time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two fits | 0.30 m³ | 92 | −2.763355 → −5.471306 | 0.00409435 | 0.469690 | 449.83 s |
| One ellipse fit | 0.15 m³ | 100 | 1.112674 → −6.261718 | 0.00184542 | 0.469527 | 915.24 s |

Total time includes screening, line-search trials and progress reporting.
The two configurations solve different target volumes. See the [warmed speed
comparison](../README.md#validation) for matched evaluator timings.

## Dependencies and independent surface solves

CPU examples need CPU Torch. GPU gradient/optimization examples need CUDA Torch; Example2 also needs SciPy.
`simsopt_async.py` additionally uses Simsopt, an independent magnetic-surface optimization library. `WORKERS` is the CPU process count.
`AlphaNuInitializer` prepares angular-coordinate seeds on GPU; `StandardSimsoptSolver` then solves on CPU.
The script prints acceptance, quality, rotational transform and timings, and saves seeds and accepted surfaces; see the [Simsopt guide](../docs/simsopt.md).
It does not run DESC equilibrium calculations. The earlier preview's full physical reports remain [historical validation](../docs/preview_validation.md).
