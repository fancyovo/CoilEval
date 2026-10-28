# Surface quasisymmetry metrics

[简体中文](qs.zh-CN.md)

CoilEval 0.6 adds Fourier (`f_B`) and optional triple-product (`f_T`)
quasisymmetry metrics on the target-volume surface. They share a differentiable
alpha+nu coordinate construction, with a target-surface alpha fit and vacuum
potential-based nu correction. The coil evaluation interface uses PyTorch on
GPU or CPU. Simsopt is used separately for reference surfaces.

## Evaluate a coil set

```python
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.qs import SurfaceQSConfig, SurfaceQSEvaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes", psi_fit="two_stage",
    psi_refit_sampling="polar_quadrature",
    alpha_radial_weighting="smooth", alpha_radial_basis="flux_legendre",
)
with SurfaceQSEvaluator(config=config, surface_qs=SurfaceQSConfig()) as evaluator:
    result = evaluator.evaluate(coils)
    print(result.f_B)
    print(result.diagnostics)
```

The value interface accepts a `CoilSet` and returns Python scalar metrics with
diagnostics. Its field calculations use float64. `library`, `field_library`
and `device_id` select a custom backend or GPU, as in the tensor interface.
The default device is `'cuda'`. Select `device='cpu'` with optional `threads`
and `axis_grid` settings for CPU execution; see the [CPU guide](cpu.md).
The coil interface supports four basic coils with `nfp=4`, helicity `(1,4)`,
256 coil segments and four psi/alpha local fit iterations. Set `fast=True`
and `flux_method='stokes'`, following the [tensor parameter conventions](torch.md).
`f_T` is `None` by default. Select `SurfaceQSConfig(compute_f_T=True)` to compute
both metrics using the analytic spatial field Jacobian.

Use `evaluate_many(coil_sets, configs=None)` for several requests. It returns a
tuple in input order and accepts one `EvaluationConfig` per coil set. For example,
evaluate two volumes of the same geometry with:

```python
from dataclasses import replace

configs = [replace(config, target_volume=v) for v in (0.025, 0.05)]
with SurfaceQSEvaluator(config=config) as evaluator:
    results = evaluator.evaluate_many([coils, coils], configs=configs)
    print([result.f_B for result in results])
```

On GPU, axis preparation is batched across compatible inputs. Identical two-stage
geometries with settings differing only in volume share their source fit; each
target surface is then sampled in turn. Constructor options `batch_axis=False`
and `share_geometry=False` disable those respective optimizations. Each result's
`diagnostics["batch"]` records its group, execution strategy and amortized wall
time. Invalid source captures raise `InvalidEvaluationError`, as in scalar use.
CPU batches capture each query independently and retain the same input order.

Run the example with:

```sh
python examples/qs_metrics.py
python examples/qs_metrics.py --accuracy balanced
python examples/qs_metrics.py --gradient --metric f_B
python examples/qs_metrics.py --gradient --metric f_T
```

## Differentiate an objective

```python
import torch
from coil_eval.qs import SurfaceQSConfig
from coil_eval.torch import TensorEvaluator, pack_coils

parameters = pack_coils(coils, device="cuda:0")
with TensorEvaluator(
    config=config, surface_qs=SurfaceQSConfig(), field_dtype=torch.float64,
) as evaluator:
    result = evaluator.evaluate(parameters)
    objective = result.surface_qs.f_B.square()
    gradient, = torch.autograd.grad(objective, result.parameters)
```

Set `compute_f_T=True` and choose `result.surface_qs.f_T` to differentiate the triple-product objective.
The gradients use the physical parameter layout in `result.parameters`.
`evaluate_batch` returns one value per input for each enabled metric; differentiating
the sum of independent objectives gives the corresponding per-input gradients.

`surface_qs=None` is the `TensorEvaluator` default. Existing evaluation calls
retain their usual outputs and computational path. The optional metrics appear
in `result.surface_qs`; the existing `result.metrics.qs_volume` and `qs_outer`
continue to describe the differential residual over their respective volume
and outer-bin regions.

The local reverse derivative includes the moving magnetic axis, psi fits,
target level, alpha, iota, nu, sampling positions and field values. Coordinate
inversions require a positive Jacobian. Inspect individual metric gradients
and geometry/current blocks when comparing against finite differences.
The surface fit's rotational transform is reported as
`diagnostics["surfaces"][i]["iota_surface"]`. It is separate from the existing
score's `result.metrics.iota_boundary`, which retains its original fit.

## Resolution and diagnostics

| `SurfaceQSConfig` option | Default | Meaning |
| --- | --- | --- |
| `nphi` | `129` | Points across one toroidal field period |
| `ntheta` | `131` | Points across one full poloidal turn |
| `nu_order` | `24` | Maximum retained poloidal and per-period toroidal nu mode |
| `compute_f_T` | `False` | Also evaluate the field-derivative triple product |
| `alpha_order` | `32` | Target-surface alpha mode limit; `0` retains the original alpha fit |
| `alpha_ridge` | `1e-12` | Regularization for the target-surface alpha fit |
| `alpha_coordinates` | `'geometric'` | Fit in geometric angles or the existing `'straight'` angle map |
| `alpha_fit_nphi`, `alpha_fit_ntheta` | `None` | Optional separate alpha-fit grid sizes; otherwise chosen from the output grid and order |
| `nu_method` | `'potential'` | Vacuum potential fit; `'transport'` retains the transport formulation |
| `max_iterations` | `30` | Maximum iterations for each angular inversion |
| `tolerance` | `1e-11` | Angular inversion tolerance, radians |
| `min_jacobian` | `1e-6` | Minimum permitted angular-map Jacobian |
| `resonance_tolerance` | `1e-10` | Minimum retained absolute nu transport-mode divisor |

Both grid sizes are odd and at least five, and each must exceed twice
`nu_order`. Odd grids give unambiguous derivatives for every retained Fourier
mode. All grids omit the repeated periodic endpoint.

`SurfaceQSConfig.preset(name, **overrides)` provides these resolution settings:

| Preset | `nphi × ntheta` | `alpha_order` | `nu_order` | Alpha coordinates |
| --- | --- | --- | --- | --- |
| `'high'` (default) | `129 × 131` | `32` | `24` | geometric |
| `'balanced'` | `97 × 99` | `24` | `16` | geometric |
| `'fast'` | `65 × 67` | `8` | `12` | straight |

These are resolution choices; convergence depends on the particular field.
Increase the output grid, alpha order and nu order separately to distinguish
sampling error from coordinate representation. When alpha-fit grids are not
specified, they use at least `4*alpha_order+1` toroidal and
`4*alpha_order+3` poloidal points, and at least the output grid resolution.
The optional `f_T` uses the analytic field Jacobian and a first angular
derivative of `B dot grad(B)`; inspect its grid convergence separately.

For tensor results, `result.surface_qs.diagnostics["surfaces"]` contains one record per input;
value results expose the same records through `result.diagnostics["surfaces"]`.
Records include angular inversion errors and minimum Jacobians, alpha and nu fit
residuals, magnetic tangency, separate covariant and Boozer-equation residuals, the normalization
radius, and `spectral_tail_fraction`. The latter is the energy fraction in the
outer third of either resolved frequency range. It helps identify a need for
finer grids; a small fraction alone does not establish convergence.
With a target-surface alpha fit, `surface_alpha_solver` records the linear-solve
method, relative residual, iteration count and any FP64 fallback. Tensor
gradient calls append their corresponding reverse-solve records.

## Metric definitions

Angles below use radians and the Fourier series is

\[
B(\theta_B,\phi_B)=\sum_{m,n}B_{mn}e^{i(m\theta_B-n\phi_B)}.
\]

For helicity `(M,N)`, `N` is the full-torus mode number. The dimensionless
Fourier metric is

\[
f_B=\sqrt{\frac{\sum_{mN-nM\ne0}|B_{mn}|^2}
                       {\sum_{m,n}|B_{mn}|^2}}.
\]

The denominator uses uniform Boozer-angle Fourier energy. For the triple
product, let \(\psi\) be toroidal flux divided by \(2\pi\),
\(B=|\mathbf B|\), and

\[
T=(\nabla\psi\times\nabla B)\cdot\nabla(\mathbf B\cdot\nabla B).
\]

\[
f_T=\sqrt{\left\langle\left(\frac{R^2T}{B^4}\right)^2\right\rangle_\psi}.
\]

The flux-shell average uses weights
\(|\mathcal J|=|G+\iota I|/B^2\). Coil evaluation uses the vacuum convention
`I=0`; `R` is the magnetic-axis mean cylindrical radius. Use this same radius
when comparing a Simsopt reference. These definitions specify the scalar RMS
normalization; other software can use different averages or normalizations.

## Evaluate supplied Boozer arrays

`boozer_spectrum_metrics` in `coil_eval.qs_metrics` computes
`f_B`, `f_B_squared`, `spectrum` and `spectral_tail_fraction` with one FFT.
It needs only field magnitudes on the uniform angular grid.

```python
from coil_eval.qs_metrics import boozer_spectrum_metrics

metrics = boozer_spectrum_metrics(B, nfp=4, helicity=(1, 4))
```

`coil_eval.qs_metrics.boozer_qs_metrics(B, nfp=..., helicity=..., iota=...,
G=..., major_radius=..., I=0.0)` accepts NumPy arrays or PyTorch tensors with
shape `(..., nphi, ntheta)`. Its angular grids are uniform Boozer angles, with
one field period in phi and one poloidal turn in theta. `G` and `I` use radian
angles and units of tesla metres. Surface parameters broadcast over leading
dimensions.

The returned dictionary contains `f_B`, `f_T`, their squares, the point arrays
`f_T_raw` and `f_T_normalized`, `jacobian`, the complex Fourier `spectrum`, and
`spectral_tail_fraction`. `major_radius` is required explicitly. NumPy use of
this helper requires neither PyTorch nor the native GPU backend. Tensor inputs
retain their derivatives.

`boozer_qs_metrics_field(B_vector, dBdx, x_theta, x_phi, ...)` instead computes
the triple product from Cartesian field values/Jacobians and Boozer tangents.
The vectors have shape `(..., nphi, ntheta, 3)` and the Jacobian has an additional
final dimension of size 3, with `dBdx[..., i, j] = d B_i / d x_j`.
Tangents differentiate radian Boozer angles. This is the field-derivative path
used by the coil interface when `compute_f_T=True`.

## Compare with Simsopt

After obtaining an accepted `StandardSurface` with the
[Simsopt workflow](simsopt.md), evaluate it with:

```python
from coil_eval.simsopt.qs_reference import surface_qs_metrics

reference = surface_qs_metrics(
    surface, field, major_radius=axis_mean_radius,
    helicity=(1, coils.nfp), sizes=(65, 97, 129),
)
print(reference["f_B"], reference["f_T"])
print(reference["quality"], reference["convergence"])
```

The reference computes the triple product using analytic Cartesian field
derivatives and angular differentiation of `B dot grad(B)`. It also rechecks
self-intersections and reports fixed-surface grid convergence. Confirmed
self-intersections are rejected; inconclusive checks retain a quality warning.

Separate a same-surface formula comparison from a comparison between the native
target surface and the solved reference surface. The latter also measures psi
and coordinate-fit differences. Magnetic islands and strongly distorted
surfaces can prevent a reliable nested-surface representation; preserve their
failure or quality diagnostics when assessing agreement. Grid convergence on
one fitted surface and long-term field-line confinement answer different
questions.
