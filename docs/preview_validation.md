# CoilEval 0.11.0 S2 preview validation

[简体中文](preview_validation.zh-CN.md)

The preview uses repaired S2 by default; S3 is optional. Calibration reuses the
frozen4096 inputs and completed Simsopt references. Its measured numerical
recipe is identical to the new public defaults, checked on CPU and GPU.
The early0.9.0 campaign's1336-pair results and ring-only example2 are historical;
they are superseded here by repaired-S2 calibration and perturb32 initialization.

## S2 accuracy and speed

The fixed cohort contains **4096** tasks: 1024 QUASR inputs at four noise scales.
Repaired S2 produces **2127** complete scores, **1944** matches to frozen Simsopt
references, and **817** pairs meeting the original reference-quality rules.
All large errors remain. The 1981 completed references are reused unchanged;
tasks without references contribute to success counts only.

![CoilEval S2 versus Simsopt QS and iota, with measured P90 bands for each population](assets/preview_s2_20261105/calibration_p90.png)

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
[raw timing evidence](assets/preview_s2_20261105/timings.json). See the [validation record](preview_validation.md)
for cohort definitions, failures, reference qualifications and numerical limitations.

## Cohort, qualification and error definitions

Seed2026110201 selected1024 IDs uniformly without replacement from371701 QUASR
entries, across coil counts, field periods and QS types. Four noise scales are
0,0.003,0.01,0.03 decades: each coefficient and current is multiplied by
`10**(sigma*z)`. Target volume is half the stored outer-reference volume;
source-radius multiplier is3.5. This calibration noise differs from example2.

The fixed reference ledger has1981 completed solves,181 rejections,16 timeouts
and1918 unavailable references following original native failures. Repaired S2
has2127 complete scores,1674 native failures,240 source-capture failures and55
surface-metric failures. All4096 remain in the denominator. Of1981 completed
references,1944 pair with S2;183 scores have no completed reference. Those183
measure recovered availability, not accuracy. No failed input was replaced.

Reference qualification requires strict Simsopt residual quality, resolved
sections with maximum covariance-axis ratio≤5, target iota≥0.01 from rationals
with denominator≤8,97→129 surface-grid checks of all three QS metrics
(absolute1e-10 plus2% relative), and512→1024 coil-quadrature vector-field L2≤1e-3.
Surface order16 stays fixed; convergence in surface order was not established.
Rational proximity alone does not establish islands. The817 qualified pairs
retain every error; selection uses the original reference rules, not score error.

Both surfaces use the shared comparison kernel: f_B is the RMS asymmetric
Boozer Fourier-energy fraction; f_C and f_T use `|G|/B²` flux-shell RMS weights.
The explicit sample helicity and the `R²*T/B⁴` length normalization match.
This compares reconstructed surfaces and fields; formula implementation is shared.
Iota is the geometric target-surface alpha solve, distinct from native volume-alpha
`iota_boundary`. All points, including outliers, are in the scatter data.

The P90 bands use linear-interpolated empirical quantiles. QS multiplicative
bands treat reciprocal ratios symmetrically; relative-error P90 is computed
separately as `P90(abs(pred/ref-1))`. Iota uses absolute signed-value differences.
These are descriptive error coverage, not confidence intervals or predictions
for every new magnetic configuration. Pooled statistics include the four noise
strata; they do not establish a separate zero-noise accuracy claim.

[All4096 rows](assets/preview_s2_20261105/calibration_rows.json),
[statistics and source hashes](assets/preview_s2_20261105/accuracy.json), and
[PDF figure](assets/preview_s2_20261105/calibration_p90.pdf) accompany the report.
The frozen reference manifest SHA256 is
`daff50895c551b855adbd92aa9b8e75399f2bc1dd8288c74a432d4d96eee3248`.

## Numerical findings and remaining limitations

Angle inversion and first-radial-branch repairs remove implementation failures.
Large errors can remain on apparently nested, smooth sections: full-volume scalar
transport fitting and column-scaled ridge may distort a very low-QS target;
the angular transport equation's unresolved spectral tail can couple into retained
modes; small denominators amplify source-flow errors near resonances. Some surfaces
also exceed the first-root star-shaped chart or smooth angular-coordinate assumptions.
These mechanisms require different remedies. A small fitting residual or a smooth
Poincaré plot alone does not quantify target QS accuracy.

S3 changes the training interior and its ridge. It improves many difficult targets,
but fixed4096 testing gives1821 scores versus2127 for S2:314 losses and8 recoveries.
Shrinking the domain and weakening ridge can induce radial scalar folds; the
two changes have separate and interacting effects. S3 remains opt-in, with
`psi_target_refit=False` in this release. The preview retains volume alpha and
geometric target-surface alpha32; it adds no high-order or reference-assisted fallback.

## Timing contract

The36 cells cover CPU8/RTX5090/A100, scalar and batch1/2/4/8 volume f_C,
and scalar target-surface f_T, each in value and value-plus-gradient modes.
The fixed eight-query workload uses original order16 QH coil IDs1916221,2485714,
1631665,1631494 at0.25 and0.5 of stored reference volume. Source-fit sharing is
disabled; each query remains independent. GPU batch-axis/pipeline controls remain
enabled. Fields use the recorded FP32 GPU local response; psi algebra uses FP64.

Each cell has a fresh process, a cold call, one warmup, and three measured calls.
The synchronized interval includes complete public forward work and, in gradient
mode, a scalar objective and one full reverse including engineering terms. Imports,
allocation, constructors, serialization and cleanup are outside. Raw whole-call
times remain available; every displayed batch table uses median/batch_size.
CPU uses eight logical Xeon Gold6530 threads and PyTorch2.7.1+cpu. GPU environments
use PyTorch2.7.1+cu128; A100 has a Xeon Platinum8358P host, RTX5090 a Gold6530 host.
Latency belongs to this workload, not the random calibration population.

## Example1: fixed QA geometry through the complete reference workflow

The original `n3-QA-i0-l5-s2-v5` coils, V=0.3m³ and a=√0.06m are unchanged.
Fresh S2 source capture followed source→alpha→nu→standard Simsopt LS/Newton→
diagnostics→DESC12, in one job. Native volume f_C=0.0039104499 and volume-alpha
boundary iota=0.46571541. DESC boundary f_C=0.0072281188 uses its official
surface-weighted mean absolute residual and its own normalization.
The two f_C values describe different measures and domains.

DESC's optimizer finishes in63 iterations at gtol; initial/final surfaces are
nested. Mean/P95/maximum normalized residual force are0.00438092/0.01121054/0.08626663.
Equilibrium certification remains false. A completed pipeline does not remove
these residual-force or Simsopt quality limitations.

[Fresh native record](assets/preview_s2_20261105/example1_native.json),
[source provenance](assets/preview_s2_20261105/example1_source_provenance.json),
and [complete stage/result record](assets/preview_s2_20261105/example1_result.json)
are provided. The selected geometry is the original example1, reevaluated by S2;
its native values and objective have been refreshed consistently.

## Example2: perturb32 initialization and BFGS

Each Fourier coefficient receives `c*exp(log1p(0.01)*z)`, independent standard
Gaussian z, with seed2026110502. Screening currents, target volume and fitting
radius remain fixed. Exactly32 candidates are scored; invalid entries remain.
At1% noise,25/32 score successfully; candidate index4 minimizes the complete
objective, initially−3.69036229. The objective includes log volume-QS, iota,
gap, curvature, length, volume, normal-drift quality and live-axis-radius costs.
Analytic gradients feed the existing dense inverse BFGS/Powell/Armijo policy.
The400-outer-iteration budget and consecutive-rejection stopping do not assert
convergence. The run accepted 116 updates over 164 outer iterations and stopped
at the consecutive-rejection limit. The selected minimum is iteration 132:
complete objective −5.48909363 and native volume f_C=0.0038968281, from 0.0178609147.
The 1% amplitude completed without reduction.

The final geometry completed the full source→alpha→nu→Simsopt→DESC workflow.
DESC reached gtol in 110 iterations, with nested initial/final surfaces and
boundary f_C=0.0075078856. Mean/P95/maximum normalized residual forces are
0.00561424/0.01405266/0.14672194; equilibrium certification remains false.
The native and DESC f_C definitions differ as in example1.
[Optimization record](assets/preview_s2_20261105/example2_optimization.json),
[native record](assets/preview_s2_20261105/example2_native.json) and
[full physical result](assets/preview_s2_20261105/example2_result.json) retain
the values, stopping reason, identities and stage times.

Independent 352-seed four-cut plots of both examples show island-like contours
near the selected boundary and dispersing outer traces. Individual periodic
orbits were not certified. DESC nestedness describes its fitted equilibrium
coordinates; it does not certify global nesting of the original coil field.

The [runnable example](../examples/preview/README.md) preserves all32 outcomes,
accepted and rejected trials, selected coefficients and the exact settings.
If an amplitude cannot complete the example, retry a halved amplitude with the
same Gaussian draws and retain the failed run; do not change volume or redraw seeds.

## Poincaré geometry checks

The frozen anomaly review traces eight poloidal starts at each of radial
fractions 0.9, 1.0 and 1.1. This is a **target-neighborhood diagnostic**. Its
review queue contains QS discrepancies above a factor of 2 or iota absolute
errors above `max(0.01, 0.02*abs(reference_iota))`. These triggers request
inspection; every such point remains in the primary accuracy populations.
A nearby island-chain label requires resolved separated closed lobes, and an
elongated-surface label requires the declared covariance axis ratio above 5.
Incomplete coverage and unresolved outer morphology remain inconclusive.

The following general illustrations use separate dense traces from the
interior through the selected surface to the outer region. Each uses 352
starts: 32 on the reference boundary and 40 radial fractions from 0.05 to 2.0
on eight rays. The FP64 coil-field tracer follows 256 field periods with
256 RK steps per period and 512 coil segments. Four geometric cuts use equal
physical R/Z scale and a viewport extending 20% beyond the reference boundary
span on each side. Black curves mark the selected reference surface; crosses
mark the stored source initializer axis. Turbo colors encode seed radius,
saturating above 1.4. All finite raw traces are retained; the viewport only
limits the displayed region.

![Dense four-cut Poincaré plot for r0320-q0783386-s0, with continuous contours near the selected reference boundary](assets/preview_validation_20261103/poincare_dense_r0320-q0783386-s0.png)

`r0320-q0783386-s0` has continuous contours around the selected boundary,
while CoilEval f_B is about 13.1 times the matched Simsopt reference. Subsequent fixed-domain/ridge controls identify scalar-fit objective and
regularization bias as major causes; the sample remains in both accuracy populations. All 352 dense traces complete the requested interval.
[Figure provenance](assets/preview_validation_20261103/poincare_dense_r0320-q0783386-s0.provenance.json).

![Dense four-cut Poincaré plot for r0236-q1609885-s2, resolving multiple island lobes near the selected reference boundary](assets/preview_validation_20261103/poincare_dense_r0236-q1609885-s2.png)

`r0236-q1609885-s2` resolves multiple island lobes near the selected boundary.
Of 352 dense traces, 330 complete and 22 leave the diagnostic bounds. Its
reference also carries rational-proximity and residual-quality flags. Nearby
islands do not uniquely attribute the metric discrepancy.
[Figure provenance](assets/preview_validation_20261103/poincare_dense_r0236-q1609885-s2.provenance.json).

![Dense four-cut Poincaré plot for r0402-q1861100-s0, including an elongated section at geometric phi 0.6283 radians](assets/preview_validation_20261103/poincare_dense_r0402-q1861100-s0.png)

`r0402-q1861100-s0` has a boundary covariance axis ratio of 5.907882 at its
most elongated cut, above the declared threshold 5. All 352 dense traces
complete. The case illustrates a shape limitation even though nested contours
are visible in this finite diagnostic.
[Figure provenance](assets/preview_validation_20261103/poincare_dense_r0402-q1861100-s0.provenance.json).

The dense protocol is `coileval-dense-full-style-poincare-20261103-v1`.
Its same-quadrature Simsopt field comparisons have relative vector L2 below
6e-16 at 32 test points. Controls trace 24 starts for 32 periods while doubling
RK resolution and coil segments. In the island case, RK refinement has a
maximum common-finite point difference of 0.246551 m despite a p95 of
4.38e-9 m; individual outer trajectories are sensitive to integration
resolution. These finite diagnostics do not certify global nested topology.
The frozen references are unchanged. Raw traces, controls and provenance
remain in the parent workspace's
`calibration/dense_execution_20261103/` manifest and registered remote root.
The static figures here are the requested documentation exception to the
ordinary raw-data/on-demand report policy.
