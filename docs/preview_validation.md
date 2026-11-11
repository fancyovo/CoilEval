# CoilEval 0.11.0 validation and limitations

**Scope.** This page records the 0.11.0 two-fit calibration and the original
optimization study. The [0.12.0 README](../README.md#validation) reports the
current timing comparison. The [example guide](../examples/README.md) gives
the runnable teaching scripts and their 0.12.0 results.

[简体中文](preview_validation.zh-CN.md)

CoilEval estimates magnetic surfaces and quasisymmetry from coils. Version 0.11.0 first fits a scalar surface label s in an initial region,
then refits it inside a selected closed region; its level sets approximate magnetic surfaces.
This report evaluates that **two-fit procedure** on fixed inputs and gives timings and two examples.
Simsopt is an independent magnetic-field and surface-optimization library; its saved reference solutions are reused for each comparison.
An optional third fit uses the entire interior slightly beyond the requested volume and is off by default.

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
The default two-fit procedure produces **2127** complete scores, **1944** matches to frozen Simsopt
references, and **817** pairs meeting the original reference-quality rules.
All large errors remain. The 1981 completed references are reused unchanged;
tasks without references contribute to success counts only.

![Four CoilEval–Simsopt comparisons, each n=1944, with empirical P90 error bands](assets/preview_s2_20261105/calibration_p90.png)

QS bands use `exp(±P90(abs(ln(prediction/reference))))`; iota bands use
`±P90(abs(prediction-reference))`. They describe empirical error coverage.
The three QS panels use factors 1.986, 1.368 and 1.157 for the 1944 paired samples. The table separately reports P90 relative error.

| Population | Pairs | f_B P90 relative error | f_C P90 relative error | f_T P90 relative error | ι P90 absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| All paired | 1944 | 93.86% | 34.67% | 15.20% | 0.01270 |
| Reference-qualified | 817 | 87.04% | 30.17% | 12.17% | 0.01015 |

Every entry below is **seconds per sample**: the median of three warm complete
calls after one cold call and one additional warmup, divided by batch size. Gradient mode includes one complete objective
backward pass. Batch amortization measures throughput; single-query rows give response latency.

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
[raw timing evidence](assets/preview_s2_20261105/timings.json). See the [validation record](preview_validation.md)
for cohort definitions, failures, reference qualifications and numerical limitations.

## Inputs, reference qualification and errors

Seed 2026110201 selected 1024 IDs uniformly without replacement from 371701 QUASR coil configurations, spanning coil counts, field periods and symmetry types.
Each configuration gives four inputs: multiply each geometric coefficient and current independently by `10**(sigma*z)`, with standard Gaussian z and sigma=0,0.003,0.01,0.03 in base-10 logarithmic units.
Target volume is half the saved outer-reference volume. The initial fitting radius is `3.5*sqrt(V_outer/(2*pi²*R_outer))`, where V_outer is that outer surface’s volume and R_outer its mean cylindrical radius, with lengths in metres. This noise differs from example 2’s ln(1.01) log perturbation.

All 4096 inputs remain. Reference solves produced 1981 completions, 181 rejections and 16 timeouts; 1918 inputs had no reference following earlier evaluation failures.
Version 0.11.0 gives 2127 complete scores, 1674 basic-evaluation failures, 240 failures constructing saved fitting records and 55 target-surface metric failures.
1944 scores have completed references for error comparisons; another 183 contribute only to availability.

The 817 reference-qualified pairs additionally satisfy predeclared rules, independent of CoilEval prediction error:

- The Simsopt solution passes basic finite-value, volume and angular-covering checks. On a dense grid, the relative L2 Boozer-equation residual and P95 of `abs(B·n)/abs(B)` are both ≤1e-5, with unit surface normal n; the sampled-section intersection check is resolved and detects no self-intersection.
- Sections are resolved and their maximum covariance-axis ratio is ≤5. The ratio is the square root of the largest-to-smallest eigenvalue ratio of the arc-length-weighted boundary-point covariance, measuring elongation.
- Target rotational transform at least 0.01 from rational numbers with denominator ≤8. Rational proximity can make angle equations sensitive; it does not itself prove islands.
- All three QS values change by no more than absolute 1e-10 plus relative 2% when angular grid size increases from 97 to 129.
- Relative magnetic-field L2 error ≤1e-3 when coil integration points increase from 512 to 1024. Relative L2 is the Euclidean difference norm divided by the reference norm.

Reference surfaces retain angular Fourier order 16; convergence with increasing surface order was not established.
Completed references are reused, with no input replacement or removal based on prediction error.

Both sides use the same metric formulas. Boozer coordinates make field lines straight and covariant angular magnetic-field components depend only on the surface.
f_B is the square root of the asymmetric Fourier-energy fraction; f_C/f_T use `|G|/B²`-weighted RMS over a thin region between neighboring equal-flux surfaces.
B is field strength and G the toroidal covariant component. The triple product `T=(∇ψ×∇|B|)·∇(B·∇|B|)` is made dimensionless as `R²*T/B⁴`, where ψ is toroidal flux divided by 2π and R the mean magnetic-axis major radius.
Each sample has explicit helicity. Scatter-plot ι comes from a new angle fit on the target surface; API `iota_boundary` comes from a volume angle fit. Keep these names distinct when comparing.

P90 is the linearly interpolated empirical 90th percentile. QS bands use `P90(abs(ln(pred/ref)))` for reciprocal-symmetric multiplicative factors;
the table uses `P90(abs(pred/ref-1))`. Iota uses absolute difference. These are empirical coverage bands, not confidence intervals.
Pooling four noise scales does not establish a separate zero-noise accuracy claim.

[All 4096 rows](assets/preview_s2_20261105/calibration_rows.json), [statistics and source identities](assets/preview_s2_20261105/accuracy.json) and the
[four-panel PDF](assets/preview_s2_20261105/calibration_p90.pdf) accompany this document.
The reference inventory SHA256 is `daff50895c551b855adbd92aa9b8e75399f2bc1dd8288c74a432d4d96eee3248`, identifying the unchanged reference files.

## Known causes of error and failure

This version includes angular-inversion and radial-root repairs. Angle inversion finds spatial positions for requested coordinates;
radial root finding locates scalar-label levels outward from the magnetic axis. The evaluator uses the first outward monotone interval connected to the axis (the first branch).
Its largest representable volume is the capacity. A fitted label that turns back radially can lose capacity without proving that the physical field lacks surfaces.

Smooth, nested Poincaré sections can still accompany large QS errors. Confirmed causes include full-volume fitting objectives and coefficient penalties favoring labels whose small geometric errors overwhelm an intrinsically small target QS;
unretained high-frequency angular content affecting retained Fourier modes; and amplification near resonances where the angle-equation divisor `m*ι−n*nfp` becomes small. Here m is the poloidal Fourier mode number and n the toroidal mode number per field period.
Regularization adds a coefficient penalty to least squares; reducing it does not always improve accuracy.
Other surfaces cannot be represented by a single intersection on each ray from the axis, or fail the global smooth-angle assumptions.
These cases need different diagnoses; appearance and fitting residual alone do not establish QS accuracy.

The optional third fit changes both the training interior and regularization. On the same 4096 inputs it produces 1821 scores, losing 314 default successes and recovering eight failures.
Domain and regularization changes have separate and interacting effects, so `psi_target_refit=False` remains the default.
Volume angle fitting is retained; the target surface separately fits angles from geometry and field, with poloidal and toroidal Fourier orders both 32.

## Timing scope

Eight fixed queries use four order-16 Fourier quasi-helical coil sets, IDs 1916221, 2485714, 1631665 and 1631494, at 0.25/0.5 of their stored reference volumes.
Tests cover single-query and batch 1/2/4/8 volume f_C, plus single-query surface f_T, each for values and values with gradients.
Every query fits independently. GPU execution retains batched axis preparation and CPU/GPU stage overlap.
Local field responses use FP32 (32-bit single precision) and scalar algebra uses FP64 (64-bit double precision).

Each test starts a fresh process, records one cold call, performs one additional warmup and measures three complete calls.
Gradient timing includes the full objective and one backward pass, including coil geometry conditions. Imports, construction, serialization and cleanup are outside the interval; GPU intervals synchronize at both ends.
CPU tests use Xeon Gold 6530 and PyTorch 2.7.1+cpu. GPUs use PyTorch 2.7.1+cu128; the RTX 5090 host is a Gold 6530 and the A100 host a Platinum 8358P.
Tables divide batch time by sample count; single-query rows describe response latency. This fixed workload does not describe the random calibration population’s timing distribution.

## Example 1: fixed coils

[example1.json](../examples/preview/example1.json) retains volume 0.3 m³ and fitting radius √0.06 m.
After a fresh CoilEval evaluation, angle coordinates initialize a Simsopt least-squares/Newton surface refinement, followed by a DESC fixed-boundary magnetohydrodynamic equilibrium solve.
DESC is an independent equilibrium library; this run uses order 12 and at most 300 iterations.

Volume f_C=0.0039104499 and volume-fit boundary ι=0.46571541; DESC boundary f_C=0.0072281188.
DESC uses its area-weighted mean absolute residual and normalization, whereas CoilEval volume f_C uses RMS; report them separately.
DESC reaches the objective-gradient stopping tolerance `gtol` after 63 iterations, with nested initial/final fitted surfaces.
Mean/P95/maximum normalized residual forces are 0.00438092/0.01121054/0.08626663; strict equilibrium certification remains false.
[Evaluation](assets/preview_s2_20261105/example1_native.json), [input/code identities](assets/preview_s2_20261105/example1_source_provenance.json) and
[complete stage record](assets/preview_s2_20261105/example1_result.json) are supplied.

## Example 2: perturb, screen and optimize

See the [example 2 guide](../examples/preview/README.md) for commands. Each geometric coefficient is independently multiplied by `exp(ln(1.01)*z)` with seed 2026110502; screening keeps currents fixed.
25 of 32 candidates score successfully; index 4 has the lowest complete objective, combining QS, rotational transform, gap, curvature, length, volume, normal drift and axis-radius costs.

BFGS updates an inverse-curvature approximation from gradients; Powell damping stabilizes updates and Armijo backtracking shortens trials until the objective decreases sufficiently.
Within the 400-iteration budget, 164 outer iterations accept 116 updates before consecutive rejected trials stop the run.
The best record is iteration 132: objective −3.69036229 → −5.48909363 and volume f_C 0.0178609147 → 0.0038968281.
The 1% perturbation completes without reduction; the stopping reason is not convergence certification.

Selected coils also complete the independent reference workflow. DESC reaches gtol after 110 iterations, with nested initial/final fitted surfaces and boundary f_C=0.0075078856.
Mean/P95/maximum normalized residual forces are 0.00561424/0.01405266/0.14672194; strict certification remains false.
A fresh full evaluation gives volume f_C=0.0038968455, differing relatively by 4.47e-6 from the optimization record; both records are retained.
[Optimization](assets/preview_s2_20261105/example2_optimization.json), [fresh evaluation](assets/preview_s2_20261105/example2_native.json) and
[full physical result](assets/preview_s2_20261105/example2_result.json) retain values and stopping reasons.

Each example independently traces field lines from 352 starts; intersections with four fixed toroidal sections form Poincaré plots.
Island-like contours occur near the selected boundary and outer traces disperse; individual periodic orbits were not certified.
Nested DESC fitted surfaces do not certify globally nested surfaces in the original coil field.

## Poincaré geometry checks

The earlier anomaly diagnostic used eight starts each at radial fractions 0.9, 1.0 and 1.1 near the target surface.
QS errors above a factor of two, or absolute ι errors above `max(0.01,0.02*abs(reference_iota))`, triggered review while remaining in the primary statistics.
Island-chain labels require separated closed lobes; elongation requires covariance-axis ratio >5. Insufficient coverage remains inconclusive.

The general illustrations below instead use 352 starts: 32 on the reference boundary plus 40 radial fractions from 0.05 to 2.0 on eight rays.
Double-precision traces cover 256 field periods with 256 Runge–Kutta integration steps per period and 512 segments per coil.
The four sections preserve equal physical R/Z scale, with cylindrical radius R and height Z in metres.
Black curves mark reference boundaries, crosses the axis, and colors seed radial fraction. The viewport extends 20% beyond the boundary; all raw traces are retained.

![Continuous nested section example](assets/preview_validation_20261103/poincare_dense_r0320-q0783386-s0.png)

All 352 traces complete for `r0320-q0783386-s0`, with continuous nested contours near the target, yet f_B is approximately 13.1 times its reference.
Controls identify fitting-objective and regularization bias as major causes; the point remains in both accuracy populations.
[Figure provenance](assets/preview_validation_20261103/poincare_dense_r0320-q0783386-s0.provenance.json).

![Nearby island-chain example](assets/preview_validation_20261103/poincare_dense_r0236-q1609885-s2.png)

`r0236-q1609885-s2` resolves separated lobes; 330 traces complete and 22 leave the diagnostic bounds.
The reference also has rational-proximity and residual-quality warnings, so islands alone do not explain all metric error.
[Figure provenance](assets/preview_validation_20261103/poincare_dense_r0236-q1609885-s2.provenance.json).

![Elongated-section example](assets/preview_validation_20261103/poincare_dense_r0402-q1861100-s0.png)

The most elongated section of `r0402-q1861100-s0` has covariance-axis ratio 5.907882, above threshold 5; all 352 traces complete.
Nested contours and elongated geometry can coexist. [Figure provenance](assets/preview_validation_20261103/poincare_dense_r0402-q1861100-s0.provenance.json).

Independent field comparisons at 32 points with matching coil discretization have relative L2 error below 6e-16.
Separate controls trace 24 starts for 32 periods while doubling integration resolution and coil segments.
For the island example, integration refinement gives a maximum common-finite point difference of 0.246551 m despite P95=4.38e-9 m, showing sensitivity of some outer trajectories.
Finite traces do not certify global topology. Linked provenance files identify tracing settings and original evidence locations.
