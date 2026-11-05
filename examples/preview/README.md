# Fixed QA preview examples

[简体中文](README.zh-CN.md)

`example1.json` contains the original four base coils (Fourier order4, nfp3,
stellarator symmetry), in the public CoilEval JSON schema. The physical array
SHA256 is `aebf705719b77ad064624362f5d90601dc4b8a2cfbaf3f8240b427faf8e1eaf7`.
Use QA helicity `(1,0)`, V=0.3m³ and fitting radius a=√0.06m. All examples use
the0.11.0 S2 defaults. Full reference results and their limitations are in the
[validation record](../../docs/preview_validation.md).

Example2 perturbs each of the108 Fourier coefficients independently as
`c*exp(log1p(fraction)*z)`. The initial fraction is0.01, seed2026110502;
all32 candidates use the same original example1 geometry. Signs and exact
zeros are preserved, and screening currents remain unchanged. Optimization
can vary all112 physical parameters, including currents, as in the original
BFGS protocol. The bundled dataset statistics supply its fixed normalization.

From the extracted source archive, with the GPU extra and CUDA PyTorch installed:

```sh
python examples/preview/perturb_optimize.py screen --output runs/example2-screen
python examples/preview/perturb_optimize.py optimize --screens runs/example2-screen/screen.json --output runs/example2-optimize
```

An optional `--library /path/to/libcoil_eval_native.so` selects a custom binary.
For multiple allocated GPUs, independently run `screen --shards 4 --shard I`
for I=0,1,2,3, assigning one GPU and a distinct output directory to each process.
Pass all four `screen.json` files to `optimize --screens`; all32 distinct indices
are required before selecting a start. The optimizer is one dependent chain.

Selection minimizes the complete objective among valid candidates: log volume
f_C, five condition penalties, normal-drift quality and the live magnetic-axis
radius cost. Targets are `[0.47,0.1,6.0,3.5,0.3]` for iota, gap(m), maximum
curvature(1/m), mean length(m) and volume(m³). K is not the objective.
One analytic reverse supplies the full gradient; Powell-damped dense inverse
BFGS uses formal value-only Armijo backtracking. The default400 outer steps
and consecutive-rejection limit bound the run; neither certifies convergence.

`screen.json` retains every candidate status. `evaluations/`, `history.json`,
`result.json`, `start.npz` and `selected.npz` preserve trials and the selected
minimum-full-objective geometry. Outputs are created in new directories to
protect earlier attempts. Source and reference failures remain explicit.

If a physical attempt cannot complete, halve `--fraction` (0.005,0.0025,…),
repeat exactly32 candidates, and retain previous outputs. The seed and Gaussian
draws remain identical; the script keeps the target volume and fitting radius
fixed. A schema/environment failure must be corrected before judging noise.

The release validation also runs selected geometries through source psi→alpha→nu→
standard Simsopt LS/Newton→diagnostics→DESC. Native optimization success and
independent physical/reference acceptance are reported separately.
