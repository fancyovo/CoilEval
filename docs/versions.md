# Versions

[简体中文](versions.zh-CN.md)

| Version | Added capability | Selection |
| --- | --- | --- |
| 0.0.0 | Independent evaluation and local finite-difference gradients | `Evaluator` |
| 0.1.0 | Faster integration settings | `EvaluationConfig(fast=True, ...)` |
| 0.2.0 | Stokes boundary flux in local evaluations | `flux_method='stokes'` with fast mode |
| 0.3.0a1 | PyTorch reverse gradients and independent-geometry batches | `coil_eval.torch.TensorEvaluator` |

The default NumPy behavior stays compatible across these versions. New numerical
paths are selected explicitly. Use an exact version and save the configuration
alongside results when repeating an evaluation.

To build an earlier release from a clean checkout:

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

Keep each native build paired with the Python version that built it. Release
tags identify complete source trees, including their documentation and examples.
