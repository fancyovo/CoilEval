# Preview validation benchmark

This document describes the reproducible preview benchmark used by the
2026-11-02 validation branch. It measures the public CoilEval API at version
0.9.0 (source version, unreleased) with one frozen geometry cohort and one
explicit `EvaluationConfig` per query. The benchmark does not change library
defaults or numerical algorithms.

The matrix has CPU8, RTX 5090, and A100 allocations. Each allocation uses
eight assigned CPU cores. It records a scalar call and public batches B1, B2,
B4, and B8, for pure value and complete value plus one engineering-inclusive
scalar loss gradient. The gradient includes the QS volume, boundary iota,
normal-drift quality, coil gap, curvature, length, and surface-volume terms;
optional surface-QS scores are separate metrics. The same geometry is not
silently shared between queries.

Every result is a fresh process. The first synchronized public call is labeled
`cold`, one or more untimed calls are labeled `warmup`, and three synchronized
calls are labeled `measured`. The timed wall interval includes public forward
work, result completion, the objective and one reverse for gradient mode. Input
loading, imports, constructor setup, allocation, serialization and cleanup are
outside the interval. Batch wall time and wall time divided by batch size are
reported separately; the latter is amortized cost, not scalar latency.

Each JSON record contains source, native library, field library, cohort,
physical-array and configuration hashes, host/device metadata, query status,
cold/warm/measured attempts, and errors. Missing or failed cells remain
explicit in the reducer. No timing is inferred from private stage timers.

The runner requires a Slurm allocation and rejects a device whose reported
GPU name does not match the requested RTX 5090 or A100 label. CPU records set
`CUDA_VISIBLE_DEVICES` empty and record `NUMPY_MADVISE_HUGEPAGE=0`, the
process-only setting needed to avoid the documented first-touch confounder.
