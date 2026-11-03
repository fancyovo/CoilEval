# Preview validation benchmark

The 2026-11-03 benchmark measures the public CoilEval 0.9.0 API (unreleased
source version) on a fixed eight-query workload. It uses the published
two-stage configuration and preserves all 16 Fourier orders of four original
QUASR QH coil sets: IDs 1916221, 2485714, 1631665 and 1631494, each with
`nfp=4`, four base coils and 400 physical parameters. Targets are 0.25 and
0.5 of each stored reference volume, with one explicit `EvaluationConfig`
per query. Batch sizes use prefixes of this ordered workload. These are
latencies for the established throughput workload; the random QUASR
calibration experiment measures population accuracy separately.

The matrix has CPU8, RTX 5090, and A100 allocations. Each allocation uses
eight assigned CPU cores. It records a scalar call and public batches B1, B2,
B4, and B8, for pure value and complete value plus one engineering-inclusive
scalar loss gradient. The gradient includes the QS volume, boundary iota,
normal-drift quality, coil gap, curvature, length, and surface-volume terms;
optional surface-QS scores are separate metrics. Source-fit sharing is disabled,
including repeated coils at two target volumes. GPU batch-axis preparation and
the public pipeline remain enabled.

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
explicit in the reducer. Authoritative latency comes from synchronized wall
time around the complete public call.

The runner requires a Slurm allocation and rejects a device whose reported
GPU name does not match the requested RTX 5090 or A100 label. CPU records set
`CUDA_VISIBLE_DEVICES` empty and record `NUMPY_MADVISE_HUGEPAGE=0`, the
process-only setting needed to avoid the documented first-touch confounder.

The CPU allocation has eight logical CPUs from an Intel Xeon Gold 6530 and
uses eight threads with PyTorch 2.7.1+cpu. RTX 5090 uses the same CPU model;
A100-SXM4-80GB uses an Intel Xeon Platinum 8358P host. Both GPU environments
use PyTorch 2.7.1+cu128. The GPU local response uses FP32 field operations and
FP64 psi operations, as recorded by the 0.9.0 API. The timing workload requests
the ordinary volume `f_C` metrics and no optional surface `f_B` or `f_T`.

Each hardware receives one isolated allocation. Its workload cells run in
fresh processes sequentially to avoid competition on the same allocated
device; the three hardware allocations run concurrently. Timing samples
describe these allocations and workloads; cross-hardware ratios also include
their recorded host differences.

## Measured latency

GPU results completed on 2026-11-03: all 20 cells have three valid measured
calls after one cold call and one warmup. CPU measurements completed in retry job 128043.
Values below are medians in seconds for the entire call. A batch contains the
number of independent queries shown in its row.

<!-- BENCHMARK_WALL_TABLE -->

| API workload | CPU8 value | CPU8 value + gradient | RTX 5090 value | RTX 5090 value + gradient | A100 value | A100 value + gradient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar | 22.483 | 42.531 | 1.553 | 1.959 | 1.903 | 2.473 |
| Batch 1 | 22.865 | 41.871 | 1.560 | 1.989 | 1.898 | 2.452 |
| Batch 2 | 63.821 | 99.577 | 2.584 | 2.813 | 3.219 | 3.650 |
| Batch 4 | 121.086 | 192.466 | 4.665 | 5.029 | 5.902 | 6.620 |
| Batch 8 | 245.300 | 385.934 | 9.013 | 9.420 | 11.282 | 12.657 |

<!-- /BENCHMARK_WALL_TABLE -->

For batch 8, the amortized value/value-plus-gradient costs are 1.127/1.178
seconds per query on RTX 5090 and 1.410/1.582 seconds per query on A100.
The cold call and constructors have separate fields in the raw records;
the table above contains warm public-call latency.

The protocol is `coileval-preview-public-api-benchmark-20261102-v1`.
The parent validation workspace retains the hash-pinned plan, cohort receipt,
raw JSON records, reducer output and audit under
`evaluation/coileval_preview_20261102/benchmark/`. CPU's first attempt used
an environment missing `threadpoolctl` and stopped before a timed physical
call; the retry uses the existing CPU environment and retains the failed
attempt as evidence.
