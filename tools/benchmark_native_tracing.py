"""Compare synchronous native trace endpoints and timings in isolated processes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--baseline', type=Path)
    result.add_argument('--candidate', type=Path)
    result.add_argument('--library', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--queries', default='1,8')
    result.add_argument('--lines', default='1,32,2304')
    result.add_argument('--threads', default='256')
    result.add_argument('--segments-per-coil', type=int, default=256)
    result.add_argument('--steps', type=int, default=240)
    result.add_argument('--samples', type=int, default=80)
    result.add_argument('--repeats', type=int, default=5)
    return result


def worker(args):
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from coil_eval import CoilSet
    from coil_eval._binding import BatchCoilFieldGpu, CoilFieldGpu

    args.output.mkdir(parents=True, exist_ok=True)
    coils = CoilSet.read_json(Path(__file__).resolve().parents[1]/'examples/coils.json')
    arrays = [coils.coeffs_x, coils.coeffs_y, coils.coeffs_z, coils.currents_a]
    records = []

    def measure(name, operation):
        operation()
        elapsed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = operation()
            elapsed.append(time.perf_counter()-start)
        np.savez(args.output/(name+'.npz'), **{f'a{i}': v for i, v in enumerate(outputs)})
        records.append(dict(case=name, wall_seconds=elapsed, wall_median=statistics.median(elapsed),
                            finite_counts=[int(np.isfinite(v).sum()) for v in outputs]))

    field = CoilFieldGpu(args.library, *arrays, coils.nfp, segments_per_coil=args.segments_per_coil)
    try:
        for threads in map(int, args.threads.split(',')):
            for lines in map(int, args.lines.split(',')):
                R, Z = np.linspace(.96, 1.04, lines), np.linspace(-.025, .025, lines)
                for precision in ('fp64', 'mixed64', 'fp32'):
                    measure(f'scalar_{precision}_t{threads}_l{lines}',
                            lambda: field.trace_period_blockline_precision(
                                R, Z, args.steps, precision=precision, threads_per_line=threads))
        measure('scalar_axis_samples', lambda: field.trace_axis_samples(
            1., 0., integration_steps=args.steps, sample_count=args.samples))
    finally:
        field.close()
    for queries in map(int, args.queries.split(',')):
        batch_arrays = [np.repeat(v[None], queries, axis=0) for v in arrays]
        # Independent geometries, with deterministic small current differences.
        batch_arrays[3] *= (1.+np.arange(queries)[:, None]*1e-4)
        field = BatchCoilFieldGpu(args.library, *batch_arrays, coils.nfp,
                                 segments_per_coil=args.segments_per_coil)
        try:
            for lines in map(int, args.lines.split(',')):
                R = np.repeat(np.linspace(.96, 1.04, lines)[None], queries, axis=0)
                Z = np.repeat(np.linspace(-.025, .025, lines)[None], queries, axis=0)
                measure(f'batch_mixed64_b{queries}_l{lines}',
                        lambda: field.trace_period(R, Z, steps=args.steps))
            measure(f'batch_axis_samples_b{queries}', lambda: field.trace_axis_samples(
                np.ones(queries), np.zeros(queries), integration_steps=args.steps,
                sample_count=args.samples))
        finally:
            field.close()
    result = dict(library=str(args.library.resolve()),
                  sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  segments_per_coil=args.segments_per_coil, steps=args.steps,
                  samples=args.samples, repeats=args.repeats, cases=records)
    (args.output/'timings.json').write_text(json.dumps(result, indent=2)+'\n')


def compare(args):
    import numpy as np

    if not args.baseline or not args.candidate:
        raise SystemExit('--baseline and --candidate are required')
    args.output.mkdir(parents=True, exist_ok=True)
    for label, library in [('baseline', args.baseline), ('candidate', args.candidate)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--library', str(library.resolve()),
                   '--output', str((args.output/label).resolve())]
        for name in ('queries', 'lines', 'threads', 'segments_per_coil', 'steps', 'samples', 'repeats'):
            command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
        subprocess.run(command, check=True)
    baseline = json.loads((args.output/'baseline/timings.json').read_text())
    candidate = json.loads((args.output/'candidate/timings.json').read_text())
    compared = []
    for old, new in zip(baseline['cases'], candidate['cases']):
        assert old['case'] == new['case']
        name, errors = old['case'], []
        with np.load(args.output/'baseline'/(name+'.npz')) as expected, np.load(args.output/'candidate'/(name+'.npz')) as actual:
            for key in expected.files:
                a, b = actual[key], expected[key]
                finite = np.isfinite(a) & np.isfinite(b)
                delta = a[finite]-b[finite]
                errors.append(dict(array=key, exact_equal=bool(np.array_equal(a, b, equal_nan=True)),
                                   finite_masks_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                                   max_abs=float(np.max(np.abs(delta), initial=0))))
        compared.append(dict(case=name, baseline_wall_s=old['wall_median'], candidate_wall_s=new['wall_median'],
                             speedup=old['wall_median']/new['wall_median'],
                             exact_equal=all(e['exact_equal'] for e in errors), errors=errors))
    result = dict(baseline={k:v for k,v in baseline.items() if k != 'cases'},
                  candidate={k:v for k,v in candidate.items() if k != 'cases'},
                  serial_reason='Matched isolated processes on the same GPU; concurrent kernels would bias timing.',
                  cases=compared)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.library:
        worker(arguments)
    else:
        compare(arguments)
