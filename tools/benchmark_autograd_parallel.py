"""Compare native autograd libraries in isolated processes on one idle GPU.

The output directory must be inside an owned project/run directory. Example:
  python tools/benchmark_autograd_parallel.py --baseline OLD.so --candidate NEW.so \
      --output ../runs/autograd_parallel --queries 1,8 --repeats 5
"""
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
    result.add_argument('--points', default='1,128,1024')
    result.add_argument('--lines', default='1,32')
    result.add_argument('--segments', type=int, default=8192)
    result.add_argument('--steps', type=int, default=100)
    result.add_argument('--repeats', type=int, default=5)
    result.add_argument('--dtype', choices=('f32', 'f64'), default='f32')
    return result


def worker(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from coil_eval._autograd.batch_field import BatchSegmentField
    from coil_eval._autograd.field import NativeFieldLibrary

    args.output.mkdir(parents=True, exist_ok=True)
    library = NativeFieldLibrary(args.library)
    dtype = {'f32': torch.float32, 'f64': torch.float64}[args.dtype]
    records = []
    for batch in map(int, args.queries.split(',')):
        generator = torch.Generator(device='cuda').manual_seed(5741 + batch)
        p = (torch.randn((batch, args.segments, 3), device='cuda', dtype=dtype,
                         generator=generator)*.1).requires_grad_()
        w = torch.randn(p.shape, device='cuda', dtype=dtype, generator=generator)*1e5
        w[..., 2] += 8e5
        w.requires_grad_()
        field = BatchSegmentField(p, w, library)
        cases = []
        for points in map(int, args.points.split(',')):
            q = (torch.randn((batch, points, 3), device='cuda', dtype=dtype,
                             generator=generator)*.2 + 1.3).requires_grad_()
            for mode in ('field_gradient', 'field_vjp', 'potential_vjp'):
                def operation(mode=mode, q=q):
                    if mode == 'potential_vjp':
                        value = (field.potential(q),)
                    else:
                        value = field(q, gradient=True)
                    if mode.endswith('_vjp'):
                        gradient = torch.autograd.grad(sum(t.square().sum() for t in value), (p, w, q))
                        return (*value, *gradient)
                    return value
                cases.append((f'{mode}_b{batch}_p{points}', operation))
        for lines in map(int, args.lines.split(',')):
            starts = torch.empty((batch, lines, 2), device='cuda', dtype=torch.float64)
            starts[..., 0] = torch.linspace(1.1, 1.4, lines, device='cuda')
            starts[..., 1] = torch.linspace(-.1, .1, lines, device='cuda')
            starts.requires_grad_()
            for mode in ('trace', 'trace_vjp'):
                def operation(mode=mode, starts=starts):
                    value = field.trace(starts, .03, .18, args.steps, return_samples=True)
                    if mode.endswith('_vjp'):
                        return (value, *torch.autograd.grad(value.square().sum(), (p, w, starts)))
                    return (value,)
                cases.append((f'{mode}_b{batch}_l{lines}', operation))
        for name, operation in cases:
            for _ in range(2):
                operation()
            torch.cuda.synchronize()
            wall, elapsed = [], []
            for _ in range(args.repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                begin.record()
                values = operation()
                end.record()
                end.synchronize()
                wall.append(time.perf_counter() - started)
                elapsed.append(begin.elapsed_time(end)*1e-3)
            arrays = [v.detach().cpu().numpy() for v in values]
            if not all(np.isfinite(v).all() for v in arrays):
                raise RuntimeError(f'nonfinite primitive output in {name}')
            np.savez(args.output/(name+'.npz'), **{f'a{i}': v for i, v in enumerate(arrays)})
            records.append(dict(case=name, gpu_seconds=elapsed, wall_seconds=wall,
                                gpu_median=statistics.median(elapsed), wall_median=statistics.median(wall)))
    result = dict(library=str(args.library.resolve()),
                  sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, dtype=args.dtype,
                  segments=args.segments, steps=args.steps, repeats=args.repeats, cases=records)
    (args.output/'timings.json').write_text(json.dumps(result, indent=2)+'\n')


def compare(args):
    import numpy as np

    if not args.baseline or not args.candidate:
        raise SystemExit('--baseline and --candidate are required')
    args.output.mkdir(parents=True, exist_ok=True)
    # One GPU, no concurrent benchmarks: avoid mutual kernel/resource contention.
    for label, library in [('baseline', args.baseline), ('candidate', args.candidate)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--library', str(library.resolve()),
                   '--output', str((args.output/label).resolve())]
        for name in ('queries', 'points', 'lines', 'segments', 'steps', 'repeats', 'dtype'):
            command.extend(['--'+name, str(getattr(args, name))])
        subprocess.run(command, check=True)
    baseline = json.loads((args.output/'baseline/timings.json').read_text())
    candidate = json.loads((args.output/'candidate/timings.json').read_text())
    compared = []
    for old, new in zip(baseline['cases'], candidate['cases']):
        assert old['case'] == new['case']
        name = old['case']
        with np.load(args.output/'baseline'/(name+'.npz')) as expected, np.load(args.output/'candidate'/(name+'.npz')) as actual:
            errors = []
            for key in expected.files:
                a, b = actual[key], expected[key]
                delta = a.astype(np.float64)-b.astype(np.float64)
                errors.append(dict(array=key, bitwise_equal=np.array_equal(a, b),
                                   max_abs=float(np.max(np.abs(delta), initial=0)),
                                   relative_l2=float(np.linalg.norm(delta.ravel()) /
                                                     max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-30))))
        compared.append(dict(case=name, baseline_gpu_s=old['gpu_median'], candidate_gpu_s=new['gpu_median'],
                             speedup=old['gpu_median']/new['gpu_median'],
                             bitwise_equal=all(e['bitwise_equal'] for e in errors), errors=errors))
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
