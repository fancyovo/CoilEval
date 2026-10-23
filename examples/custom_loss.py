"""Compute a custom objective and physical parameter gradient."""
import argparse
from pathlib import Path
import numpy as np
from coil_eval import CoilSet, EvaluationConfig, Evaluator, quality_penalty

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', help='Optional path to a custom native backend')
    args = parser.parse_args()
    coils = CoilSet.read_json(Path(__file__).with_name('coils.json'))
    scales = np.r_[np.ones(3*coils.coeffs_x.size), np.full(coils.n_base_coils, 100000.)]
    def loss(metrics):
        return np.log(metrics.qs_volume + 1e-8) + quality_penalty(metrics.relative_normal_drift_p95)
    with Evaluator(args.library, config=EvaluationConfig(target_volume=0.05)) as ev:
        result = ev.loss_and_grad(coils, loss, scales=scales)
        print('Value:', result.value)
        print('Physical gradient:', result.gradient)

if __name__ == '__main__':
    main()
