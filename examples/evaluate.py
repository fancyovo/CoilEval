"""Evaluate the bundled coil set."""
import argparse
from pathlib import Path
from coil_eval import CoilSet, EvaluationConfig, Evaluator

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--coils', type=Path, default=Path(__file__).with_name('coils.json'))
    args = parser.parse_args()
    with Evaluator(args.library, config=EvaluationConfig(target_volume=0.05)) as ev:
        result = ev.evaluate(CoilSet.read_json(args.coils))
        print(result.to_dict())

if __name__ == '__main__':
    main()
