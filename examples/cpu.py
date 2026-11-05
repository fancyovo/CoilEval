"""Evaluate the same physical model on a computer without an NVIDIA GPU."""
import argparse
import json
from pathlib import Path
from coil_eval import CoilSet, EvaluationConfig, Evaluator

parser=argparse.ArgumentParser()
parser.add_argument('--coils',type=Path,default=Path(__file__).with_name('coils.json'))
parser.add_argument('--volume',type=float,default=.05)
parser.add_argument('--radius',type=float,default=.15)
parser.add_argument('--threads',type=int)
parser.add_argument('--axis-grid',type=int,help='CPU primary magnetic-axis seed grid; default 48')
parser.add_argument('--library',type=Path)
parser.add_argument('--two-stage',action='store_true',default=True,help='S2 is the default; retained compatibility flag')
parser.add_argument('--helicity',type=int,nargs=2,default=(1,4),metavar=('M','N'),
                    help='QS helicity: QA 1 0; standard QH 1 nfp (default: 1 4)')
args=parser.parse_args()
coils=CoilSet.read_json(args.coils)
config=EvaluationConfig(target_volume=args.volume,psi_a=args.radius,helicity=tuple(args.helicity),fast=True,
    psi_fit='two_stage' if args.two_stage else 'single_stage',
    psi_refit_sampling='polar_quadrature',
    alpha_radial_basis='flux_legendre' if args.two_stage else 'zernike')
with Evaluator(args.library,config=config,device='cpu',threads=args.threads,axis_grid=args.axis_grid) as evaluator:
    result=evaluator.evaluate(coils)
print(json.dumps(result.to_dict(),indent=2,allow_nan=False))
