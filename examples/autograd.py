"""Differentiate a custom scalar objective with PyTorch."""
import argparse
from pathlib import Path
from time import perf_counter
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',help='Optional path to a custom native backend')
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--coils',type=Path,default=Path(__file__).with_name('coils.json'))
    parser.add_argument('--two-stage',action='store_true',default=True,help='S2 is the default; retained compatibility flag')
    parser.add_argument('--sampling',choices=('cartesian','polar_quadrature'),default='polar_quadrature')
    parser.add_argument('--alpha-weighting',choices=('histogram','smooth'),default='smooth')
    parser.add_argument('--alpha-basis',choices=('zernike','flux_legendre'),default='flux_legendre')
    parser.add_argument('--volume',type=float,default=.05,help='Target volume in m^3')
    parser.add_argument('--radius',type=float,default=.15,help='Fitting radius in metres')
    parser.add_argument('--device',type=int,default=0,help='CUDA device index')
    parser.add_argument('--helicity',type=int,nargs=2,default=(1,4),metavar=('M','N'),
                        help='QS helicity: QA 1 0; standard QH 1 nfp (default: 1 4)')
    args=parser.parse_args()
    if not args.two_stage and (args.sampling!='cartesian' or args.alpha_weighting!='histogram' or args.alpha_basis!='zernike'):
        parser.error('polar sampling, smooth alpha weights and flux_legendre alpha require --two-stage')
    coils=CoilSet.read_json(args.coils)
    x=pack_coils(coils,device=f'cuda:{args.device}')
    config=EvaluationConfig(target_volume=args.volume,psi_a=args.radius,helicity=tuple(args.helicity),fast=True,flux_method='stokes',
                            psi_fit='two_stage' if args.two_stage else 'single_stage',
                            psi_refit_sampling=args.sampling,alpha_radial_weighting=args.alpha_weighting,
                            alpha_radial_basis=args.alpha_basis)
    with TensorEvaluator(args.library,config=config,device_id=args.device,nfp=coils.nfp) as ev:
        torch.cuda.synchronize(ev.device)
        started=perf_counter()
        if args.batch:
            other=x.clone();other[:,-1]*=1.1
            result=ev.evaluate_batch(torch.stack((x,other)))
        else:
            result=ev.evaluate(x)
        m=result.metrics
        loss=torch.log(m.qs_volume+1e-8)+(m.iota_boundary-1.3).square()
        gradient,=torch.autograd.grad(loss.sum(),result.parameters)
        torch.cuda.synchronize(ev.device)
        elapsed=perf_counter()-started
        print('Objective:',loss.detach().cpu())
        print('Physical gradient:',gradient.cpu())
        print('Value + one gradient seconds/sample:',elapsed/(2 if args.batch else 1))

if __name__=='__main__':
    main()
