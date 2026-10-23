"""Differentiate a custom scalar objective with PyTorch."""
import argparse
from pathlib import Path
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',help='Optional path to a custom native backend')
    parser.add_argument('--batch',action='store_true')
    args=parser.parse_args()
    coils=CoilSet.read_json(Path(__file__).with_name('coils.json'))
    x=pack_coils(coils)
    config=EvaluationConfig(target_volume=.05,fast=True,flux_method='stokes')
    with TensorEvaluator(args.library,config=config) as ev:
        if args.batch:
            other=x.clone();other[:,27]*=1.1
            result=ev.evaluate_batch(torch.stack((x,other)))
        else:
            result=ev.evaluate(x)
        m=result.metrics
        loss=torch.log(m.qs_volume+1e-8)+(m.iota_boundary-1.3).square()
        gradient,=torch.autograd.grad(loss.sum(),result.parameters)
        print('Objective:',loss.detach().cpu())
        print('Physical gradient:',gradient.cpu())

if __name__=='__main__':
    main()
