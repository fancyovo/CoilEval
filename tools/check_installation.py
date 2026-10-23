"""Exercise an installed binary wheel, including value and physical gradient."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import coil_eval
from coil_eval import CoilSet, EvaluationConfig, Evaluator, quality_penalty


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coils',type=Path)
    parser.add_argument('--torch',action='store_true')
    parser.add_argument('--require-no-nvcc',action='store_true')
    args=parser.parse_args()
    if args.require_no_nvcc:
        assert shutil.which('nvcc') is None, 'nvcc is visible in PATH'
    package=Path(coil_eval.__file__).resolve()
    assert 'site-packages' in package.parts, f'Expected installed package, got {package}'
    coils=CoilSet.read_json(args.coils)
    cfg=EvaluationConfig(target_volume=.05,fast=True,flux_method='stokes')
    scales=np.r_[np.ones(3*coils.coeffs_x.size),np.full(coils.n_base_coils,100000.)]
    with Evaluator(config=cfg) as ev:
        result=ev.loss_and_grad(coils,lambda m: np.log(m.qs_volume+1e-8)+
                                quality_penalty(m.relative_normal_drift_p95),scales=scales)
    assert result.center.valid and np.isfinite(result.gradient).all()
    assert np.linalg.norm(result.gradient)>0
    report={'version':coil_eval.__version__,'package':str(package),'nvcc':shutil.which('nvcc'),
            'loss':result.value,'gradient_norm':float(np.linalg.norm(result.gradient))}
    if args.torch:
        import torch
        from coil_eval.torch import TensorEvaluator,pack_coils
        with TensorEvaluator(config=cfg) as ev:
            result=ev.evaluate_batch(torch.stack((pack_coils(coils),pack_coils(coils))))
            grad,=torch.autograd.grad(torch.log(result.metrics.qs_volume+1e-8).sum(),result.parameters)
        assert torch.isfinite(grad).all() and torch.linalg.vector_norm(grad)>0
        torch.testing.assert_close(grad[0],grad[1])
        report['torch_gradient_norm']=float(torch.linalg.vector_norm(grad))
    maps=Path('/proc/self/maps')
    if maps.is_file():
        libraries=sorted({line.split()[-1] for line in maps.read_text().splitlines()
                          if any(name in line for name in ('libcublas','libcusolver','libcusparse',
                                                          'libcudart','libnvJitLink','libcoil_eval'))})
        assert all('site-packages' in path for path in libraries), libraries
        report['loaded_libraries']=libraries
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
