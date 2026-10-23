"""Independent single/batch value, gradient and cross-query regression."""
import gc
import os
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, TensorMetrics, pack_coils

LIBRARY=os.environ.get('COIL_EVAL_TEST_LIBRARY')
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not LIBRARY,reason='set COIL_EVAL_TEST_LIBRARY')]

def test_single_batch_values_gradients_and_isolation():
    coils=CoilSet.read_json(Path(__file__).parents[1]/'examples/coils.json')
    x=pack_coils(coils)
    second=x.clone();second[:,27]*=1.1
    configs=[EvaluationConfig(target_volume=v,fast=True,flux_method='stokes') for v in (.05,.04)]
    names=[f.name for f in fields(TensorMetrics)]
    scales=torch.ones_like(x);scales[:,27]=100000.
    values=[];gradients=[]
    with TensorEvaluator(LIBRARY,config=configs[0]) as ev:
        for i,p in enumerate((x,second)):
            with TensorEvaluator(LIBRARY,config=configs[i]) as one:
                out=one.evaluate(p)
                values.append(torch.stack([getattr(out.metrics,k).detach() for k in names]))
                gradients.append(torch.stack([torch.autograd.grad(getattr(out.metrics,k),out.parameters,
                    retain_graph=j<len(names)-1)[0].detach() for j,k in enumerate(names)]))
                del out
        gc.collect()
        out=ev.evaluate_batch(torch.stack((x,second)),configs=configs)
        for j,name in enumerate(names):
            value=getattr(out.metrics,name)
            expected=torch.stack([v[j] for v in values])
            torch.testing.assert_close(value.detach().double(),expected.double(),rtol=2e-3,atol=5e-7)
            g=torch.autograd.grad(value.sum(),out.parameters,retain_graph=True)[0]*scales
            ref=torch.stack([v[j] for v in gradients])*scales
            for a,b in zip(g,ref):
                an,bn=torch.linalg.vector_norm(a),torch.linalg.vector_norm(b)
                if max(an,bn)<1e-5:
                    assert torch.linalg.vector_norm(a-b)<1e-5
                else:
                    assert torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)>.99,name
                    assert .98<an/bn<1.02,name
            isolated=torch.autograd.grad(value[0],out.parameters,retain_graph=j<len(names)-1)[0]
            assert torch.count_nonzero(isolated[1])==0,name
