"""Missing extras fail before backend loading, without eager optional imports."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from coil_eval import EvaluationConfig, Evaluator
from coil_eval import _dependencies
from coil_eval.qs import SurfaceQSEvaluator


@pytest.mark.parametrize('missing', [('scipy',), ('threadpoolctl',), ('torch',),
                                   ('torch', 'scipy', 'threadpoolctl')])
@pytest.mark.parametrize('factory', [Evaluator, SurfaceQSEvaluator])
def test_cpu_missing_dependencies_are_reported_together(monkeypatch, missing, factory):
    monkeypatch.setattr(_dependencies, 'find_spec', lambda name: None if name in missing else object())
    with pytest.raises(ModuleNotFoundError) as error:
        factory(config=EvaluationConfig(target_volume=.05), device='cpu')
    text = str(error.value)
    for name in missing:
        assert name in text
    assert '.whl[cpu]' in text and 'same Python environment' in text
    if 'torch' in missing:
        assert 'https://download.pytorch.org/whl/cpu' in text
    assert error.value.name in missing


def test_tensor_cpu_checks_remaining_dependencies(monkeypatch):
    pytest.importorskip('torch')
    from coil_eval.torch import TensorEvaluator
    monkeypatch.setattr(_dependencies, 'find_spec', lambda name: None if name == 'scipy' else object())
    with pytest.raises(ModuleNotFoundError, match='scipy'):
        TensorEvaluator(config=EvaluationConfig(target_volume=.05), device='cpu')


def test_broken_dependency_is_not_relabelled_as_missing(monkeypatch):
    def broken(name):
        raise ImportError('broken search path')
    monkeypatch.setattr(_dependencies, 'find_spec', broken)
    with pytest.raises(ImportError, match='broken search path'):
        _dependencies.require_dependencies('cpu')


def test_base_import_never_loads_optional_dependencies():
    import coil_eval
    code = '''
import importlib.abc, json, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'threadpoolctl', 'simsopt'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import coil_eval
from coil_eval.qs import SurfaceQSConfig
print(json.dumps({'version': coil_eval.__version__}))
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(coil_eval.__file__).resolve().parent.parent)}
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout)['version'] == coil_eval.__version__


def test_tensor_import_missing_torch_has_installation_hint(monkeypatch):
    import coil_eval
    code = '''
from coil_eval import _dependencies
original = _dependencies.find_spec
_dependencies.find_spec = lambda name: None if name == 'torch' else original(name)
try:
    import coil_eval.torch
except ModuleNotFoundError as error:
    assert error.name == 'torch'
    assert '.whl[torch]' in str(error)
    assert 'CUDA 12.8' in str(error)
else:
    raise AssertionError('expected missing dependency')
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(coil_eval.__file__).resolve().parent.parent)}
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
