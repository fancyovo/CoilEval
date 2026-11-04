"""Run the actual formal/local production kernels with multi-root polynomials."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
def test_native_formal_and_local_first_branch(tmp_path):
    library = os.environ.get('COIL_EVAL_TEST_LIBRARY')
    compiler = shutil.which('nvcc')
    if not library or not compiler:
        pytest.skip('set COIL_EVAL_TEST_LIBRARY and provide the matching CUDA compiler')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    library = Path(library).resolve()
    executable = tmp_path/'radial-native-gpu'
    architecture = ''.join(map(str, torch.cuda.get_device_capability()))
    command = [compiler, '-std=c++17', '-O3', '-rdc=true', f'-arch=sm_{architecture}',
               '-I'+str(PACKAGE/'native/include'), '-I'+str(PACKAGE/'native/src')]
    if os.environ.get('CXX'):
        command += ['-ccbin', os.environ['CXX']]
    command += [str(PACKAGE/'tests/radial_first_branch_cuda_test.cu')]
    command += [str(PACKAGE/'native/src/pipeline'/name)
                for name in ('flux.cu','local_kernels.cu','surface.cu')]
    command += [str(library), '-Xlinker','-rpath','-Xlinker',str(library.parent),
                '-o',str(executable)]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout+build.stderr
    fixture = json.loads((PACKAGE/'tests/fixtures/radial_first_branch.json').read_text())
    rays = fixture['rays']
    inputs = [f"{len(rays)} {len(rays[0]['polynomial'])-1} {fixture['level']}"]
    inputs += [' '.join(map(repr,ray['polynomial']+[ray['expected_first_root']]))
               for ray in rays]
    run = subprocess.run([str(executable)], input='\n'.join(inputs),
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert 'production formal/local GPU and host first-branch regression passed: 16 rays' in run.stdout
