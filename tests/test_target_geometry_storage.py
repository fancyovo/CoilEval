"""Preserve exact Bernstein decisions and selected boundaries after storage reuse."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


def test_bernstein_storage_and_coefficient_reuse(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++17 compiler unavailable')
    root=Path(__file__).resolve().parents[1]
    binary=tmp_path/('target-storage.exe' if os.name=='nt' else 'target-storage')
    subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off',
        '-I'+str(root/'native/include'),str(root/'tests/target_geometry_storage_test.cpp'),
        '-o',str(binary)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    assert '1000 polynomial boundary branches identical' in result.stdout
