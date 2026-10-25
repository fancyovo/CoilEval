"""Actual native source/full snapshot construction without a CUDA installation."""
from pathlib import Path
import shutil
import subprocess

import pytest


PACKAGE=Path(__file__).resolve().parents[1]


def test_source_snapshot_matches_full_fields_and_skips_reverse_work(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:pytest.skip('host C++17 compiler unavailable')
    (tmp_path/'cuda_runtime.h').write_text('''#pragma once
#include <cstddef>
#define __global__
typedef int cudaError_t;
constexpr int cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
cudaError_t cudaFree(void*);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaMemcpy(void*,const void*,std::size_t,cudaMemcpyKind);
cudaError_t cudaMemcpy2D(void*,std::size_t,const void*,std::size_t,std::size_t,std::size_t,cudaMemcpyKind);
cudaError_t cudaSetDevice(int);
''',encoding='utf-8')
    exe=tmp_path/('source-contract.exe' if __import__('os').name=='nt' else 'source-contract')
    cmd=[compiler,'-x','c++','-std=c++17','-O2','-ffp-contract=off','-I'+str(tmp_path),
         '-I'+str(PACKAGE/'native/include'),'-I'+str(PACKAGE/'native/src'),
         str(PACKAGE/'native/src/pipeline/reverse_capture.cu'),
         str(PACKAGE/'tests/native_source_capture_contract.cpp'),'-o',str(exe)]
    build=subprocess.run(cmd,capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'skipped reverse contractions passed' in run.stdout
