"""Actual native warm-branch ownership and local row decoding without CUDA."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def test_warm_capture_and_local_kernel_share_exact_formal_source_rows(tmp_path):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++17 compiler unavailable')
    (tmp_path/'cuda_runtime.h').write_text('''#pragma once
#include <cstddef>
#define __global__
typedef int cudaError_t;
constexpr int cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
cudaError_t cudaFree(void*);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaMemcpy(void*,const void*,std::size_t,cudaMemcpyKind);
''', encoding='utf-8')
    volume = (PACKAGE/'native/src/pipeline/volume.cu').read_text(encoding='utf-8')
    helper = volume[volume.index('bool capture_alpha_volume_sampling('):
                    volume.index('bool build_volume_points_native(')]
    kernels = (PACKAGE/'native/src/pipeline/local_kernels.cu').read_text(encoding='utf-8')
    start = kernels.index('__global__ void local_batch_generate_volume_points_kernel(')
    body = kernels.index('    const int candidate_count =', start)
    decode = kernels[body:kernels.index('    const float boundary =', body)]
    program = r'''
#include "pipeline/alpha.cuh"
#include "pipeline/session_state.cuh"
#include "coil_field_internal.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
using std::min;
bool fail_copy=false;
cudaError_t cudaMalloc(void** p,size_t n) {*p=std::malloc(n);return *p?0:1;}
cudaError_t cudaFree(void* p) {std::free(p);return 0;}
cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind) {
    if(fail_copy)return 1;std::memcpy(d,s,n);return 0;
}
extern "C" void sgpu_internal_set_error(const char*) {}
namespace coil_eval::pipeline {
thread_local AlphaWarmPreconditionerCache g_alpha_warm_preconditioner;
''' + helper + r'''
}
using namespace coil_eval::pipeline;
std::array<int,3> decode_row(const int* source_indices,int output_index,
                             int n_phi,int n_theta,int n_radial,int output_count) {
''' + decode + r'''
    return {phi_index,theta_index,radial_index};
}
int main() {
    DeviceVolumePoints points;points.count=6;assert(points.source_indices.allocate(6));
    const int selected[]={0,1,4,5,6,7};
    std::memcpy(points.source_indices.data(),selected,sizeof(selected));
    SgpuScoreConfig config{};config.volume_point_count=6;config.volume_phi_count=2;
    config.volume_theta_count=2;config.volume_rho_min=.08;config.device_id=2;
    assert(capture_alpha_volume_sampling(points,config,77));
    auto& warm=g_alpha_warm_preconditioner;
    assert(warm.volume_sampling_ready && warm.volume_radial_count==2);
    assert(warm.volume_device_id==2 && warm.volume_target_cache_id==77);
    assert(warm.volume_source_indices==std::vector<int>(selected,selected+6));
    for(int k=0;k<6;++k) {
        const auto decoded=decode_row(warm.volume_source_indices.data(),k,2,2,2,6);
        const int source=(decoded[0]*2+decoded[1])*2+decoded[2];
        assert(source==selected[k]);
    }
    // A different center is not borrowed through a dangling GPU pointer.
    points.source_indices.data()[0]=1;
    assert(warm.volume_source_indices.front()==0);
    assert(!capture_alpha_volume_sampling(points,config,78));
    assert(!warm.volume_sampling_ready && warm.volume_source_indices.empty());
    points.source_indices.data()[0]=0;fail_copy=true;
    assert(!capture_alpha_volume_sampling(points,config,79));
    assert(!warm.volume_sampling_ready && warm.volume_source_indices.empty());
    fail_copy=false;assert(capture_alpha_volume_sampling(points,config,80));
    config.volume_point_count=5;
    assert(!capture_alpha_volume_sampling(points,config,81));
    assert(!warm.volume_sampling_ready && warm.volume_source_indices.empty());
    std::cout << "formal selection ownership and native local row contract passed\n";
}
'''
    cpp = tmp_path/'volume-contract.cpp'
    cpp.write_text(program, encoding='utf-8')
    exe = tmp_path/('volume-contract.exe' if os.name=='nt' else 'volume-contract')
    build = subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off',
        '-I'+str(tmp_path), '-I'+str(PACKAGE/'native/include'),
        '-I'+str(PACKAGE/'native/src'), str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert build.returncode==0, build.stdout+build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode==0, run.stdout+run.stderr
    assert 'native local row contract passed' in run.stdout
