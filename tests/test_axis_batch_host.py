"""Host contracts for one-shot ownership and unchanged finite-state refinement."""
from pathlib import Path
import shutil
import subprocess
import pytest

PACKAGE=Path(__file__).resolve().parents[1]

def compile_run(source,tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('host C++17 compiler unavailable')
    cpp=tmp_path/'contract.cpp';cpp.write_text(source,encoding='utf-8')
    exe=tmp_path/('contract.exe' if __import__('os').name=='nt' else 'contract')
    result=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off',
        '-I'+str(PACKAGE/'native/include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_exact_physical_config_grid_keys_and_single_take(tmp_path):
    source=(PACKAGE/'tests/native_axis_batch_cache_contract.cpp').read_text(encoding='utf-8')
    assert 'ownership/key contracts passed' in compile_run(source,tmp_path)


def test_modular_axis_entrypoints_have_consistent_host_declarations(tmp_path):
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
    modules=['axis.cu','axis_batch.cu','api.cu','two_stage.cu']
    command=[compiler,'-x','c++','-fsyntax-only','-std=c++17','-I'+str(tmp_path),
             '-I'+str(PACKAGE/'native/include'),'-I'+str(PACKAGE/'native/src')]
    command.extend(str(PACKAGE/'native/src/pipeline'/name) for name in modules)
    checked=subprocess.run(command,capture_output=True,text=True)
    assert checked.returncode==0,checked.stdout+checked.stderr


def test_original_refinement_matches_batched_rows_bitwise(tmp_path):
    source=(PACKAGE/'native/src/pipeline/axis.cu').read_text(encoding='utf-8')
    begin=source.index('bool refine_axis_candidates(')
    reference=source[begin:source.index('void assign_axis_topology(',begin)]
    prefix=r'''
#include "coil_field.h"
#include "pipeline/axis_refinement_batch.cuh"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
struct AxisCandidate { double R, Z, residual = 0.; bool winding = false; };
struct AxisDomain { double r_min, r_max, z_min, z_max; };
struct Map { int kind; double r, z; };
int calls = 0;
bool trace_map(void* opaque, const std::vector<double>& R, const std::vector<double>& Z,
               int, int, bool, std::vector<double>& re, std::vector<double>& ze) {
    const auto& map = *static_cast<Map*>(opaque);
    ++calls; re.resize(R.size()); ze.resize(Z.size());
    for (size_t i = 0; i < R.size(); ++i) {
        const double x = R[i] - map.r, y = Z[i] - map.z;
        if (map.kind == 0) {
            re[i] = R[i] + .4*x + .2*y + 3.*x*x*x;
            ze[i] = Z[i] - .2*x + .5*y + 2.*y*y*y;
        } else if (map.kind == 1) {
            // Constant displacement yields a singular residual Jacobian.
            re[i] = R[i] + .1; ze[i] = Z[i] - .1;
        } else if (map.kind == 2) {
            re[i] = R[i] + x*x*x - 2.*x + .2;
            ze[i] = Z[i] + .4*y + .1*x;
        } else if (map.kind == 3) {
            re[i] = std::numeric_limits<double>::quiet_NaN(); ze[i] = Z[i];
        } else {
            re[i] = R[i]; ze[i] = Z[i];
        }
    }
    return true;
}
'''
    suffix=r'''
int main() {
    constexpr int Q = 8;
    std::vector<Map> maps = {{0,1.2,.1},{1,1.1,0.},{2,1.0,.1},{3,1.1,.2},
                            {4,1.,0.},{0,1.3,-.2},{0,1.,0.},{2,1.2,.1}};
    std::vector<AxisDomain> domains(Q, {.1,2.1,-.8,.8});
    std::vector<SgpuScoreConfig> configs(Q);
    std::vector<std::vector<AxisCandidate>> reference(Q);
    for (int q = 0; q < Q; ++q) {
        auto& c = configs[q];
        c.axis_fd_absolute = 1e-5; c.axis_fd_relative = 1e-4;
        c.axis_tolerance = 1e-8; c.axis_trace_steps = 240;
        c.axis_newton_iters = q == 5 ? 0 : (q == 7 ? 2 : 6);
        if (q != 6) {
            for (int i = 0; i < 1 + q % 4; ++i)
                reference[q].push_back({.3 + .4*i, -.3 + .17*i, 0., i%2 != 0});
        }
    }
    auto batched = reference;
    for (int q = 0; q < Q; ++q)
        assert(refine_axis_candidates(&maps[q], reference[q], domains[q], 4,
                                      configs[q], configs[q].axis_newton_iters));
    const int reference_calls = calls;
    int batch_calls = 0;
    auto trace = [&](const std::vector<std::vector<double>>& r,
                     const std::vector<std::vector<double>>& z,
                     std::vector<std::vector<double>>& re,
                     std::vector<std::vector<double>>& ze) {
        ++batch_calls; re.resize(Q); ze.resize(Q);
        for (int q = 0; q < Q; ++q)
            if (!trace_map(&maps[q], r[q], z[q], 4, 240, false, re[q], ze[q])) return false;
        return true;
    };
    assert(sgpu_axis_refinement::refine(batched, domains, configs, trace));
    size_t compared = 0;
    for (int q = 0; q < Q; ++q) {
        assert(batched[q].size() == reference[q].size());
        for (size_t i = 0; i < reference[q].size(); ++i) {
            const double a[] = {reference[q][i].R, reference[q][i].Z, reference[q][i].residual};
            const double b[] = {batched[q][i].R, batched[q][i].Z, batched[q][i].residual};
            if (std::memcmp(a,b,sizeof(a)) != 0) {
                std::cerr << "query " << q << " candidate " << i << " differs\n";
                return 1;
            }
            assert(batched[q][i].winding == reference[q][i].winding);
            compared += 3;
        }
    }
    assert(batch_calls < reference_calls);
    std::cout << "passed " << compared << " scalar bit patterns; batched trace calls "
              << batch_calls << " versus sequential " << reference_calls << "\n";
}
'''
    assert 'scalar bit patterns' in compile_run(prefix+reference+suffix,tmp_path)
