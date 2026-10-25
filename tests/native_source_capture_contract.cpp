// Exercise the actual snapshot builder with host-backed CUDA buffer stubs.
#include "pipeline/reverse_capture.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/target_volume.cuh"
#include "coil_field_internal.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>

int copies=0, factors=0, predictions=0;
bool fail_copy=false;
std::string last_error;
cudaError_t cudaMalloc(void** p,size_t n) { *p=std::malloc(n);return *p?0:1; }
cudaError_t cudaFree(void* p) { std::free(p);return 0; }
cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind) {
    ++copies;if(fail_copy)return 1;std::memcpy(dst,src,n);return 0;
}
cudaError_t cudaMemcpy2D(void*,size_t,const void*,size_t,size_t,size_t,cudaMemcpyKind) { return 1; }
cudaError_t cudaSetDevice(int) { return 0; }
extern "C" void sgpu_internal_set_error(const char* error) { last_error=error; }
extern "C" int sgpu_copy_psi_warm_preconditioner(double* upper,double* scales,int count) {
    ++factors;
    for(int i=0;i<count;++i) { scales[i]=i+1;for(int j=0;j<count;++j)upper[i*count+j]=i==j?1:0; }
    return 0;
}
namespace coil_eval::pipeline {
thread_local TargetVolumeCache g_target_volume_cache;
thread_local AlphaWarmPreconditionerCache g_alpha_warm_preconditioner;
bool predict_target_levels_gpu(const std::vector<AxisData>&,const PsiModes&,const double*,int,int,
                              const SgpuScoreConfig&,const SgpuScoreResult&,std::vector<double>& levels) {
    ++predictions;levels={g_target_volume_cache.level};return true;
}
}

using namespace coil_eval::pipeline;
std::map<std::string,std::vector<double>> snapshot() {
    std::map<std::string,std::vector<double>> out;
    for(const auto& a:g_reverse_snapshot)out[a.name]=a.values;
    return out;
}

int main() {
    FixedFrontG2Cache cache;cache.ready=true;
    cache.axis.R={1,1.1};cache.axis.Z={.1,.2};cache.axis.R_phi={.2,.3};cache.axis.Z_phi={.4,.5};
    cache.psi.coeffs={.2,.4};cache.psi.modes.a={1,2};cache.psi.modes.b={2,3};
    cache.psi.modes.m={0,1};cache.psi.modes.kind={0,1};
    auto& a=cache.alpha;
    a.adjoint_ready=true;a.mode_count=2;a.column_count=3;a.qr_rows=3;
    a.modes.l={0,1};a.modes.m={1,2};a.modes.n={0,1};a.modes.kind={0,1};
    a.modes.radial_coefficients={1,2};a.modes.radial_order=0;a.iota_coefficients={1.2f,.1f};
    a.modes.radial_basis=1;
    assert(a.scales.allocate(3) && a.scaled_solution.allocate(3));
    const float scale[]={2,4,8}, scaled[]={3,5,7};
    std::memcpy(a.scales.data(),scale,sizeof(scale));std::memcpy(a.scaled_solution.data(),scaled,sizeof(scaled));
    g_alpha_warm_preconditioner.upper_factor={1,0,0,0,1,0,0,0,1};
    g_alpha_warm_preconditioner.center_scale={2,4,8};
    g_alpha_warm_preconditioner.center_scaled_solution={3,5,7};
    auto& t=g_target_volume_cache;t.id=17;t.level=.25;t.target=.1;t.coefficients=cache.psi.coeffs;
    t.geometry.rays.resize(1);t.radii={.2};
    SgpuScoreConfig cfg{};cfg.target_volume=.1;cfg.flux_phi_count=1;cfg.flux_theta_count=1;
    cfg.volume_point_count=2;cfg.volume_phi_count=2;cfg.volume_theta_count=1;
    assert(cache.points.allocate(2));assert(cache.points.source_indices.allocate(2));
    assert(cache.B.allocate(6) && cache.grad_B.allocate(18));
    const int selected[]={0,3};std::memcpy(cache.points.source_indices.data(),selected,sizeof(selected));
    SgpuScoreResult result{};result.target_volume_cache_id=17;
    assert(sgpu_set_reverse_capture(1)==0);
    assert(capture_reverse_snapshot(cache,4,cfg,result));
    const auto full=snapshot();assert(factors==1 && predictions==1);
    assert(full.at("volume_source_indices")==std::vector<double>({0,3}));
    assert(full.at("volume_sampling_shape")==std::vector<double>({2,1,2,4}));
    assert(full.count("psi_upper") && full.count("alpha_upper") && full.count("target_adjoint_weights"));
    assert(sgpu_set_reverse_capture(4)==0);
    const int initial_copies=copies;
    assert(capture_reverse_snapshot(cache,4,cfg,result));
    const auto source=snapshot();
    assert(factors==1 && predictions==1 && copies-initial_copies==2);
    assert(!source.count("psi_upper") && !source.count("alpha_upper") && !source.count("target_adjoint_weights"));
    assert(!source.count("volume_source_indices") && !source.count("volume_sampling_shape"));
    assert(source.at("alpha_coefficients")==std::vector<double>({1.5,1.25}));
    assert(source.at("alpha_radial_basis_mode")==std::vector<double>({1.}));
    for(const auto& item:source)
        if(item.first!="alpha_coefficients")assert(full.at(item.first)==item.second);
    const auto size=g_reverse_snapshot.size();
    assert(sgpu_source_capture_version()==1 && g_reverse_capture_flags==4 && g_reverse_snapshot.size()==size);
    assert(sgpu_set_reverse_capture(2)!=0 && g_reverse_capture_flags==4 && g_reverse_snapshot.size()==size);
    t.id=18;assert(!capture_reverse_snapshot(cache,4,cfg,result));assert(g_reverse_snapshot.empty());t.id=17;
    fail_copy=true;assert(!capture_reverse_snapshot(cache,4,cfg,result));assert(g_reverse_snapshot.empty());fail_copy=false;
    a.adjoint_ready=false;assert(!capture_reverse_snapshot(cache,4,cfg,result));assert(g_reverse_snapshot.empty());a.adjoint_ready=true;
    assert(sgpu_set_reverse_capture(3)==0 && capture_reverse_snapshot(cache,4,cfg,result));
    assert(snapshot().count("points_xyz") && snapshot().count("target_radii"));
    assert(snapshot().at("volume_source_indices")==full.at("volume_source_indices"));
    assert(factors==2 && predictions==2);
    assert(sgpu_set_reverse_capture(0)==0 && g_reverse_snapshot.empty());
    std::cout<<"source/full snapshot parity, flag lifecycle, two-vector copies and skipped reverse contractions passed\n";
}
