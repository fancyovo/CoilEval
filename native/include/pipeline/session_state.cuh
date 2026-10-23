#pragma once

#include "pipeline/types.cuh"

// One definition in session_state.cu. State is per host thread, as in the original ABI.
// Capture/evaluate/local-batch calls retain their original initialization and invalidation order.
namespace coil_eval::pipeline {

struct TargetVolumeCache;
struct FixedFrontG2Cache;
struct AlphaWarmPreconditionerCache;


extern thread_local const std::vector<double>* g_active_psi_warm_coefficients;
extern thread_local int g_active_psi_warm_iterations;
extern thread_local bool g_active_psi_warm_preconditioned;
extern thread_local TargetVolumeCache g_target_volume_cache;
extern thread_local std::uint64_t g_target_volume_sequence;
extern thread_local FixedFrontG2Cache* g_active_g2_cache;
extern thread_local int g_active_gradient_group;
extern thread_local AlphaWarmPreconditionerCache g_alpha_warm_preconditioner;
extern thread_local std::vector<float> g_captured_iota_profile;
extern thread_local double g_boundary_iota;
extern thread_local std::vector<double> g_batch_boundary_iota;
}  // namespace coil_eval::pipeline
