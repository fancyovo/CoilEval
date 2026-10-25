// Single definitions of per-thread capture and derivative state.
#include "pipeline/types.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/gradients.cuh"
#include "pipeline/target_volume.cuh"

namespace coil_eval::pipeline {

thread_local const std::vector<double>* g_active_psi_warm_coefficients = nullptr;

thread_local int g_active_psi_warm_iterations = 0;

thread_local bool g_active_psi_warm_preconditioned = false;

thread_local TargetVolumeCache g_target_volume_cache;

thread_local std::uint64_t g_target_volume_sequence=0;

thread_local FixedFrontG2Cache* g_active_g2_cache = nullptr;

thread_local int g_active_gradient_group = 0;

thread_local AlphaWarmPreconditionerCache g_alpha_warm_preconditioner;

thread_local AlphaRadialWeightingOptions g_alpha_radial_weighting;

thread_local std::vector<float> g_captured_iota_profile;

thread_local double g_boundary_iota = std::numeric_limits<double>::quiet_NaN();

thread_local std::vector<double> g_batch_boundary_iota;


}  // namespace coil_eval::pipeline
