#pragma once

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

bool score_local_batch_impl(
    void* batch_field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int axis_count,
    const double* axis_residual,
    const double* axis_topology_trace,
    const double* axis_topology_det,
    const double* psi_coefficients_host,
    int psi_coefficient_count,
    const double* psi_train_rms,
    const double* coil_components,
    const SgpuScoreResult& center,
    const SgpuScoreConfig& config,
    int surface_theta_count,
    int alpha_iterations,
    SgpuScoreResult* results,
    double* stats_out,
    int stats_len
);


}  // namespace coil_eval::pipeline
