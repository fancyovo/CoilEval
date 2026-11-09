#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

bool run_downstream_gpu(
    void* field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const AxisData& axis,
    const PsiData& psi,
    std::vector<SurfaceScreen>& screens,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    bool verify_long_horizon = true,
    bool surface_geometry_only = false
);


}  // namespace coil_eval::pipeline
