#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

bool validate_config(const SgpuScoreConfig& config, std::string& reason);

void fill_early_components(
    const SgpuScoreConfig& config,
    const CoilMetrics& coil,
    const AxisData& axis,
    const PsiData& psi,
    const std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
);

void finalize_score(const SgpuScoreConfig& config, SgpuScoreResult& result);


}  // namespace coil_eval::pipeline
