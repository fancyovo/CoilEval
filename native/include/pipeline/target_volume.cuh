#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"

#include "pipeline/types.cuh"
#include "target_volume_geometry.h"

namespace coil_eval::pipeline {

struct TargetVolumeCache {
    sgpu_target::Geometry geometry;
    std::vector<double> radii;
    std::vector<double> coefficients, adjoint_weights, axis_weights;
    double adjoint_residual=0;
    double level=0, volume=0, dVds=0, target=0;
    std::uint64_t id=0;
};

bool screen_target_volume_native(void* field,const AxisData& axis,const PsiData& psi,
    int nfp,const SgpuScoreConfig& config,std::vector<SurfaceScreen>& screens,SgpuScoreResult& result);

void target_quality(const SgpuScoreConfig& config,SgpuScoreResult& result);

bool predict_target_levels_gpu(const std::vector<AxisData>& axes,const PsiModes& modes,
    const double* coefficients,int nc,int nfp,const SgpuScoreConfig& config,
    const SgpuScoreResult& center,std::vector<double>& levels);

__global__ void target_boundary_guard_kernel(const float* polys,const float* radii,
    const float* residuals,const float* axis_R,const float* axis_R_phi,
    int axis_count,int nr,int nt,int nl,int degree,int nfp,float a,float tolerance,
    double* derivatives);


}  // namespace coil_eval::pipeline
