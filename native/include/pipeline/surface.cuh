#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct SurfaceScreen {
    double level = std::numeric_limits<double>::quiet_NaN();
    double drift_p95 = std::numeric_limits<double>::infinity();
    double relative_drift_p95 = std::numeric_limits<double>::infinity();
    double radius_mean = std::numeric_limits<double>::quiet_NaN();
    double radius_max = std::numeric_limits<double>::quiet_NaN();
    double one_period_relative_drift_p95 = std::numeric_limits<double>::infinity();
    int long_trace_periods_completed = 0;
    bool stable = false;
    bool strict = false;
    bool verified = false;
    bool long_verified = false;
};

void ray_polynomial_phi0(
    const PsiData& psi,
    const SgpuScoreConfig& config,
    double theta,
    std::array<double, 25>& polynomial
);

double solve_ray_radius(
    const std::array<double, 25>& polynomial,
    double level,
    const SgpuScoreConfig& config
);

bool screen_surfaces_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens,
    double* timings
);

bool screen_surface_confidence_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
);

bool verify_surface_long_horizon(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    SurfaceScreen& screen,
    double* timings
);


}  // namespace coil_eval::pipeline
