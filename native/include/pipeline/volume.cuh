#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/flux.cuh"
#include "pipeline/psi.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct DeviceVolumePoints {
    DeviceBuffer<float> xyz;
    DeviceBuffer<float> grad_s;
    DeviceBuffer<float> grad_theta;
    DeviceBuffer<float> grad_phi;
    DeviceBuffer<float> s;
    DeviceBuffer<float> rho;
    DeviceBuffer<float> theta;
    DeviceBuffer<float> phi;
    DeviceBuffer<float> R;
    DeviceBuffer<float> volume_weight;
    DeviceBuffer<float> flux_derivative;
    int count = 0;

    bool allocate(int point_count) {
        count = point_count;
        return xyz.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_s.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_theta.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_phi.allocate(static_cast<size_t>(point_count) * 3) &&
               s.allocate(point_count) && rho.allocate(point_count) &&
               theta.allocate(point_count) && phi.allocate(point_count) &&
               R.allocate(point_count) && volume_weight.allocate(point_count) &&
               flux_derivative.allocate(point_count);
    }
};

bool build_volume_points_native(
    const DevicePsiData& device_psi,
    const AxisData& axis,
    const PsiData& psi,
    const FluxCalibrationNative& flux,
    int nfp,
    const SgpuScoreConfig& config,
    DeviceVolumePoints& output,
    SgpuScoreResult& result
);


}  // namespace coil_eval::pipeline
