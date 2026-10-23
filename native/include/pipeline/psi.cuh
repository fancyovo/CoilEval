#pragma once
#include "pipeline/axis.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct PsiModes {
    std::vector<int> a;
    std::vector<int> b;
    std::vector<int> m;
    std::vector<int> kind;
};

struct PsiData {
    PsiModes modes;
    std::vector<double> coeffs;
    double train_rms = std::numeric_limits<double>::quiet_NaN();
    double angle_mean = std::numeric_limits<double>::quiet_NaN();
    double angle_p95 = std::numeric_limits<double>::quiet_NaN();
    double angle_l2 = std::numeric_limits<double>::quiet_NaN();
};

struct DevicePsiData {
    DeviceBuffer<float> coeffs;
    DeviceBuffer<int> mode_a;
    DeviceBuffer<int> mode_b;
    DeviceBuffer<int> mode_m;
    DeviceBuffer<int> mode_kind;
    DeviceBuffer<float> axis_R;
    DeviceBuffer<float> axis_Z;
    DeviceBuffer<float> axis_R_phi;
    DeviceBuffer<float> axis_Z_phi;
};

void periodic_hermite_host(
    double phi,
    const std::vector<double>& values,
    const std::vector<double>& derivatives,
    int nfp,
    double& value,
    double& derivative
);

PsiModes build_psi_modes(int degree, int m_tor);

void evaluate_psi_host(
    const PsiData& psi,
    const AxisData& axis,
    const SgpuScoreConfig& config,
    int nfp,
    double R,
    double Z,
    double phi,
    double& value,
    double& grad_R,
    double& grad_Z,
    double& grad_phi
);

bool fit_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi,
    double& point_generation_time,
    double& fit_time
);

bool validate_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi
);

bool upload_psi_data(
    const PsiData& psi,
    const AxisData& axis,
    DevicePsiData& device
);


}  // namespace coil_eval::pipeline
