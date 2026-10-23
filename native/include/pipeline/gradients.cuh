#pragma once
#include "pipeline/alpha.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/volume.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct FixedFrontG2Cache {
    AxisData axis;
    PsiData psi;
    DeviceVolumePoints points;
    DeviceBuffer<float> B;
    DeviceBuffer<float> grad_B;
    AlphaFitNative alpha;
    std::array<double, 9> qs_sums{};
    double G = 0.0;
    void* field = nullptr;
    bool ready = false;
};

struct FixedFrontG2Gradient {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> current;
    double point_vjp_s = 0.0;
    double field_vjp_s = 0.0;
    double parameter_map_s = 0.0;
};

bool compute_fixed_front_g2_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
);

bool evaluate_fixed_front_g2_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    std::string& error
);

bool evaluate_fixed_front_g3_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coordinate_component,
    double& iota_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    double& iota_minimum,
    double& iota_maximum,
    std::string& error
);

bool evaluate_fixed_branch_g4_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    SgpuScoreResult& query,
    std::string& error
);

bool compute_alpha_iota_g3_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
);


}  // namespace coil_eval::pipeline
