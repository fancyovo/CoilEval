#pragma once
#include "pipeline/volume.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct AlphaRadialWeightingOptions {
    int mode = 0;  // 0: compatible hard histogram, 1: smooth density balance.
    double sigma = 0.05;
    double count_floor = 32.0;
};
extern thread_local AlphaRadialWeightingOptions g_alpha_radial_weighting;

struct ClebschModesNative {
    std::vector<int> l;
    std::vector<int> m;
    std::vector<int> n;
    std::vector<int> kind;
    std::vector<float> radial_coefficients;
    int radial_order = 0;
};

struct AlphaFitNative {
    std::vector<float> iota_coefficients;
    double relative_l2 = std::numeric_limits<double>::quiet_NaN();
    double normal_B_relative_l2 = std::numeric_limits<double>::quiet_NaN();
    int column_count = 0;
    int mode_count = 0;
    int fit_count = 0;
    int qr_rows = 0;
    int radial_weighting_mode = 0;
    double radial_weight_sigma = 0.05;
    double radial_count_floor = 32.0;
    std::vector<double> radial_soft_counts;
    double residual_norm = 0.0;
    double rhs_norm = 0.0;
    double weight_norm2 = 0.0;
    double normal_numerator = 0.0;
    double field_denominator = 0.0;
    ClebschModesNative modes;
    DeviceBuffer<float> qr_matrix;
    DeviceBuffer<float> matrix_reference;
    DeviceBuffer<float> rhs_reference;
    DeviceBuffer<float> scales;
    DeviceBuffer<float> scaled_solution;
    DeviceBuffer<float> residual;
    DeviceBuffer<float> b_theta;
    DeviceBuffer<float> b_phi;
    DeviceBuffer<float> weights;
    DeviceBuffer<float> rho;
    DeviceBuffer<float> theta;
    DeviceBuffer<float> phi;
    bool adjoint_ready = false;
};

struct AlphaWarmPreconditionerCache {
    int column_count = 0;
    int qr_rows = 0;
    std::vector<float> upper_factor;
    std::vector<float> center_scale;
    std::vector<float> center_scaled_solution;
};

ClebschModesNative build_clebsch_modes_native(
    int radial_order,
    int poloidal_order,
    int toroidal_order
);

bool fit_alpha_native(
    const DeviceVolumePoints& points,
    const float* d_B,
    int nfp,
    const SgpuScoreConfig& config,
    AlphaFitNative& fit,
    SgpuScoreResult& result
);


}  // namespace coil_eval::pipeline
