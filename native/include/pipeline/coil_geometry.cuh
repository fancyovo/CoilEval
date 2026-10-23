#pragma once

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct CoilMetrics {
    double length_mean = 0.0;
    double curvature_p95 = 0.0;
    double curvature_max = 0.0;
    double min_intercoil_distance = std::numeric_limits<double>::infinity();
    double min_axis_distance = std::numeric_limits<double>::infinity();
    double high_mode_fraction = 0.0;
    double current_abs_max = 0.0;
};

struct CoilComponentGradient {
    double value = 0.0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> current;
};

struct CoilMetricDerivativeOutput {
    double length_mean = 0.0;
    double curvature_p95 = 0.0;
    double curvature_max = 0.0;
    double min_intercoil_distance = 0.0;
    double min_axis_distance = 0.0;
    double high_mode_fraction = 0.0;
    double current_abs_max = 0.0;
    int spacing_pair[2] = {-1, -1};
    std::vector<double> d_length[3];
    std::vector<double> d_p95[3];
    std::vector<double> d_max[3];
    std::vector<double> d_spacing[3];
    std::vector<double> d_radius[3];
    std::vector<double> d_high_mode[3];
};

void eval_fourier_interleaved(
    const double* coeff,
    int n_coeff,
    double t,
    double& value,
    double& derivative,
    double& second_derivative
);

Vec3d rotate_z(Vec3d point, double angle);

CoilMetrics compute_coil_metrics(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp
);

double coil_component(const CoilMetrics& metrics);

double vec_dot(Vec3d left, Vec3d right);

void fourier_basis(int n_coeff, double t, int coefficient, double& value, double& first, double& second);

Vec3d transformed_basis_vector(
    int coordinate,
    double basis,
    int reflected,
    int period,
    int nfp
);

double q_down_derivative(double value, double scale, double power);

bool compute_coil_component_gradient_impl(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    CoilComponentGradient& output,
    std::string& error,
    CoilMetricDerivativeOutput* metrics_out = nullptr
);


}  // namespace coil_eval::pipeline
