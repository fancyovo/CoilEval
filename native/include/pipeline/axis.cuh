#pragma once

#include "pipeline/types.cuh"
#include "pipeline/axis_coarse_cache.cuh"

namespace coil_eval::pipeline {

struct AxisDomain {
    double r_min, r_max, z_min, z_max;
};

struct AxisCandidate {
    double R = 0.0;
    double Z = 0.0;
    double residual = std::numeric_limits<double>::infinity();
    bool winding = false;
    double topology_trace = std::numeric_limits<double>::quiet_NaN();
    double topology_det = std::numeric_limits<double>::quiet_NaN();
    double ellipse_aspect = std::numeric_limits<double>::infinity();
    bool elliptic = false;
};

struct AxisData {
    AxisCandidate selected;
    std::vector<double> R;
    std::vector<double> Z;
    std::vector<double> R_phi;
    std::vector<double> Z_phi;
    int candidate_count = 0;
    bool used_hint = false;
    bool branch_lost = false;
    double hint_distance = std::numeric_limits<double>::quiet_NaN();
};

AxisDomain build_axis_domain(const double* x, const double* y, const double* z,
    int nbase, int ncoeff, const SgpuScoreConfig& config);
std::vector<AxisCandidate> find_grid_candidates(const std::vector<double>& rs,
    const std::vector<double>& zs, const std::vector<double>& dR,
    const std::vector<double>& dZ, int max_candidates);
bool refine_axis_candidates(void* field, std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain, int nfp, const SgpuScoreConfig& config, int iterations);

bool trace_map(
    void* field,
    const std::vector<double>& R,
    const std::vector<double>& Z,
    int nfp,
    int steps,
    bool verify_fp64,
    std::vector<double>& R_end,
    std::vector<double>& Z_end
);

double axis_topology_stability_margin(const AxisCandidate& candidate);

bool magnetic_axis_circulation_native(
    void* field,
    const AxisData& axis,
    int nfp,
    double& circulation
);

bool vacuum_G_native(
    void* field,
    const AxisData& axis,
    int nfp,
    double edge_toroidal_flux,
    double& G
);

bool find_axis_native(
    void* field,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    AxisData& axis,
    double& trace_time,
    double* timings,
    sgpu_axis_coarse::Entry* coarse_cache = nullptr
);


}  // namespace coil_eval::pipeline
