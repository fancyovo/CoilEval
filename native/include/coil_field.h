#pragma once

#include <cstddef>
#include <cstdint>

#define SGPU_SCORE_ABI_VERSION 12u
#define SGPU_SCORE_MAX_SURFACE_LEVELS 16
#define SGPU_SCORE_COMPONENT_COUNT 7
#define SGPU_SCORE_TIMING_COUNT 32
#define SGPU_SCORE_GRADIENT_ABI_VERSION 1u

enum SgpuScoreStatus {
    SGPU_SCORE_OK = 0,
    SGPU_SCORE_NO_AXIS = 1,
    SGPU_SCORE_NO_SURFACE = 2,
    SGPU_SCORE_DRIFT_REJECTED = 3,
    SGPU_SCORE_FLUX_REJECTED = 4,
    SGPU_SCORE_ALPHA_FAILED = 5,
    SGPU_SCORE_BRANCH_LOST = 6,
    SGPU_SCORE_INTERNAL_ERROR = 100,
};

enum SgpuScoreComponent {
    SGPU_SCORE_COMPONENT_AXIS = 0,
    SGPU_SCORE_COMPONENT_PSI = 1,
    SGPU_SCORE_COMPONENT_SURFACE = 2,
    SGPU_SCORE_COMPONENT_COORDINATE = 3,
    SGPU_SCORE_COMPONENT_VOLUME_QS = 4,
    SGPU_SCORE_COMPONENT_IOTA = 5,
    SGPU_SCORE_COMPONENT_COIL = 6,
};

enum SgpuScoreTiming {
    SGPU_SCORE_TIME_TOTAL = 0,
    SGPU_SCORE_TIME_FIELD_CREATE = 1,
    SGPU_SCORE_TIME_COIL_GEOMETRY = 2,
    SGPU_SCORE_TIME_AXIS_SEARCH = 3,
    SGPU_SCORE_TIME_AXIS_TRACE = 4,
    SGPU_SCORE_TIME_PSI_POINTS = 5,
    SGPU_SCORE_TIME_PSI_FIT = 6,
    SGPU_SCORE_TIME_PSI_VALIDATE = 7,
    SGPU_SCORE_TIME_SURFACE_SCREEN = 8,
    SGPU_SCORE_TIME_FLUX = 9,
    SGPU_SCORE_TIME_VOLUME_POINTS = 10,
    SGPU_SCORE_TIME_FIELD_VOLUME = 11,
    SGPU_SCORE_TIME_ALPHA_ASSEMBLE = 12,
    SGPU_SCORE_TIME_ALPHA_QR = 13,
    SGPU_SCORE_TIME_QS_METRICS = 14,
    SGPU_SCORE_TIME_SCORE = 15,
    SGPU_SCORE_TIME_AXIS_DOMAIN = 16,
    SGPU_SCORE_TIME_AXIS_PRIMARY_GRID_TRACE = 17,
    SGPU_SCORE_TIME_AXIS_FALLBACK_GRID_TRACE = 18,
    SGPU_SCORE_TIME_AXIS_CANDIDATE_EXTRACT = 19,
    SGPU_SCORE_TIME_AXIS_CANDIDATE_REFINE = 20,
    SGPU_SCORE_TIME_AXIS_FP64_VERIFY = 21,
    SGPU_SCORE_TIME_AXIS_TOPOLOGY = 22,
    SGPU_SCORE_TIME_SURFACE_RAY_ROOTS = 23,
    SGPU_SCORE_TIME_SURFACE_MIXED_TRACE = 24,
    SGPU_SCORE_TIME_SURFACE_MIXED_REDUCE = 25,
    SGPU_SCORE_TIME_SURFACE_FP64_TRACE = 26,
    SGPU_SCORE_TIME_SURFACE_FP64_REDUCE = 27,
    SGPU_SCORE_TIME_SURFACE_LONG_TRACE = 28,
    SGPU_SCORE_TIME_SURFACE_LONG_REDUCE = 29,
    SGPU_SCORE_TIME_FLUX_CALIBRATION = 30,
    SGPU_SCORE_TIME_SURFACE_CONFIDENCE = 31,
};

struct SgpuScoreConfig {
    std::uint32_t abi_version;
    std::uint32_t struct_size;

    std::int32_t device_id;
    std::int32_t segments_per_coil;
    std::int32_t target_M;
    std::int32_t target_N;

    std::int32_t axis_grid;
    std::int32_t axis_fallback_grid;
    std::int32_t axis_max_candidates;
    std::int32_t axis_fallback_max_candidates;
    std::int32_t axis_newton_iters;
    std::int32_t axis_fallback_newton_iters;
    std::int32_t axis_trace_steps;
    std::int32_t axis_sample_count;
    std::int32_t axis_fallback_max_nfp;
    double axis_span;
    double axis_tolerance;
    double axis_r_floor;
    double axis_fd_relative;
    double axis_fd_absolute;
    double axis_topology_margin;

    std::int32_t psi_poly_degree;
    std::int32_t psi_m_tor;
    std::int32_t psi_n_r;
    std::int32_t psi_n_z;
    std::int32_t psi_n_phi;
    std::int32_t psi_validation_points;
    std::int32_t psi_solver_mode;
    std::int32_t psi_precision_mode;
    double psi_a;
    double psi_rho_min;
    double psi_ridge;

    std::int32_t surface_level_count;
    std::int32_t surface_theta_count;
    std::int32_t surface_trace_steps;
    std::int32_t surface_newton_iters;
    double surface_levels[SGPU_SCORE_MAX_SURFACE_LEVELS];
    double surface_newton_tolerance;
    double surface_max_radius_scale;
    double surface_drift_relative_tolerance;
    double surface_drift_absolute_tolerance;

    std::int32_t flux_level_count;
    std::int32_t flux_phi_count;
    std::int32_t flux_theta_count;
    std::int32_t flux_radial_quadrature;
    std::int32_t flux_polynomial_degree;
    double flux_boundary_tolerance;
    double flux_section_relative_std_tolerance;

    std::int32_t volume_point_count;
    std::int32_t volume_phi_count;
    std::int32_t volume_theta_count;
    std::int32_t alpha_fit_point_count;
    std::int32_t alpha_radial_order;
    std::int32_t alpha_poloidal_order;
    std::int32_t alpha_toroidal_order;
    std::int32_t iota_degree;
    std::int32_t radial_bin_count;
    std::int32_t alpha_solver_mode;
    double volume_rho_min;
    double alpha_ridge;

    double score_weights[SGPU_SCORE_COMPONENT_COUNT];
    double score_axis_residual_scale;
    double score_psi_angle_p95_scale;
    double score_psi_angle_l2_scale;
    double score_surface_inverse_aspect_saturation;
    double score_surface_drift_scale;
    double score_flux_section_std_scale;
    double score_flux_boundary_residual_scale;
    double score_alpha_normal_B_scale;
    double score_alpha_relative_l2_scale;
    double score_qs_global_scale;
    double score_qs_edge_scale;
    double score_qh_iota_threshold;
    double score_qh_iota_power;
    double score_volume_qs_size_floor;
    double score_volume_qs_iota_floor;
    double score_qh_total_iota_floor;
    std::int32_t surface_long_trace_periods;
    double surface_long_trace_relative_tolerance;
    double score_qh_total_helicity_floor;
    double score_qh_helicity_bad;
    double score_qh_helicity_good;
    double score_qh_helicity_exploration_fraction;

    std::int32_t surface_selection_mode;
    std::int32_t surface_confidence_periods;
    std::int32_t surface_flux_bisection_iters;
    double surface_confidence_drift_center;
    double surface_confidence_drift_temperature;
    double surface_confidence_smoothmax_temperature;
    double surface_confidence_minimum;

    std::int32_t axis_hint_enabled;
    // 0: allow grid fallback, 1: strict FP64 verification, 2: strict mixed verification.
    std::int32_t axis_hint_require_continuation;
    double axis_hint_R;
    double axis_hint_Z;
    double axis_hint_max_distance;
    // Zero disables target-volume mode and preserves the legacy numerical path.
    double target_volume;
    double target_volume_relative_tolerance;
    double target_volume_neighbor_tolerance;
    double target_volume_forward_tolerance;
    double target_quality_threshold;
    double target_quality_temperature;
    double target_quality_weight;
};

struct SgpuScoreResult {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::int32_t status;
    std::int32_t stage_completed;
    std::int32_t device_id;
    std::int32_t flux_attempt_count;

    double score;
    double components[SGPU_SCORE_COMPONENT_COUNT];
    double timings[SGPU_SCORE_TIMING_COUNT];

    double axis_R;
    double axis_Z;
    double axis_residual;
    double axis_topology_trace;
    double axis_topology_det;
    double axis_ellipse_aspect;

    double psi_train_rms;
    double psi_angle_mean;
    double psi_angle_p95;
    double psi_angle_l2;

    double surface_level;
    double surface_drift_relative_p95;
    double surface_one_period_drift_relative_p95;
    double surface_effective_minor_radius;
    double surface_inverse_aspect_ratio;
    double surface_volume;

    double flux_edge;
    double flux_fit_relative_rms;
    double flux_section_relative_std_edge;
    double flux_boundary_residual_max;
    double flux_derivative_min;
    double flux_derivative_max;

    double alpha_relative_l2;
    double alpha_normal_B_relative_l2;
    double iota_min;
    double iota_max;
    double score_surface_size;
    double score_iota;
    double score_qs_residual;
    double score_volume_qs_size_factor;
    double score_volume_qs_iota_factor;
    double score_before_qh_iota_gate;
    double score_qh_total_iota_factor;
    double score_qh_helicity_advantage;
    double score_qh_helicity_quality;
    double score_qh_total_helicity_factor;

    double qs_global_error;
    double qs_edge_error;
    double qs_qa_global_error;
    double qs_qp_global_error;
    double qs_vacuum_G;
    double qs_target_global_error_per_helicity;
    double qs_target_edge_error_per_helicity;
    double qs_qa_global_error_per_helicity;
    double qs_qp_global_error_raw;
    double qs_qp_global_error_per_helicity;
    double qs_abs_p95;
    double qs_abs_p95_per_helicity;
    double volume_valid_fraction;
    double volume_weight_effective_fraction;
    double edge_weight_effective_fraction;

    double surface_confidence_mean;
    double surface_confidence_edge;
    double surface_effective_level;
    double surface_confidence_risk;
    double axis_hint_distance;

    double coil_length_mean;
    double coil_curvature_p95;
    double coil_curvature_max;
    double coil_min_intercoil_distance;
    double coil_min_axis_distance;
    double coil_high_mode_energy_fraction;
    double coil_current_abs_max_a;

    std::int32_t axis_candidate_count;
    std::int32_t stable_surface_count;
    std::int32_t volume_candidate_count;
    std::int32_t volume_available_count;
    std::int32_t volume_point_count;
    std::int32_t alpha_column_count;
    std::int32_t surface_long_trace_periods_completed;
    std::int32_t surface_long_trace_rejected_count;
    std::int32_t axis_used_hint;
    char error_message[256];
    double target_volume_requested;
    double target_volume_relative_error;
    double target_volume_dVds;
    double target_volume_maximum;
    double target_quality_error;
    double target_quality_penalty;
    std::uint64_t target_volume_cache_id;
    std::int32_t target_volume_iterations;
};

struct SgpuScoreGradientResult {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::int32_t status;
    std::int32_t gradient_group;
    double forward_wall_s;
    double gradient_wall_s;
    double point_vjp_s;
    double field_vjp_s;
    double parameter_map_s;
    double score_gradient_rms;
    double coil_component_gradient_rms;
    char error_message[256];
};

extern "C" {

int sgpu_default_score_config(SgpuScoreConfig* config);

std::size_t sgpu_score_config_size();
std::size_t sgpu_score_result_size();

int sgpu_score_coils(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* result
);

std::size_t sgpu_score_gradient_result_size();

// Experimental opt-in G1 path. The production sgpu_score_coils ABI and path
// remain independent and do not allocate or evaluate gradients.
int sgpu_score_coils_g1_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
);

// Experimental cumulative G1+G2 path. G2 freezes the selected axis, psi,
// volume points, weights, fitted iota, and all discrete branch choices.
// A valid non-OK score returns code 0 with the score preserved, zero gradient
// arrays, and a nonzero gradient_result status so callers can backtrack.
int sgpu_score_coils_g2_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
);

// Experimental cumulative G1+G2+G3 path. G3 additionally differentiates the
// alpha/iota ridge-QR fit while keeping axis, psi, points, and branches fixed.
// The geometry-covariant normal-field residual is deferred until psi/surface
// motion can be differentiated as one bundle.
int sgpu_score_coils_g3_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
);

// Diagnostic-only closure oracle for G2. The center forward pass supplies the
// frozen volume points, flux geometry, weights, and fitted iota. Each query
// recomputes only the coil component and B/grad(B)/G on that frozen front, then
// rebuilds the exact scalar whose center derivative is returned by G1+G2.
// Query coefficient arrays are flattened as [query][coil][coefficient].
int sgpu_score_coils_g2_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors
);

// Diagnostic-only closure oracle for cumulative G1+G2+G3. Geometry, psi,
// volume points, and coordinate weights stay frozen at the center, while each
// query refits alpha/iota by the production QR path. The coordinate normal-B
// term remains frozen because its compensating geometry derivative belongs to
// G4. Query coefficient arrays are flattened as [query][coil][coefficient].
int sgpu_score_coils_g3_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coordinate_components,
    double* iota_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors,
    double* iota_minima,
    double* iota_maxima
);

// Diagnostic fixed-branch G4 oracle. The center selects the magnetic axis and
// surface level once. Every query then refits psi, recalibrates flux, rebuilds
// volume points/weights, refits alpha/iota, and recomputes QS without rerunning
// axis search or surface tracing. Query results expose branch changes normally.
int sgpu_score_coils_g4_fixed_branch_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    SgpuScoreResult* query_score_results
);

// Experimental local-oracle path: every query recomputes the complete score,
// but initializes its full-density psi solve from the center QR coefficients.
int sgpu_score_coils_psi_warm_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int warm_iterations,
    int use_center_qr_preconditioner,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    SgpuScoreResult* query_score_results
);

// Captures the center psi QR factor for the query-major local oracle and
// returns the fitted center coefficients. The capture remains valid until
// sgpu_clear_psi_warm_preconditioner is called on the same host thread.
int sgpu_get_boundary_iota(int query, double* value);
int sgpu_get_captured_iota_profile(double* coefficients, int capacity, int* count);

int sgpu_score_coils_capture_psi_center(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* result,
    double* psi_coefficients,
    int psi_coefficient_capacity,
    int* psi_coefficient_count
);

// Internal-oracle entrypoint: returns the reported 0--100 coil component and
// its piecewise analytical gradient with active percentile/minimum indices
// frozen at the supplied coil.
int sgpu_coil_component_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    double* component_value,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current
);

// Internal-oracle entrypoint: returns the same coil metrics that feed the coil
// component together with the piecewise analytical derivatives used to build
// the blended gradient above. Exposing the separated quantities lets callers
// linearize an additional metric penalty exactly the way the reported coil
// component already linearizes its own spacing/curvature terms.
//
// metric_values layout, 7 doubles:
//   0 length_mean, 1 curvature_p95, 2 curvature_max, 3 min_intercoil_distance,
//   4 min_axis_distance, 5 high_mode_fraction, 6 current_abs_max
// metric_derivatives layout, 6 * 3 * n_base_coils * n_coeff doubles:
//   index ((metric * 3 + coordinate) * n_base_coils + coil) * n_coeff + coefficient
//   metric order = length_mean, curvature_p95, curvature_max,
//                  min_intercoil_distance, min_axis_distance, high_mode_fraction
//   coordinate order = x, y, z
// spacing_pair_indices, optional 2 ints: base-coil indices that realize the
//   minimum intercoil distance, or -1 when undefined.
// Optional torsion extension; gradient layout (coordinate, coil, coefficient).
// Mean-torsion interface is historical; pass a positive fixed sample count.
// Null gradient requests values only. Absolute value uses zero subgradient at zero.
// P95 uses256 samples, arc-length weights, pooling all base coils.
int sgpu_coil_torsion_p95_gradient(const double* x,const double* y,const double* z,
 int n_coils,int n_coeff,double* value,double* gradient);
int sgpu_coil_torsion_gradient(const double* x, const double* y, const double* z,
    int n_coils, int n_coeff, int samples, double* value, double* gradient);

int sgpu_coil_metric_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    double* metric_values,
    double* metric_derivatives,
    int* spacing_pair_indices
);

int sgpu_create_field(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
);

// Experimental query-major FP32 field used by the local full-gradient oracle.
// Coefficients are flattened as [query][coil][coefficient]. All point and line
// arrays passed to the batch operations are flattened as [query][item][...].
int sgpu_create_field_batch_f32(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
);

void sgpu_destroy_field_batch(void* handle);

int sgpu_batch_eval_B_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    int points_per_query
);

int sgpu_batch_eval_B_grad_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    float* grad_B_host,
    int points_per_query
);

int sgpu_batch_trace_period_mixed(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int lines_per_query,
    int nfp,
    int steps
);

int sgpu_batch_trace_axis_samples(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    int nfp,
    int integration_steps,
    int sample_count,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
);

int sgpu_batch_refine_axis_hint(
    void* handle,
    const double* hint_R_host,
    const double* hint_Z_host,
    int nfp,
    int trace_steps,
    int newton_iterations,
    double finite_difference_step,
    double maximum_newton_step,
    double residual_tolerance,
    double hint_max_distance,
    double* axis_R_host,
    double* axis_Z_host,
    double* residual_host,
    double* topology_trace_host,
    double* topology_det_host,
    unsigned char* valid_host
);

void sgpu_destroy_field(void* handle);

int sgpu_segment_count(void* handle);

int sgpu_eval_B(
    void* handle,
    const double* xyz_host,
    double* B_host,
    int n_points
);

int sgpu_eval_B_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    int n_points
);

int sgpu_eval_B_grad(
    void* handle,
    const double* xyz_host,
    double* B_host,
    double* grad_B_host,
    int n_points
);

int sgpu_eval_B_grad_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    float* grad_B_host,
    int n_points
);

// Experimental point-coordinate VJP for the FP32 B/grad(B) evaluator. This
// contracts the spatial Hessian internally and never materializes it.
int sgpu_eval_B_grad_point_vjp_f32(
    void* handle,
    const float* xyz_host,
    const float* adj_B_host,
    const float* adj_grad_B_host,
    float* adj_xyz_host,
    int n_points
);

int sgpu_normal_eq(
    void* handle,
    const double* mat_host,
    const double* rhs_host,
    double* ata_host,
    double* atb_host,
    int n_rows,
    int n_cols
);

int sgpu_normal_eq_f32(
    void* handle,
    const float* mat_host,
    const float* rhs_host,
    float* ata_host,
    float* atb_host,
    int n_rows,
    int n_cols
);

int sgpu_set_psi_warm_preconditioner_capture(int enabled);
int sgpu_has_psi_warm_preconditioner(int coefficient_count);
void sgpu_clear_psi_warm_preconditioner();

int sgpu_fit_psi_batch_pcgls_f32(
    void* batch_field_handle,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int axis_count,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int coefficient_count,
    int nfp,
    double radius_scale,
    int radial_grid,
    int vertical_grid,
    int phi_grid,
    double rho_min,
    double ridge,
    int iterations,
    const double* center_coefficients_host,
    double* coefficients_host,
    double* train_rms_host,
    double* stats_out,
    int stats_len
);

// Experimental local full-gradient oracle. The selected center surface level
// and coordinate component are fixed, while each query retains its own axis,
// psi, flux calibration, volume geometry, iota and differential-QS response.
int sgpu_score_coils_local_batch(
    void* batch_field_handle,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const double* axis_R,
    const double* axis_Z,
    const double* axis_R_phi,
    const double* axis_Z_phi,
    int axis_count,
    const double* axis_residual,
    const double* axis_topology_trace,
    const double* axis_topology_det,
    const double* psi_coefficients,
    int psi_coefficient_count,
    const double* psi_train_rms,
    const double* coil_components,
    const SgpuScoreResult* center_result,
    const SgpuScoreConfig* config,
    int surface_theta_count,
    int alpha_iterations,
    SgpuScoreResult* query_results,
    double* stats_out,
    int stats_len
);

int sgpu_fit_psi_fullgpu(
    void* handle,
    const double* R_host,
    const double* Z_host,
    const double* phi_host,
    int n_points,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int n_axis,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double ridge,
    int solver_mode,
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
    double* stats_out,
    int stats_len
);

// Positive data-row weights with mean one; RMS uses sum(w*r*r)/n_points.
int sgpu_fit_psi_fullgpu_weighted(
    void* handle,
    const double* R_host,
    const double* Z_host,
    const double* phi_host,
    int n_points,
    const double* row_weights_host,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int n_axis,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double ridge,
    int solver_mode,
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
    double* stats_out,
    int stats_len
);

int sgpu_surface_points_from_level(
    const double* coeff_host,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    const double* axis_R_host,
    const double* axis_Z_host,
    int n_axis,
    int order,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    int device_id,
    double* xyz_host,
    double* radii_host,
    double* stats_out,
    int stats_len
);

int sgpu_trace_period(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps
);

int sgpu_trace_period_blockline(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps,
    int threads_per_line
);

int sgpu_trace_period_blockline_mixed(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps,
    int threads_per_line,
    int mode
);

int sgpu_trace_axis_samples(
    void* handle,
    double R0,
    double Z0,
    int nfp,
    int integration_steps,
    int n_samples,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
);

const char* sgpu_last_error();

int sgpu_copy_psi_warm_preconditioner(double* upper, double* scales, int count);

}




