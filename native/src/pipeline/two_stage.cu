// Two-stage scalar fit context and complete formal downstream continuation.
#include "pipeline/types.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "pipeline/evaluate.cuh"
#include "pipeline/policy.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"
#include "pipeline/target_volume.cuh"
#include "pipeline/reverse_capture.cuh"
#include "coil_field_internal.h"
#include <cstdio>
#include <memory>
using namespace coil_eval::pipeline;
namespace {
struct PsiFactorCapture {
    PsiFactorCapture() { sgpu_set_psi_warm_preconditioner_capture(source_capture_only() ? 0 : 1); }
    ~PsiFactorCapture() { sgpu_set_psi_warm_preconditioner_capture(0); }
};
struct AlphaWeightingOverride {
    AlphaRadialWeightingOptions previous;
    explicit AlphaWeightingOverride(const AlphaRadialWeightingOptions& options)
        : previous(g_alpha_radial_weighting) { g_alpha_radial_weighting=options; }
    ~AlphaWeightingOverride() { g_alpha_radial_weighting=previous; }
};
struct AlphaBasisOverride {
    int previous;
    explicit AlphaBasisOverride(int basis) : previous(g_alpha_radial_basis) { g_alpha_radial_basis=basis; }
    ~AlphaBasisOverride() { g_alpha_radial_basis=previous; }
};
bool two_stage_alpha_capture(const FixedFrontG2Cache& cache, const SgpuScoreConfig& config,
                             const SgpuScoreResult& result) {
    const auto& a=cache.alpha;
    if (!a.adjoint_ready) return false;
    auto& warm=g_alpha_warm_preconditioner;
    warm.radial_basis=a.modes.radial_basis;
    warm.column_count=a.column_count; warm.qr_rows=a.qr_rows;
    warm.upper_factor.resize(static_cast<size_t>(a.column_count)*a.column_count);
    warm.center_scale.resize(a.column_count);warm.center_scaled_solution.resize(a.column_count);
    return cudaMemcpy2D(warm.upper_factor.data(),static_cast<size_t>(a.column_count)*sizeof(float),
        a.qr_matrix.data(),static_cast<size_t>(a.qr_rows)*sizeof(float),
        static_cast<size_t>(a.column_count)*sizeof(float),a.column_count,cudaMemcpyDeviceToHost)==cudaSuccess &&
        cudaMemcpy(warm.center_scale.data(),a.scales.data(),a.column_count*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
        cudaMemcpy(warm.center_scaled_solution.data(),a.scaled_solution.data(),a.column_count*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
        capture_alpha_volume_sampling(cache.points, config, result.target_volume_cache_id);
}
} // namespace
namespace {
struct TwoStageNativeContextV1 {
    void* field = nullptr;
    AxisData axis;
    PsiData first;
    SgpuScoreConfig config{};
    AlphaRadialWeightingOptions alpha_weighting;
    int alpha_radial_basis = 0;
    std::vector<double> x, y, z, current;
    int coils = 0, coefficients = 0, nfp = 0;
    ~TwoStageNativeContextV1() { if (field) sgpu_destroy_field(field); }
};

void two_stage_axis_result(const AxisData& axis, SgpuScoreResult& out) {
    out.axis_candidate_count=axis.candidate_count;
    out.axis_used_hint=axis.used_hint ? 1 : 0;
    out.axis_hint_distance=axis.hint_distance;
    if (axis.R.empty()) return;
    out.axis_R=axis.selected.R; out.axis_Z=axis.selected.Z;
    out.axis_residual=axis.selected.residual;
    out.axis_topology_trace=axis.selected.topology_trace;
    out.axis_topology_det=axis.selected.topology_det;
    out.axis_ellipse_aspect=axis.selected.ellipse_aspect;
}

bool two_stage_snapshot(const char* directory, const FixedFrontG2Cache& cache) {
    if (!directory || !*directory) return true;
    auto write=[&](const char* name,const float* values,size_t n) {
        const std::string path=std::string(directory)+"/"+name+".f32";
        FILE* file=std::fopen(path.c_str(),"wb");
        if (!file) return false;
        const size_t written=std::fwrite(values,sizeof(float),n,file);
        return std::fclose(file)==0 && written==n;
    };
    auto device=[&](const char* name,const float* values,size_t n) {
        std::vector<float> host(n);
        return cudaMemcpy(host.data(),values,n*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
            write(name,host.data(),n);
    };
    const size_t n=cache.points.count;
    return device("xyz",cache.points.xyz.data(),3*n) &&
        device("B",cache.B.data(),3*n) && device("grad_B",cache.grad_B.data(),9*n) &&
        device("grad_s",cache.points.grad_s.data(),3*n) &&
        device("grad_theta",cache.points.grad_theta.data(),3*n) &&
        device("grad_phi",cache.points.grad_phi.data(),3*n) &&
        device("s",cache.points.s.data(),n) && device("rho",cache.points.rho.data(),n) &&
        device("theta",cache.points.theta.data(),n) && device("phi",cache.points.phi.data(),n) &&
        device("weight",cache.points.volume_weight.data(),n) &&
        device("flux_derivative",cache.points.flux_derivative.data(),n) &&
        write("iota_coefficients",cache.alpha.iota_coefficients.data(),cache.alpha.iota_coefficients.size());
}
} // namespace

extern "C" int sgpu_two_stage_begin_v1(
    const double* x,const double* y,const double* z,const double* currents,
    int coils,int coefficients,int nfp,const SgpuScoreConfig* config,
    SgpuScoreResult* result,void** context,void** field_handle,
    double* axis_arrays,int axis_capacity,int* axis_count,
    double* psi_coefficients,int psi_capacity,int* psi_count
) {
    auto coarse_cache = g_axis_coarse_slot.take();
    if (!result) return 1;
    initialize_result(result,config ? config->device_id : -1);
    if (context) *context=nullptr;
    if (field_handle) *field_handle=nullptr;
    if (axis_count) *axis_count=0;
    if (psi_count) *psi_count=0;
    if (!x || !y || !z || !currents || !config || !context || !field_handle ||
        !axis_arrays || !axis_count || !psi_coefficients || !psi_count ||
        coils<=0 || coefficients<3 || coefficients%2==0 || nfp<=0) {
        return fail_result(result,"invalid two-stage stage1 inputs");
    }
    // Stage A deliberately has no target-volume or psi-quality evaluation.
    SgpuScoreConfig first_config=*config;
    first_config.target_volume=0;
    std::string reason;
    if (!validate_config(first_config,reason)) return fail_result(result,reason.c_str());
    if (coarse_cache.available && !coarse_cache.matches(x,y,z,currents,coils,coefficients,nfp,&first_config)) {
        g_axis_coarse_slot.status=3;
        if (coarse_cache.refinement_available) g_axis_coarse_slot.refinement_status=3;
        return fail_result(result,"one-shot axis coarse cache physical/config key mismatch");
    }
    if (g_active_psi_warm_coefficients || g_active_g2_cache || g_active_gradient_group)
        return fail_result(result,"two-stage entry must not run inside a gradient/warm context");
    if (cudaSetDevice(config->device_id)!=cudaSuccess) return fail_result(result,"cudaSetDevice failed");
    const auto total=Clock::now();
    auto* ctx=new TwoStageNativeContextV1;
    ctx->config=first_config; ctx->coils=coils; ctx->coefficients=coefficients; ctx->nfp=nfp;
    const size_t count=static_cast<size_t>(coils)*coefficients;
    ctx->x.assign(x,x+count); ctx->y.assign(y,y+count); ctx->z.assign(z,z+count);
    ctx->current.assign(currents,currents+coils);
    auto started=Clock::now();
    if (sgpu_create_field(x,y,z,currents,coils,coefficients,nfp,config->segments_per_coil,config->device_id,&ctx->field)) {
        delete ctx; return fail_from_backend(result,"two-stage field creation");
    }
    result->timings[SGPU_SCORE_TIME_FIELD_CREATE]=seconds_since(started);
    result->stage_completed=SCORE_STAGE_FIELD;
    double trace_time=0;
    started=Clock::now();
    if (!find_axis_native(ctx->field,x,y,z,coils,coefficients,nfp,first_config,ctx->axis,trace_time,result->timings,&coarse_cache)) {
        delete ctx; return fail_from_backend(result,"two-stage axis search");
    }
    result->timings[SGPU_SCORE_TIME_AXIS_TRACE]=trace_time;
    result->timings[SGPU_SCORE_TIME_AXIS_SEARCH]=std::max(0.,seconds_since(started)-trace_time);
    two_stage_axis_result(ctx->axis,*result);
    result->stage_completed=SCORE_STAGE_AXIS;
    if (ctx->axis.R.empty()) {
        result->status=ctx->axis.branch_lost ? SGPU_SCORE_BRANCH_LOST : SGPU_SCORE_NO_AXIS;
        result->timings[SGPU_SCORE_TIME_TOTAL]=seconds_since(total);
        delete ctx; return 0;
    }
    const int n=static_cast<int>(ctx->axis.R.size());
    if (axis_capacity<n) {delete ctx;return fail_result(result,"two-stage axis output capacity too small");}
    PsiFactorCapture factor_capture;
    if (!fit_psi_native(ctx->field,ctx->axis,nfp,first_config,ctx->first,
        result->timings[SGPU_SCORE_TIME_PSI_POINTS],result->timings[SGPU_SCORE_TIME_PSI_FIT])) {
        delete ctx;return fail_from_backend(result,"two-stage cylindrical psi fit");
    }
    const int p=static_cast<int>(ctx->first.coeffs.size());
    if (psi_capacity<p) {delete ctx;return fail_result(result,"two-stage psi output capacity too small");}
    for (int j=0;j<n;++j) {
        axis_arrays[j]=ctx->axis.R[j]; axis_arrays[n+j]=ctx->axis.Z[j];
        axis_arrays[2*n+j]=ctx->axis.R_phi[j]; axis_arrays[3*n+j]=ctx->axis.Z_phi[j];
    }
    std::copy(ctx->first.coeffs.begin(),ctx->first.coeffs.end(),psi_coefficients);
    *axis_count=n; *psi_count=p;
    result->psi_train_rms=ctx->first.train_rms;
    result->stage_completed=SCORE_STAGE_PSI;
    result->status=SGPU_SCORE_OK; // PARTIAL fit success, explicitly not a complete score.
    result->timings[SGPU_SCORE_TIME_TOTAL]=seconds_since(total);
    *context=ctx; *field_handle=ctx->field;
    sgpu_internal_set_error("");
    return 0;
}

extern "C" int sgpu_two_stage_resume_v1(
    void* context,const double* coefficients,int coefficient_count,double train_rms,
    const SgpuScoreConfig* config,SgpuScoreResult* result,
    double* iota_coefficients,int iota_capacity,int* iota_count,
    double* boundary_iota,const char* snapshot_directory
) {
    if (source_capture_only()) g_reverse_snapshot.clear();
    if (!result) return 1;
    initialize_result(result,config ? config->device_id : -1);
    if (iota_count) *iota_count=0;
    if (boundary_iota) *boundary_iota=std::numeric_limits<double>::quiet_NaN();
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || !coefficients || !config || !iota_coefficients || !iota_count || !boundary_iota)
        return fail_result(result,"invalid two-stage resume input");
    std::string reason;
    if (!validate_config(*config,reason)) return fail_result(result,reason.c_str());
    if (!(config->target_volume>0) || !std::isfinite(config->target_volume))
        return fail_result(result,"two-stage resume requires the requested positive target volume");
    if (config->device_id!=ctx->config.device_id || config->segments_per_coil!=ctx->config.segments_per_coil ||
        config->psi_a!=ctx->config.psi_a || config->psi_poly_degree!=ctx->config.psi_poly_degree ||
        config->psi_m_tor!=ctx->config.psi_m_tor || coefficient_count!=static_cast<int>(ctx->first.coeffs.size()) ||
        iota_capacity<config->iota_degree+1 || !std::isfinite(train_rms))
        return fail_result(result,"resume basis/physical-field configuration differs from stage1 or output capacity invalid");
    for (int j=0;j<coefficient_count;++j) if (!std::isfinite(coefficients[j]))
        return fail_result(result,"nonfinite externally fitted s2 coefficient");
    if (g_active_g2_cache || g_active_gradient_group || g_active_psi_warm_coefficients)
        return fail_result(result,"resume must not run inside gradient/warm context");
    if (cudaSetDevice(config->device_id)!=cudaSuccess) return fail_result(result,"cudaSetDevice failed");
    AlphaWeightingOverride alpha_weighting_scope(ctx->alpha_weighting);
    AlphaBasisOverride alpha_basis_scope(ctx->alpha_radial_basis);
    const auto total=Clock::now();
    g_boundary_iota=std::numeric_limits<double>::quiet_NaN();
    g_captured_iota_profile.clear();
    result->target_volume_requested=config->target_volume;
    two_stage_axis_result(ctx->axis,*result);
    result->stage_completed=SCORE_STAGE_AXIS;
    auto started=Clock::now();
    const CoilMetrics coil=compute_coil_metrics(ctx->x.data(),ctx->y.data(),ctx->z.data(),ctx->current.data(),
        ctx->coils,ctx->coefficients,ctx->nfp);
    result->timings[SGPU_SCORE_TIME_COIL_GEOMETRY]=seconds_since(started);
    result->coil_length_mean=coil.length_mean;result->coil_curvature_p95=coil.curvature_p95;
    result->coil_curvature_max=coil.curvature_max;result->coil_min_intercoil_distance=coil.min_intercoil_distance;
    result->coil_min_axis_distance=coil.min_axis_distance;result->coil_high_mode_energy_fraction=coil.high_mode_fraction;
    result->coil_current_abs_max_a=coil.current_abs_max;
    PsiData psi;
    psi.modes=build_psi_modes(config->psi_poly_degree,config->psi_m_tor);
    psi.coeffs.assign(coefficients,coefficients+coefficient_count);
    psi.train_rms=train_rms;
    std::vector<SurfaceScreen> screens;
    int code=0;
    FixedFrontG2Cache cache;
    do {
        started=Clock::now();
        if (!validate_psi_native(ctx->field,ctx->axis,ctx->nfp,*config,psi)) {
            code=fail_from_backend(result,"two-stage replacement psi validation");break;
        }
        result->timings[SGPU_SCORE_TIME_PSI_VALIDATE]=seconds_since(started);
        result->psi_train_rms=psi.train_rms;result->psi_angle_mean=psi.angle_mean;
        result->psi_angle_p95=psi.angle_p95;result->psi_angle_l2=psi.angle_l2;
        result->stage_completed=SCORE_STAGE_PSI;
        started=Clock::now();
        const bool screen_ok=screen_target_volume_native(ctx->field,ctx->axis,psi,ctx->nfp,*config,screens,*result);
        result->timings[SGPU_SCORE_TIME_SURFACE_SCREEN]=seconds_since(started);
        if (!screen_ok) {code=fail_from_backend(result,"two-stage target-volume surface screen");break;}
        result->stable_surface_count=static_cast<int>(std::count_if(screens.begin(),screens.end(),
            [](const auto& screen){return screen.stable;}));
        result->stage_completed=SCORE_STAGE_SURFACE;
        const auto best=std::max_element(screens.begin(),screens.end(),[](const auto& left,const auto& right){
            return (left.strict && left.verified ? left.level : -1.) < (right.strict && right.verified ? right.level : -1.);
        });
        if (!result->stable_surface_count) {result->status=SGPU_SCORE_NO_SURFACE;break;}
        if (best==screens.end() || !best->strict || !best->verified) {result->status=SGPU_SCORE_DRIFT_REJECTED;break;}
        // Cache only retains existing forward products after their computation;
        // gradient_group stays zero. No gradient, re-fit, or extra score is run.
        g_active_gradient_group=3;
        g_active_g2_cache=&cache;
        const bool ok=run_downstream_gpu(ctx->field,ctx->current.data(),ctx->coils,ctx->nfp,
            ctx->axis,psi,screens,*config,*result,config->surface_selection_mode==0);
        g_active_g2_cache=nullptr;
        g_active_gradient_group=0;
        cache.axis=ctx->axis;cache.psi=psi;cache.field=ctx->field;
        if (!ok) {if (result->status==SGPU_SCORE_INTERNAL_ERROR) code=1;break;}
        if (!cache.ready) {code=fail_result(result,"native downstream product capture missing");break;}
        std::copy(cache.alpha.iota_coefficients.begin(),cache.alpha.iota_coefficients.end(),iota_coefficients);
        *iota_count=static_cast<int>(cache.alpha.iota_coefficients.size());
        *boundary_iota=g_boundary_iota;
        g_captured_iota_profile=cache.alpha.iota_coefficients;
        if (g_reverse_capture_flags && ((!source_capture_only() && !two_stage_alpha_capture(cache,*config,*result)) ||
            !capture_reverse_snapshot(cache,ctx->nfp,*config,*result))) {
            code=fail_result(result,"two-stage reverse snapshot capture failed");break;
        }
        if (!two_stage_snapshot(snapshot_directory,cache)) {code=fail_result(result,"two-stage snapshot write failed");break;}
    } while(false);
    g_active_g2_cache=nullptr;
    g_active_gradient_group=0;
    fill_early_components(*config,coil,ctx->axis,psi,screens,*result);
    started=Clock::now();
    finalize_score(*config,*result);
    result->timings[SGPU_SCORE_TIME_SCORE]=seconds_since(started);
    result->timings[SGPU_SCORE_TIME_TOTAL]=seconds_since(total);
    return code;
}

extern "C" void sgpu_two_stage_close_v1(void* context) {
    delete static_cast<TwoStageNativeContextV1*>(context);
}


extern "C" int sgpu_two_stage_fit_v1(
    void* context,const double* R,const double* Z,const double* phi,int count,
    double* coefficients,double* train_rms,double* stats,int stats_count
) {
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || !R || !Z || !phi || count<1 || !coefficients || !train_rms) return 1;
    const auto& cfg=ctx->config;const auto& axis=ctx->axis;const auto& modes=ctx->first.modes;
    PsiFactorCapture factor_capture;
    return sgpu_fit_psi_fullgpu(ctx->field,R,Z,phi,count,
        axis.R.data(),axis.Z.data(),axis.R_phi.data(),axis.Z_phi.data(),static_cast<int>(axis.R.size()),
        modes.a.data(),modes.b.data(),modes.m.data(),modes.kind.data(),static_cast<int>(modes.a.size()),
        ctx->nfp,cfg.psi_a,cfg.psi_poly_degree,cfg.psi_m_tor,cfg.psi_ridge,
        cfg.psi_solver_mode,cfg.psi_precision_mode,coefficients,train_rms,stats,stats_count);
}


extern "C" int sgpu_two_stage_fit_weighted_v1(
    void* context,const double* R,const double* Z,const double* phi,int count,const double* row_weights,
    double* coefficients,double* train_rms,double* stats,int stats_count
) {
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || !R || !Z || !phi || !row_weights || count<1 || !coefficients || !train_rms) return 1;
    const auto& cfg=ctx->config;const auto& axis=ctx->axis;const auto& modes=ctx->first.modes;
    PsiFactorCapture factor_capture;
    return sgpu_fit_psi_fullgpu_weighted(ctx->field,R,Z,phi,count,row_weights,
        axis.R.data(),axis.Z.data(),axis.R_phi.data(),axis.Z_phi.data(),static_cast<int>(axis.R.size()),
        modes.a.data(),modes.b.data(),modes.m.data(),modes.kind.data(),static_cast<int>(modes.a.size()),
        ctx->nfp,cfg.psi_a,cfg.psi_poly_degree,cfg.psi_m_tor,cfg.psi_ridge,
        cfg.psi_solver_mode,cfg.psi_precision_mode,coefficients,train_rms,stats,stats_count);
}


extern "C" int sgpu_two_stage_set_alpha_radial_weighting_v1(
    void* context,int mode,double sigma,double count_floor
) {
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || (mode!=0 && mode!=1) || !std::isfinite(sigma) || sigma<=0. ||
        !std::isfinite(count_floor) || count_floor<=0.) return 1;
    ctx->alpha_weighting.mode=mode;
    ctx->alpha_weighting.sigma=sigma;
    ctx->alpha_weighting.count_floor=count_floor;
    return 0;
}

extern "C" int sgpu_two_stage_set_alpha_radial_basis_v1(void* context,int basis) {
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || (basis!=0 && basis!=1)) return 1;
    ctx->alpha_radial_basis=basis;
    return 0;
}

// S1 and its frozen training boundary have already been constructed. Change
// only the subsequent S2 objective; do not silently rebuild S1 with this ridge.
extern "C" int sgpu_two_stage_set_refit_ridge_v1(void* context,double ridge) {
    auto* ctx=static_cast<TwoStageNativeContextV1*>(context);
    if (!ctx || !std::isfinite(ridge) || ridge<0.) return 1;
    ctx->config.psi_ridge=ridge;
    return 0;
}
