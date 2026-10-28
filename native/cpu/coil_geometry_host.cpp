#include "../src/pipeline/coil_geometry.cu"

extern "C" void sgpu_internal_set_error(const char*);

// The CPU ABI exposes the same active-branch analytical Jacobians as CUDA's
// host geometry adapter; compute_coil_component_gradient_impl is shared.
extern "C" int ce_cpu_coil_metric_gradient(
    const double* coeffs_x,const double* coeffs_y,const double* coeffs_z,
    const double* currents_a,int n_base_coils,int n_coeff,int nfp,
    double* metric_values,double* metric_derivatives,int* spacing_pair_indices) {
    using namespace coil_eval::pipeline;
    if(!metric_values||!metric_derivatives) {
        sgpu_internal_set_error("coil metric gradient output pointer is null");return 1;
    }
    CoilComponentGradient gradient;CoilMetricDerivativeOutput metrics;std::string error;
    if(!compute_coil_component_gradient_impl(coeffs_x,coeffs_y,coeffs_z,currents_a,
            n_base_coils,n_coeff,nfp,gradient,error,&metrics)) {
        sgpu_internal_set_error(error.c_str());return 1;
    }
    metric_values[0]=metrics.length_mean;
    metric_values[1]=metrics.curvature_p95;
    metric_values[2]=metrics.curvature_max;
    metric_values[3]=metrics.min_intercoil_distance;
    metric_values[4]=metrics.min_axis_distance;
    metric_values[5]=metrics.high_mode_fraction;
    metric_values[6]=metrics.current_abs_max;
    const std::vector<double>* blocks[6][3]={
        {&metrics.d_length[0],&metrics.d_length[1],&metrics.d_length[2]},
        {&metrics.d_p95[0],&metrics.d_p95[1],&metrics.d_p95[2]},
        {&metrics.d_max[0],&metrics.d_max[1],&metrics.d_max[2]},
        {&metrics.d_spacing[0],&metrics.d_spacing[1],&metrics.d_spacing[2]},
        {&metrics.d_radius[0],&metrics.d_radius[1],&metrics.d_radius[2]},
        {&metrics.d_high_mode[0],&metrics.d_high_mode[1],&metrics.d_high_mode[2]},
    };
    const int count=n_base_coils*n_coeff;
    for(int metric=0;metric<6;++metric)for(int coordinate=0;coordinate<3;++coordinate) {
        const std::vector<double>& source=*blocks[metric][coordinate];
        if(static_cast<int>(source.size())!=count) {
            sgpu_internal_set_error("coil metric derivative block size mismatch");return 1;
        }
        std::copy(source.begin(),source.end(),metric_derivatives+
            static_cast<std::size_t>(metric*3+coordinate)*count);
    }
    if(spacing_pair_indices) {
        spacing_pair_indices[0]=metrics.spacing_pair[0];
        spacing_pair_indices[1]=metrics.spacing_pair[1];
    }
    sgpu_internal_set_error("");return 0;
}
