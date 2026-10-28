// Compile the same candidate ranking, Newton, fallback and topology policy.
#include "axis_polish.h"
#define COIL_EVAL_CPU_AXIS_FINAL_POLISH
#include "../src/pipeline/axis.cu"

// Diagnostic entry point: expose grid candidates after the unmodified search
// and verification policy, including those rejected by the final tolerance.
extern "C" int ce_cpu_axis_candidates(void* field,const double* x,const double* y,
    const double* z,int nbase,int ncoeff,int nfp,const SgpuScoreConfig* config,
    int grid,int fallback,double* output,int* count,double* bounds,double* timings) {
    using namespace coil_eval::pipeline;
    if(!field||!config||grid<4||!output||!count||!bounds||!timings)return 1;
    const AxisDomain domain=build_axis_domain(x,y,z,nbase,ncoeff,*config);
    bounds[0]=domain.r_min;bounds[1]=domain.r_max;bounds[2]=domain.z_min;bounds[3]=domain.z_max;
    std::vector<AxisCandidate> candidates;
    if(!search_axis_grid(field,domain,nfp,*config,grid,
        fallback?config->axis_fallback_max_candidates:config->axis_max_candidates,
        fallback?config->axis_fallback_newton_iters:config->axis_newton_iters,
        candidates,fallback!=0,timings,nullptr))return 1;
    std::vector<double> R,Z,Re,Ze;
    for(const auto& candidate:candidates){R.push_back(candidate.R);Z.push_back(candidate.Z);}
    if(!candidates.empty()&&!trace_map(field,R,Z,nfp,config->axis_trace_steps,false,Re,Ze))return 1;
    *count=static_cast<int>(candidates.size());
    for(int i=0;i<*count;++i){
        const auto& c=candidates[i];double* row=output+10*i;
        row[0]=c.R;row[1]=c.Z;row[2]=c.residual;row[3]=c.topology_trace;
        row[4]=c.topology_det;row[5]=c.ellipse_aspect;row[6]=c.elliptic?1:0;
        row[7]=c.winding?1:0;row[8]=axis_topology_stability_margin(c);
        row[9]=std::hypot(Re[i]-c.R,Ze[i]-c.Z);
    }
    return 0;
}
