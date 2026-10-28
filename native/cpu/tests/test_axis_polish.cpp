#include "../axis_polish.h"
#include <cassert>
#include <cmath>

extern "C" void ce_cpu_axis_polish_info(double*,int);
namespace coil_eval::pipeline {
int trace_calls=0;
bool reject_topology=false,singular_map=false;
bool trace_map(void*,const std::vector<double>& R,const std::vector<double>& Z,
    int,int,bool fp64,std::vector<double>& Re,std::vector<double>& Ze){
    assert(fp64);++trace_calls;Re.resize(R.size());Ze.resize(Z.size());
    for(size_t i=0;i<R.size();++i){
        if(singular_map){Re[i]=R[i]+2e-7;Ze[i]=Z[i];}
        else{Re[i]=.8+.5*(R[i]-.8)-.8*Z[i];Ze[i]=.8*(R[i]-.8)+.5*Z[i];}
    }
    return true;
}
void classify_axis_topology(void*,std::vector<AxisCandidate>& values,const AxisDomain&,
    int,const SgpuScoreConfig&){
    for(auto& c:values){c.elliptic=!reject_topology;c.topology_trace=1.;c.topology_det=.89;c.ellipse_aspect=1.;}
}
}
int main(){
    using namespace coil_eval::pipeline;
    SgpuScoreConfig config{};config.axis_tolerance=1e-7;config.axis_fd_absolute=2e-6;
    config.axis_fd_relative=2e-4;config.axis_trace_steps=240;
    AxisDomain domain{.3,1.3,-.5,.5};double timing[SGPU_SCORE_TIMING_COUNT]={},info[12]={};
    AxisCandidate a;a.R=.8+1.3e-7;a.Z=0.;a.residual=std::hypot(.5*1.3e-7,.8*1.3e-7);a.elliptic=true;
    std::vector<AxisCandidate> values{a};
    assert(cpu_polish_rejected_axis_candidates(nullptr,values,domain,4,config,timing));
    assert(values[0].residual<1e-14&&std::abs(values[0].R-.8)<1e-14);
    ce_cpu_axis_polish_info(info,0);assert(info[0]==1&&info[1]==1&&info[2]==1);
    // Any originally accepted root prevents changes to every candidate.
    AxisCandidate accepted=a;accepted.residual=5e-8;values={a,accepted};trace_calls=0;
    assert(cpu_polish_rejected_axis_candidates(nullptr,values,domain,4,config,timing));
    assert(trace_calls==0&&values[0].R==a.R&&values[1].R==accepted.R);
    // Far and nonelliptic candidates never trigger a speculative refinement.
    AxisCandidate far=a;far.residual=1e-3;AxisCandidate hyper=a;hyper.elliptic=false;values={far,hyper};
    assert(cpu_polish_rejected_axis_candidates(nullptr,values,domain,4,config,timing));assert(trace_calls==0);
    // Recomputed topology and nonsingular local Jacobian are mandatory.
    reject_topology=true;values={a};assert(cpu_polish_rejected_axis_candidates(nullptr,values,domain,4,config,timing));
    assert(values[0].R==a.R);ce_cpu_axis_polish_info(info,0);assert(info[9]==1&&info[1]==0);
    reject_topology=false;singular_map=true;values={a};
    assert(cpu_polish_rejected_axis_candidates(nullptr,values,domain,4,config,timing));
    assert(values[0].R==a.R);ce_cpu_axis_polish_info(info,0);assert(info[5]==1&&info[1]==0);
}
