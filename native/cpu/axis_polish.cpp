// Recover roots limited by FP32 refinement noise without relaxing acceptance.
#include "axis_polish.h"
#include <array>
#include <chrono>

namespace coil_eval::pipeline {
void classify_axis_topology(void*,std::vector<AxisCandidate>&,const AxisDomain&,
    int,const SgpuScoreConfig&);
namespace {
thread_local std::array<double,12> stats{};
using Timer=std::chrono::steady_clock;
double elapsed(Timer::time_point start){return std::chrono::duration<double>(Timer::now()-start).count();}
}
bool cpu_polish_rejected_axis_candidates(void* field,std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain,int nfp,const SgpuScoreConfig& config,double* timings) {
    stats.fill(0.);
    // Existing accepted roots, ordering and all their values remain unchanged.
    for(const auto& c:candidates)if(c.elliptic&&c.residual<=config.axis_tolerance)return true;
    const auto started=Timer::now();
    const double span=std::max(domain.r_max-domain.r_min,domain.z_max-domain.z_min);
    const double limit=std::max(10*config.axis_tolerance,1e-6*span);
    const double h=std::max(config.axis_fd_absolute,config.axis_fd_relative*span);
    for(auto& original:candidates){
        if(!original.elliptic){stats[3]+=1;continue;}
        if(!std::isfinite(original.residual)||original.residual>limit){stats[4]+=1;continue;}
        stats[0]+=1;
        AxisCandidate candidate=original;bool rejected=false;
        for(int iteration=0;iteration<2&&candidate.residual>config.axis_tolerance;++iteration){
            std::vector<double> R{candidate.R,std::min(domain.r_max,candidate.R+h),
                std::max(domain.r_min,candidate.R-h),candidate.R,candidate.R};
            std::vector<double> Z{candidate.Z,candidate.Z,candidate.Z,
                std::min(domain.z_max,candidate.Z+h),std::max(domain.z_min,candidate.Z-h)};
            std::vector<double> Re,Ze;auto mark=Timer::now();
            if(!trace_map(field,R,Z,nfp,config.axis_trace_steps,true,Re,Ze))return false;
            timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY]+=elapsed(mark);
            const double dr=R[1]-R[2],dz=Z[3]-Z[4];
            const double a=((Re[1]-R[1])-(Re[2]-R[2]))/dr;
            const double b=((Re[3]-R[3])-(Re[4]-R[4]))/dz;
            const double c=((Ze[1]-Z[1])-(Ze[2]-Z[2]))/dr;
            const double d=((Ze[3]-Z[3])-(Ze[4]-Z[4]))/dz;
            const double determinant=a*d-b*c;
            if(!std::isfinite(determinant)||std::abs(determinant)<=1e-14){stats[5]+=1;rejected=true;break;}
            const double fr=Re[0]-R[0],fz=Ze[0]-Z[0];
            const double next_R=candidate.R+(-fr*d+b*fz)/determinant;
            const double next_Z=candidate.Z+(c*fr-a*fz)/determinant;
            const double move=std::hypot(next_R-original.R,next_Z-original.Z);
            if(!std::isfinite(move)||move>limit||next_R<domain.r_min||next_R>domain.r_max||
                next_Z<domain.z_min||next_Z>domain.z_max){stats[6]+=1;rejected=true;break;}
            mark=Timer::now();
            if(!trace_map(field,{next_R},{next_Z},nfp,config.axis_trace_steps,true,Re,Ze))return false;
            timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY]+=elapsed(mark);
            const double residual=std::hypot(Re[0]-next_R,Ze[0]-next_Z);
            if(!std::isfinite(residual)||residual>=candidate.residual){stats[7]+=1;rejected=true;break;}
            candidate.R=next_R;candidate.Z=next_Z;candidate.residual=residual;stats[2]+=1;
        }
        if(rejected)continue;
        if(candidate.residual>config.axis_tolerance){stats[8]+=1;continue;}
        // Re-evaluate the original FP64 topology test at the polished root.
        candidate.elliptic=false;candidate.topology_trace=candidate.topology_det=
            std::numeric_limits<double>::quiet_NaN();candidate.ellipse_aspect=std::numeric_limits<double>::infinity();
        std::vector<AxisCandidate> verified{candidate};auto mark=Timer::now();
        classify_axis_topology(field,verified,domain,nfp,config);
        timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY]+=elapsed(mark);
        if(!verified[0].elliptic){stats[9]+=1;continue;}
        stats[10]=std::max(stats[10],std::hypot(candidate.R-original.R,candidate.Z-original.Z));
        original=verified[0];stats[1]+=1;
    }
    stats[11]=elapsed(started);return true;
}
}

extern "C" void ce_cpu_axis_polish_info(double* output,int reset){
    using namespace coil_eval::pipeline;
    if(reset)stats.fill(0.);
    if(output)std::copy(stats.begin(),stats.end(),output);
}
