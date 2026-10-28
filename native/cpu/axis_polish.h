#pragma once
#include "pipeline/axis.cuh"

namespace coil_eval::pipeline {
bool cpu_polish_rejected_axis_candidates(void*,std::vector<AxisCandidate>&,
    const AxisDomain&,int,const SgpuScoreConfig&,double*);
}
