#pragma once
#include "pipeline/gradients.cuh"
namespace coil_eval::pipeline {
struct ReverseCapturedArray {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<double> values;
};

extern thread_local int g_reverse_capture_flags;
extern thread_local std::vector<ReverseCapturedArray> g_reverse_snapshot;
bool capture_reverse_snapshot(const FixedFrontG2Cache&,int,const SgpuScoreConfig&,const SgpuScoreResult&);
}
extern "C" {
int sgpu_set_reverse_capture(int flags);
int sgpu_reverse_capture_array_count();
int sgpu_reverse_capture_array_info(int index, char* name, int name_capacity,
                                  std::int64_t* shape, int shape_capacity, int* ndim);
int sgpu_reverse_capture_copy(int index, double* values, std::int64_t capacity);
}
