#pragma once

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

double seconds_since(Clock::time_point start);

double clip01(double value);

double q_down(double value, double scale, double power, double fallback = 0.0);

double q_up(double value, double scale, double power, double fallback = 0.0);

double q_saturating_up(double value, double saturation, double fallback = 0.0);

double minimum_absolute_iota(double iota_min, double iota_max);

double blend(std::initializer_list<std::pair<double, double>> values);

void initialize_result(SgpuScoreResult* result, int device_id);

int fail_result(SgpuScoreResult* result, const char* message);

int fail_from_backend(SgpuScoreResult* result, const char* stage);

double percentile(std::vector<double> values, double fraction);

bool cuda_stage_ok(cudaError_t error, SgpuScoreResult& result, const char* stage);


}  // namespace coil_eval::pipeline
