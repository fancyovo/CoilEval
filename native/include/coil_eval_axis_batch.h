#pragma once
#include "coil_field.h"

// Additive opt-in API; SgpuScoreConfig/result and capture ABI are unchanged.
extern "C" {
int sgpu_axis_coarse_grid(const double* x, const double* y, const double* z,
    int nbase, int ncoeff, const SgpuScoreConfig* config,
    double* R0, double* Z0, int capacity);
int sgpu_axis_coarse_domain(const double* x, const double* y, const double* z,
    int nbase, int ncoeff, const SgpuScoreConfig* config, double* domain);
int sgpu_axis_coarse_cache_set(const double* x, const double* y, const double* z,
    const double* currents, int nbase, int ncoeff, int nfp,
    const SgpuScoreConfig* config, const double* R0, const double* Z0,
    const double* R1, const double* Z1, int count);
int sgpu_axis_coarse_cache_clear();
int sgpu_axis_coarse_cache_status();
// Optional refinement payload uses [query,capacity,6] doubles and int winding.
int sgpu_axis_refinement_batch(void* batch, const SgpuScoreConfig* configs,
    int queries, int nfp, const double* R0, const double* Z0, const double* R1,
    const double* Z1, const double* domains, int grid, int capacity,
    double* payload, int* winding, int* counts);
int sgpu_axis_refinement_cache_set(const SgpuScoreConfig* config,
    const double* payload, const int* winding, int count);
int sgpu_axis_refinement_cache_status();
// Independent original single-query refiner for validation only.
int sgpu_axis_refinement_reference(void* field, const SgpuScoreConfig* config,
    int nfp, const double* domain, const double* payload, int count, double* refined);
// Use the same CPU segment generator as the original single-query scorer.
int sgpu_batch_field_use_host_segments(void* handle,
    const double* x, const double* y, const double* z, const double* currents,
    int nbase, int ncoeff, int nfp, int segments_per_coil);
}
