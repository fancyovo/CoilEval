#pragma once
// Host-only compatibility for shared policy headers. No GPU entry point is
// linked or used by this target; DeviceBuffer storage is ordinary host memory.
#include <cstdlib>
#include <cstddef>
using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline int cudaMalloc(void** p, std::size_t n) { *p=std::malloc(n); return *p || !n ? 0 : 1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline const char* cudaGetErrorString(int e) { return e ? "CPU allocation failed" : "success"; }
