#pragma once

// Common scalar conventions and device-buffer ownership. Domain records live in their module headers.
#include "coil_field.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace coil_eval::pipeline {

using Clock = std::chrono::steady_clock;

#ifndef SGPU_COIL_CURVATURE_P95_SCALE
#define SGPU_COIL_CURVATURE_P95_SCALE 10.0
#endif
constexpr double PI = 3.1415926535897932384626433832795;
constexpr double TWOPI = 2.0 * PI;
constexpr double COIL_CURVATURE_P95_SCALE = SGPU_COIL_CURVATURE_P95_SCALE;
static_assert(COIL_CURVATURE_P95_SCALE > 0.0, "curvature p95 scale must be positive");
constexpr int SCORE_STAGE_NONE = 0;
constexpr int SCORE_STAGE_FIELD = 1;
constexpr int SCORE_STAGE_AXIS = 2;
constexpr int SCORE_STAGE_PSI = 3;
constexpr int SCORE_STAGE_SURFACE = 4;
constexpr int SCORE_STAGE_FLUX = 5;
constexpr int SCORE_STAGE_ALPHA = 6;
constexpr int SCORE_STAGE_QS = 7;
constexpr int SCORE_STAGE_COMPLETE = 8;

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { cudaFree(pointer_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(pointer_);
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    bool allocate(size_t count) {
        cudaFree(pointer_);
        pointer_ = nullptr;
        count_ = 0;
        if (count == 0) return true;
        if (cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) != cudaSuccess) return false;
        count_ = count;
        return true;
    }

    T* data() { return pointer_; }
    const T* data() const { return pointer_; }
    size_t size() const { return count_; }

private:
    T* pointer_ = nullptr;
    size_t count_ = 0;
};

struct Vec3d {
    double x;
    double y;
    double z;
};

}  // namespace coil_eval::pipeline
