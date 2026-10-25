#include "field/state.h"
#include "coil_field.h"
#include "coil_field_internal.h"

#include <algorithm>
#include <cmath>
#include <exception>

namespace coil_eval::field {

void eval_fourier(const double* c, int order, double t, double& x, double& dxdt) {
    x = c[0];
    dxdt = 0.0;
    for (int m = 1; m <= order; ++m) {
        double s = sin(TWOPI * m * t);
        double co = cos(TWOPI * m * t);
        double sin_coef = c[2 * m - 1];
        double cos_coef = c[2 * m];
        x += sin_coef * s + cos_coef * co;
        dxdt += TWOPI * m * (sin_coef * co - cos_coef * s);
    }
}

void rotate_z(double angle, double x, double y, double& xo, double& yo) {
    double c = cos(angle);
    double s = sin(angle);
    xo = c * x - s * y;
    yo = s * x + c * y;
}

void append_segment(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z,
    std::vector<double>& wx,
    std::vector<double>& wy,
    std::vector<double>& wz,
    double px,
    double py,
    double pz,
    double dlx,
    double dly,
    double dlz,
    double current
) {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
    wx.push_back(current * dlx);
    wy.push_back(current * dly);
    wz.push_back(current * dlz);
}

void generate_segments(
    const double* cx,
    const double* cy,
    const double* cz,
    const double* currents,
    int nbase,
    int ncoeff,
    int nfp,
    int segs_per_coil,
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z,
    std::vector<double>& wx,
    std::vector<double>& wy,
    std::vector<double>& wz
) {
    int order = (ncoeff - 1) / 2;
    x.clear(); y.clear(); z.clear(); wx.clear(); wy.clear(); wz.clear();
    x.reserve(static_cast<size_t>(nbase) * 2 * nfp * segs_per_coil);
    y.reserve(x.capacity()); z.reserve(x.capacity());
    wx.reserve(x.capacity()); wy.reserve(x.capacity()); wz.reserve(x.capacity());

    for (int b = 0; b < nbase; ++b) {
        const double* bx = cx + static_cast<size_t>(b) * ncoeff;
        const double* by = cy + static_cast<size_t>(b) * ncoeff;
        const double* bz = cz + static_cast<size_t>(b) * ncoeff;
        double current = currents[b];
        for (int s = 0; s < segs_per_coil; ++s) {
            double t = (static_cast<double>(s) + 0.5) / static_cast<double>(segs_per_coil);
            double px, py, pz, vx, vy, vz;
            eval_fourier(bx, order, t, px, vx);
            eval_fourier(by, order, t, py, vy);
            eval_fourier(bz, order, t, pz, vz);
            double scale = 1.0 / static_cast<double>(segs_per_coil);
            vx *= scale; vy *= scale; vz *= scale;
            for (int k = 0; k < nfp; ++k) {
                double angle = TWOPI * static_cast<double>(k) / static_cast<double>(nfp);
                double rx, ry, rdx, rdy;
                rotate_z(angle, px, py, rx, ry);
                rotate_z(angle, vx, vy, rdx, rdy);
                append_segment(x, y, z, wx, wy, wz, rx, ry, pz, rdx, rdy, vz, current);
                double mx = px, my = -py, mz = -pz;
                double mdx = vx, mdy = -vy, mdz = -vz;
                rotate_z(angle, mx, my, rx, ry);
                rotate_z(angle, mdx, mdy, rdx, rdy);
                append_segment(x, y, z, wx, wy, wz, rx, ry, mz, rdx, rdy, mdz, -current);
            }
        }
    }
}

__device__ void eval_fourier_device(
    const double* coefficients,
    int order,
    double t,
    double& value,
    double& derivative
) {
    value = coefficients[0];
    derivative = 0.0;
    for (int mode = 1; mode <= order; ++mode) {
        double sine = 0.0;
        double cosine = 0.0;
        sincos(TWOPI * static_cast<double>(mode) * t, &sine, &cosine);
        const double sin_coefficient = coefficients[2 * mode - 1];
        const double cos_coefficient = coefficients[2 * mode];
        value += sin_coefficient * sine + cos_coefficient * cosine;
        derivative += TWOPI * static_cast<double>(mode) *
            (sin_coefficient * cosine - cos_coefficient * sine);
    }
}

__global__ void generate_segment_batch_kernel(
    const double* __restrict__ coeffs_x,
    const double* __restrict__ coeffs_y,
    const double* __restrict__ coeffs_z,
    const double* __restrict__ currents,
    int query_count,
    int nbase,
    int ncoeff,
    int nfp,
    int segments_per_coil,
    int segments_per_query,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ z,
    float* __restrict__ wx,
    float* __restrict__ wy,
    float* __restrict__ wz
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(query_count) * segments_per_query;
    if (flat >= total) return;
    const int query = static_cast<int>(flat / segments_per_query);
    int local = static_cast<int>(flat - static_cast<size_t>(query) * segments_per_query);
    const int image = local % (2 * nfp);
    local /= 2 * nfp;
    const int segment = local % segments_per_coil;
    const int coil = local / segments_per_coil;
    const int period_index = image / 2;
    const bool reflected = (image & 1) != 0;
    const size_t coefficient_offset =
        (static_cast<size_t>(query) * nbase + coil) * ncoeff;
    const double t = (static_cast<double>(segment) + 0.5) /
        static_cast<double>(segments_per_coil);
    const int order = (ncoeff - 1) / 2;
    double px = 0.0, py = 0.0, pz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    eval_fourier_device(coeffs_x + coefficient_offset, order, t, px, vx);
    eval_fourier_device(coeffs_y + coefficient_offset, order, t, py, vy);
    eval_fourier_device(coeffs_z + coefficient_offset, order, t, pz, vz);
    const double segment_scale = 1.0 / static_cast<double>(segments_per_coil);
    vx *= segment_scale;
    vy *= segment_scale;
    vz *= segment_scale;
    double current = currents[static_cast<size_t>(query) * nbase + coil];
    if (reflected) {
        py = -py;
        pz = -pz;
        vy = -vy;
        vz = -vz;
        current = -current;
    }
    const double angle = TWOPI * static_cast<double>(period_index) /
        static_cast<double>(nfp);
    double sine = 0.0;
    double cosine = 0.0;
    sincos(angle, &sine, &cosine);
    const double rx = cosine * px - sine * py;
    const double ry = sine * px + cosine * py;
    const double rvx = cosine * vx - sine * vy;
    const double rvy = sine * vx + cosine * vy;
    x[flat] = static_cast<float>(rx);
    y[flat] = static_cast<float>(ry);
    z[flat] = static_cast<float>(pz);
    wx[flat] = static_cast<float>(current * rvx);
    wy[flat] = static_cast<float>(current * rvy);
    wz[flat] = static_cast<float>(current * vz);
}

template <typename T>
int copy_to_device(T** dst, const std::vector<T>& src, const char* name) {
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(T)), name)) return 1;
    if (cuda_check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), name)) return 1;
    return 0;
}

std::vector<float> to_float_vector(const std::vector<double>& src) {
    std::vector<float> out;
    out.reserve(src.size());
    for (double v : src) out.push_back(static_cast<float>(v));
    return out;
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_create_field(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
) {
    try {
        if (!out_handle) {
            set_error("out_handle is null");
            return 1;
        }
        *out_handle = nullptr;
        if (n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || segments_per_coil <= 0) {
            set_error("invalid create_field dimensions");
            return 1;
        }
        if (cuda_check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;
        std::vector<double> x, y, z, wx, wy, wz;
        generate_segments(coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff, nfp, segments_per_coil, x, y, z, wx, wy, wz);
        std::vector<float> xf = to_float_vector(x);
        std::vector<float> yf = to_float_vector(y);
        std::vector<float> zf = to_float_vector(z);
        std::vector<float> wxf = to_float_vector(wx);
        std::vector<float> wyf = to_float_vector(wy);
        std::vector<float> wzf = to_float_vector(wz);
        CoilField* f = new CoilField();
        f->device_id = device_id;
        f->n_segments = static_cast<int>(x.size());
        if (cublas_check(cublasCreate(&f->blas), "cublasCreate field") ||
            cusolver_check(cusolverDnCreate(&f->solver), "cusolverDnCreate field")) {
            sgpu_destroy_field(f);
            return 1;
        }
        if (copy_to_device(&f->d_x, x, "d_x") ||
            copy_to_device(&f->d_y, y, "d_y") ||
            copy_to_device(&f->d_z, z, "d_z") ||
            copy_to_device(&f->d_wx, wx, "d_wx") ||
            copy_to_device(&f->d_wy, wy, "d_wy") ||
            copy_to_device(&f->d_wz, wz, "d_wz") ||
            copy_to_device(&f->d_x_f, xf, "d_x_f") ||
            copy_to_device(&f->d_y_f, yf, "d_y_f") ||
            copy_to_device(&f->d_z_f, zf, "d_z_f") ||
            copy_to_device(&f->d_wx_f, wxf, "d_wx_f") ||
            copy_to_device(&f->d_wy_f, wyf, "d_wy_f") ||
            copy_to_device(&f->d_wz_f, wzf, "d_wz_f")) {
            sgpu_destroy_field(f);
            return 1;
        }
        *out_handle = f;
        set_error("");
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return 1;
    }
}

int sgpu_create_field_batch_f32(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
) {
    if (!out_handle) {
        set_error("batch field out_handle is null");
        return 1;
    }
    *out_handle = nullptr;
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || (n_coeff & 1) == 0 || nfp <= 0 ||
        segments_per_coil <= 0) {
        set_error("invalid batch field dimensions");
        return 1;
    }
    if (cuda_check(cudaSetDevice(device_id), "batch field cudaSetDevice")) return 1;
    BatchCoilField* field = new BatchCoilField();
    field->device_id = device_id;
    field->query_count = query_count;
    field->n_segments = n_base_coils * 2 * nfp * segments_per_coil;
    const size_t coefficient_count = static_cast<size_t>(query_count) * n_base_coils * n_coeff;
    const size_t current_count = static_cast<size_t>(query_count) * n_base_coils;
    const size_t segment_count = static_cast<size_t>(query_count) * field->n_segments;
    double* d_coeffs_x = nullptr;
    double* d_coeffs_y = nullptr;
    double* d_coeffs_z = nullptr;
    double* d_currents = nullptr;
    auto cleanup_inputs = [&] {
        cudaFree(d_coeffs_x);
        cudaFree(d_coeffs_y);
        cudaFree(d_coeffs_z);
        cudaFree(d_currents);
    };
    auto allocate = [](void** pointer, size_t bytes) {
        return cuda_check(cudaMalloc(pointer, bytes), "batch field allocation") == 0;
    };
    if (!allocate(reinterpret_cast<void**>(&d_coeffs_x), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_coeffs_y), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_coeffs_z), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_currents), current_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&field->d_x), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_y), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_z), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wx), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wy), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wz), segment_count * sizeof(float))) {
        cleanup_inputs();
        sgpu_destroy_field_batch(field);
        return 1;
    }
    const size_t coefficient_bytes = coefficient_count * sizeof(double);
    const size_t current_bytes = current_count * sizeof(double);
    if (cuda_check(cudaMemcpy(d_coeffs_x, coeffs_x, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_x") ||
        cuda_check(cudaMemcpy(d_coeffs_y, coeffs_y, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_y") ||
        cuda_check(cudaMemcpy(d_coeffs_z, coeffs_z, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_z") ||
        cuda_check(cudaMemcpy(d_currents, currents_a, current_bytes, cudaMemcpyHostToDevice), "batch copy currents")) {
        cleanup_inputs();
        sgpu_destroy_field_batch(field);
        return 1;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
    generate_segment_batch_kernel<<<blocks, threads>>>(
        d_coeffs_x, d_coeffs_y, d_coeffs_z, d_currents, query_count,
        n_base_coils, n_coeff, nfp, segments_per_coil, field->n_segments,
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz
    );
    const bool failed =
        cuda_check(cudaGetLastError(), "batch segment generation") ||
        cuda_check(cudaDeviceSynchronize(), "batch segment generation sync");
    cleanup_inputs();
    if (failed) {
        sgpu_destroy_field_batch(field);
        return 1;
    }
    *out_handle = field;
    set_error("");
    return 0;
}

int sgpu_batch_field_use_host_segments(void* handle,
    const double* coeffs_x, const double* coeffs_y, const double* coeffs_z,
    const double* currents, int nbase, int ncoeff, int nfp, int segments_per_coil) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !coeffs_x || !coeffs_y || !coeffs_z || !currents || nbase <= 0 ||
        ncoeff < 3 || (ncoeff % 2) == 0 || nfp <= 0 || segments_per_coil <= 0 ||
        static_cast<std::int64_t>(nbase) * 2 * nfp * segments_per_coil != field->n_segments) {
        set_error("invalid host-segment batch dimensions");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "host-segment batch cudaSetDevice")) return 1;
    try {
        const size_t coefficients = static_cast<size_t>(nbase) * ncoeff;
        std::vector<double> x, y, z, wx, wy, wz;
        for (int query = 0; query < field->query_count; ++query) {
            const size_t offset = static_cast<size_t>(query) * coefficients;
            generate_segments(coeffs_x + offset, coeffs_y + offset, coeffs_z + offset,
                currents + static_cast<size_t>(query) * nbase, nbase, ncoeff, nfp,
                segments_per_coil, x, y, z, wx, wy, wz);
            const size_t segment_offset = static_cast<size_t>(query) * field->n_segments;
            const auto copy = [&](float* destination, const std::vector<double>& values) {
                const auto floats = to_float_vector(values);
                return cuda_check(cudaMemcpy(destination + segment_offset, floats.data(),
                    floats.size() * sizeof(float), cudaMemcpyHostToDevice), "copy exact host batch segments");
            };
            if (copy(field->d_x, x) || copy(field->d_y, y) || copy(field->d_z, z) ||
                copy(field->d_wx, wx) || copy(field->d_wy, wy) || copy(field->d_wz, wz)) return 1;
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    }
}

void sgpu_destroy_field_batch(void* handle) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field) return;
    cudaSetDevice(field->device_id);
    cudaFree(field->d_x);
    cudaFree(field->d_y);
    cudaFree(field->d_z);
    cudaFree(field->d_wx);
    cudaFree(field->d_wy);
    cudaFree(field->d_wz);
    delete field;
}

int sgpu_internal_batch_query_count(void* handle) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    return field ? field->query_count : 0;
}

void sgpu_destroy_field(void* handle) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f) return;
    cudaSetDevice(f->device_id);
    if (f->blas) cublasDestroy(f->blas);
    if (f->solver) cusolverDnDestroy(f->solver);
    cudaFree(f->d_x);
    cudaFree(f->d_y);
    cudaFree(f->d_z);
    cudaFree(f->d_wx);
    cudaFree(f->d_wy);
    cudaFree(f->d_wz);
    cudaFree(f->d_x_f);
    cudaFree(f->d_y_f);
    cudaFree(f->d_z_f);
    cudaFree(f->d_wx_f);
    cudaFree(f->d_wy_f);
    cudaFree(f->d_wz_f);
    delete f;
}

int sgpu_segment_count(void* handle) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    return f ? f->n_segments : 0;
}

} // extern "C"
