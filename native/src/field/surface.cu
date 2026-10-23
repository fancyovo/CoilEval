#include "field/state.h"
#include "coil_field.h"

#include <cmath>
#include <chrono>

namespace coil_eval::field {

__device__ inline double periodic_interp_uniform(double phi, const double* values, int n, double period) {
    double p = fmod(phi, period);
    if (p < 0.0) p += period;
    double pos = p * static_cast<double>(n) / period;
    int i0 = static_cast<int>(floor(pos));
    double t = pos - static_cast<double>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    return values[i0] * (1.0 - t) + values[i1] * t;
}

__device__ inline float periodic_interp_uniform_f32(float phi, const float* values, int n, float period) {
    float p = fmodf(phi, period);
    if (p < 0.0f) p += period;
    float pos = p * static_cast<float>(n) / period;
    int i0 = static_cast<int>(floorf(pos));
    float t = pos - static_cast<float>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    return values[i0] * (1.0f - t) + values[i1] * t;
}

__device__ inline void eval_ray_poly_and_deriv(
    const double* coeff_deg,
    int poly_degree,
    double u,
    double a_scale,
    double& psi,
    double& dpsi
) {
    psi = 0.0;
    dpsi = 0.0;
    if (poly_degree < 2) return;
    double u_pow_prev = u;
    double u_pow = u * u;
    psi = coeff_deg[2] * u_pow;
    dpsi = 2.0 * coeff_deg[2] * u_pow_prev / a_scale;
    for (int deg = 3; deg <= poly_degree; ++deg) {
        u_pow_prev = u_pow;
        u_pow *= u;
        psi += coeff_deg[deg] * u_pow;
        dpsi += static_cast<double>(deg) * coeff_deg[deg] * u_pow_prev / a_scale;
    }
}

__device__ inline double eval_ray_poly_only(
    const double* coeff_deg,
    int poly_degree,
    double u
) {
    if (poly_degree < 2) return 0.0;
    double u_pow = u * u;
    double psi = coeff_deg[2] * u_pow;
    for (int deg = 3; deg <= poly_degree; ++deg) {
        u_pow *= u;
        psi += coeff_deg[deg] * u_pow;
    }
    return psi;
}

__global__ void surface_build_ray_poly_kernel(
    const double* __restrict__ coeffs,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_coeff,
    int nfp,
    int poly_degree,
    int m_tor,
    int nphi,
    int ntheta,
    double* __restrict__ ray_coeff
) {
    extern __shared__ double sh[];
    double* cosv = sh;
    double* sinv = sh + (MAX_PSI_MTOR + 1);
    int iphi = blockIdx.x;
    double phi = (static_cast<double>(iphi) * TWOPI) / (static_cast<double>(nphi) * static_cast<double>(nfp));
    for (int m = threadIdx.x; m <= m_tor; m += blockDim.x) {
        if (m == 0) {
            cosv[m] = 1.0;
            sinv[m] = 0.0;
        } else {
            double arg = static_cast<double>(m * nfp) * phi;
            cosv[m] = cos(arg);
            sinv[m] = sin(arg);
        }
    }
    __syncthreads();

    for (int itheta = threadIdx.x; itheta < ntheta; itheta += blockDim.x) {
        double theta = (static_cast<double>(itheta) * TWOPI) / static_cast<double>(ntheta);
        double cth = cos(theta);
        double sth = sin(theta);
        double xpow[MAX_PSI_DEGREE + 1];
        double zpow[MAX_PSI_DEGREE + 1];
        double coeff_deg[MAX_PSI_DEGREE + 1];
        xpow[0] = 1.0;
        zpow[0] = 1.0;
        coeff_deg[0] = 0.0;
        coeff_deg[1] = 0.0;
        for (int deg = 2; deg <= poly_degree; ++deg) coeff_deg[deg] = 0.0;
        for (int deg = 1; deg <= poly_degree; ++deg) {
            xpow[deg] = xpow[deg - 1] * cth;
            zpow[deg] = zpow[deg - 1] * sth;
        }
        coeff_deg[2] = cth * cth;
        for (int k = 0; k < n_coeff; ++k) {
            int ax = mode_a[k];
            int bz = mode_b[k];
            int deg = ax + bz;
            int m = mode_m[k];
            double trig = (m == 0) ? 1.0 : ((mode_kind[k] == 0) ? cosv[m] : sinv[m]);
            coeff_deg[deg] += coeffs[k] * xpow[ax] * zpow[bz] * trig;
        }
        size_t base = (static_cast<size_t>(iphi) * ntheta + itheta) * static_cast<size_t>(poly_degree + 1);
        for (int deg = 0; deg <= poly_degree; ++deg) {
            ray_coeff[base + deg] = coeff_deg[deg];
        }
    }
}

__global__ void surface_newton_from_ray_poly_kernel(
    const double* __restrict__ ray_coeff,
    int poly_degree,
    const double* __restrict__ axis_R,
    const double* __restrict__ axis_Z,
    int n_axis,
    int nfp,
    double a_scale,
    int nphi,
    int ntheta,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    double* __restrict__ xyz,
    double* __restrict__ radii
) {
    __shared__ double ra_sh;
    __shared__ double za_sh;
    __shared__ double phi_sh;
    int iphi = blockIdx.x;
    if (threadIdx.x == 0) {
        phi_sh = (static_cast<double>(iphi) * TWOPI) / (static_cast<double>(nphi) * static_cast<double>(nfp));
        double period = TWOPI / static_cast<double>(nfp);
        ra_sh = periodic_interp_uniform(phi_sh, axis_R, n_axis, period);
        za_sh = periodic_interp_uniform(phi_sh, axis_Z, n_axis, period);
    }
    __syncthreads();

    double max_radius = max_radius_scale * a_scale;
    double fallback = a_scale * sqrt(fmax(psi_level, 1.0e-16));
    double cp = cos(phi_sh);
    double sp = sin(phi_sh);
    for (int itheta = threadIdx.x; itheta < ntheta; itheta += blockDim.x) {
        size_t base = (static_cast<size_t>(iphi) * ntheta + itheta) * static_cast<size_t>(poly_degree + 1);
        const double* coeff_deg = ray_coeff + base;
        double q = coeff_deg[2];
        double rho = (q > 1.0e-10) ? (a_scale * sqrt(fmax(psi_level, 0.0) / q)) : fallback;
        rho = fmin(fmax(rho, 1.0e-12 * a_scale), max_radius);
        for (int iter = 0; iter < maxiter; ++iter) {
            double u = rho / a_scale;
            double psi = 0.0, dpsi = 0.0;
            eval_ray_poly_and_deriv(coeff_deg, poly_degree, u, a_scale, psi, dpsi);
            double f = psi - psi_level;
            if (fabs(f) <= tol) break;
            double denom = (fabs(dpsi) > 1.0e-14) ? dpsi : copysign(1.0e-14, dpsi == 0.0 ? 1.0 : dpsi);
            double lim = 0.45 * fmax(fabs(rho), 1.0e-8 * a_scale);
            double step = f / denom;
            if (step > lim) step = lim;
            if (step < -lim) step = -lim;
            double trial = fmin(fmax(rho - step, 1.0e-12 * a_scale), max_radius);
            double psi_trial = eval_ray_poly_only(coeff_deg, poly_degree, trial / a_scale);
            if (fabs(psi_trial - psi_level) <= fabs(f)) {
                rho = trial;
            } else {
                rho = 0.5 * (rho + trial);
            }
        }
        double theta = (static_cast<double>(itheta) * TWOPI) / static_cast<double>(ntheta);
        double cth = cos(theta);
        double sth = sin(theta);
        double R = ra_sh + rho * cth;
        double Z = za_sh + rho * sth;
        size_t idx = static_cast<size_t>(iphi) * ntheta + itheta;
        radii[idx] = rho;
        xyz[3 * idx + 0] = R * cp;
        xyz[3 * idx + 1] = R * sp;
        xyz[3 * idx + 2] = Z;
    }
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_surface_points_from_level(
    const double* coeff_host,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    const double* axis_R_host,
    const double* axis_Z_host,
    int n_axis,
    int order,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    int device_id,
    double* xyz_host,
    double* radii_host,
    double* stats_out,
    int stats_len
) {
    using clock = std::chrono::steady_clock;
    if (!coeff_host || !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host ||
        !axis_R_host || !axis_Z_host || !xyz_host || !radii_host || n_coeff <= 0 || nfp <= 0 ||
        poly_degree < 2 || poly_degree > MAX_PSI_DEGREE || m_tor < 0 || m_tor > MAX_PSI_MTOR ||
        n_axis <= 1 || order < 1 || maxiter <= 0 || tol <= 0.0 || max_radius_scale <= 0.0) {
        set_error("invalid surface_points_from_level arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;
    int nphi = 2 * order + 1;
    int ntheta = 2 * order + 1;
    size_t coeff_bytes = static_cast<size_t>(n_coeff) * sizeof(double);
    size_t mode_bytes = static_cast<size_t>(n_coeff) * sizeof(int);
    size_t axis_bytes = static_cast<size_t>(n_axis) * sizeof(double);
    size_t ray_coeff_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * static_cast<size_t>(poly_degree + 1) * sizeof(double);
    size_t xyz_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * 3 * sizeof(double);
    size_t radii_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * sizeof(double);

    double *d_coeff = nullptr, *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_ray_coeff = nullptr, *d_xyz = nullptr, *d_radii = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_coeff);
        cudaFree(d_axis_R);
        cudaFree(d_axis_Z);
        cudaFree(d_ray_coeff);
        cudaFree(d_xyz);
        cudaFree(d_radii);
        cudaFree(d_mode_a);
        cudaFree(d_mode_b);
        cudaFree(d_mode_m);
        cudaFree(d_mode_kind);
    };

    double copy_in_s = 0.0, coeff_build_s = 0.0, newton_s = 0.0, copy_out_s = 0.0;
    auto t_total = clock::now();
    auto t = clock::now();
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff), coeff_bytes), "surface d_coeff") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R), axis_bytes), "surface d_axis_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z), axis_bytes), "surface d_axis_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_a), mode_bytes), "surface d_mode_a") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_b), mode_bytes), "surface d_mode_b") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_m), mode_bytes), "surface d_mode_m") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_kind), mode_bytes), "surface d_mode_kind") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ray_coeff), ray_coeff_bytes), "surface d_ray_coeff") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), xyz_bytes), "surface d_xyz") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_radii), radii_bytes), "surface d_radii") ||
        cuda_check(cudaMemcpy(d_coeff, coeff_host, coeff_bytes, cudaMemcpyHostToDevice), "surface copy coeff") ||
        cuda_check(cudaMemcpy(d_axis_R, axis_R_host, axis_bytes, cudaMemcpyHostToDevice), "surface copy axis_R") ||
        cuda_check(cudaMemcpy(d_axis_Z, axis_Z_host, axis_bytes, cudaMemcpyHostToDevice), "surface copy axis_Z") ||
        cuda_check(cudaMemcpy(d_mode_a, mode_a_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_a") ||
        cuda_check(cudaMemcpy(d_mode_b, mode_b_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_b") ||
        cuda_check(cudaMemcpy(d_mode_m, mode_m_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_m") ||
        cuda_check(cudaMemcpy(d_mode_kind, mode_kind_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_kind")) {
        cleanup();
        return 1;
    }
    copy_in_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    surface_build_ray_poly_kernel<<<nphi, SURFACE_THREADS, static_cast<size_t>(2 * (MAX_PSI_MTOR + 1)) * sizeof(double)>>>(
        d_coeff, d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, poly_degree, m_tor, nphi, ntheta, d_ray_coeff
    );
    if (cuda_check(cudaGetLastError(), "surface_build_ray_poly_kernel") ||
        cuda_check(cudaDeviceSynchronize(), "surface_build_ray_poly sync")) {
        cleanup();
        return 1;
    }
    coeff_build_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    surface_newton_from_ray_poly_kernel<<<nphi, SURFACE_THREADS>>>(
        d_ray_coeff,
        poly_degree,
        d_axis_R,
        d_axis_Z,
        n_axis,
        nfp,
        a,
        nphi,
        ntheta,
        psi_level,
        maxiter,
        tol,
        max_radius_scale,
        d_xyz,
        d_radii
    );
    if (cuda_check(cudaGetLastError(), "surface_newton_from_ray_poly_kernel") ||
        cuda_check(cudaDeviceSynchronize(), "surface_newton_from_ray_poly sync")) {
        cleanup();
        return 1;
    }
    newton_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    if (cuda_check(cudaMemcpy(xyz_host, d_xyz, xyz_bytes, cudaMemcpyDeviceToHost), "surface copy xyz") ||
        cuda_check(cudaMemcpy(radii_host, d_radii, radii_bytes, cudaMemcpyDeviceToHost), "surface copy radii")) {
        cleanup();
        return 1;
    }
    copy_out_s = std::chrono::duration<double>(clock::now() - t).count();

    if (stats_out && stats_len >= 5) {
        stats_out[0] = copy_in_s;
        stats_out[1] = coeff_build_s;
        stats_out[2] = newton_s;
        stats_out[3] = copy_out_s;
        stats_out[4] = std::chrono::duration<double>(clock::now() - t_total).count();
    }
    cleanup();
    set_error("");
    return 0;
}

} // extern "C"
