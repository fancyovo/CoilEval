// Compile with production flux.cu, local_kernels.cu and surface.cu, and link
// the remaining dependencies to the built library. CUDA global entry points
// must belong to this executable's device link, rather than an external stub.
#include "pipeline/flux.cuh"
#include "pipeline/local_kernels.cuh"
#include "pipeline/surface.cuh"
#include <iostream>

using namespace coil_eval::pipeline;

int main() {
    int frozen_count, degree;
    double frozen_level;
    if (!(std::cin >> frozen_count >> degree >> frozen_level) || degree != 10) return 1;
    const int count = frozen_count + 4;
    std::vector<std::array<double,25>> original(count);
    std::vector<float> polynomial(count * (degree + 1)), levels(count), expected(count);
    for (int i = 0; i < frozen_count; ++i) {
        for (int j = 0; j <= degree; ++j) std::cin >> original[i][j];
        std::cin >> expected[i];
        levels[i] = frozen_level;
    }
    auto& p6 = original[frozen_count];
    p6[2] = 1; p6[4] = 4; p6[6] = -8;
    levels[frozen_count] = .5; expected[frozen_count] = std::pow(2., -.75);
    auto& p8 = original[frozen_count + 1];
    p8[2] = 1; p8[4] = 8; p8[6] = -16; p8[8] = 8;
    levels[frozen_count + 1] = 1; expected[frozen_count + 1] = std::sqrt(.5);
    original[frozen_count + 2] = p8;
    levels[frozen_count + 2] = 1.1; expected[frozen_count + 2] = 0;
    original[frozen_count + 3][2] = 1;
    levels[frozen_count + 3] = .09; expected[frozen_count + 3] = .3;
    for (int i = 0; i < count; ++i)
        for (int j = 0; j <= degree; ++j) polynomial[i * (degree + 1) + j] = original[i][j];

    float *dp, *dl, *dr, *de, *br, *be;
    if (cudaMalloc(&dp, polynomial.size()*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dl, count*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dr, count*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&de, count*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&br, count*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&be, count*sizeof(float)) != cudaSuccess) return 2;
    cudaMemcpy(dp, polynomial.data(), polynomial.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dl, levels.data(), count*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dr, 255, count*sizeof(float));cudaMemset(de, 255, count*sizeof(float));
    cudaMemset(br, 255, count*sizeof(float));cudaMemset(be, 255, count*sizeof(float));
    for (int i = 0; i < count; ++i) {
        solve_boundary_radii_kernel<<<1,32>>>(dp+i*(degree+1),1,degree,1.,dl+i,
            1,20,1e-8,1.2,dr+i,de+i);
        if (cudaGetLastError() != cudaSuccess) return 6;
    }
    local_batch_solve_boundary_radii_kernel<<<1,32>>>(dp,1,count,degree,1.,dl,
        1,20,1e-8,1.2,br,be);
    if (cudaGetLastError() != cudaSuccess) return 7;
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> radius(count), residual(count), batched(count), batch_error(count);
    cudaMemcpy(radius.data(),dr,count*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(residual.data(),de,count*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(batched.data(),br,count*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_error.data(),be,count*sizeof(float),cudaMemcpyDeviceToHost);
    SgpuScoreConfig config{};
    config.psi_a=1.; config.psi_poly_degree=degree; config.surface_max_radius_scale=1.2;
    config.surface_newton_iters=20; config.surface_newton_tolerance=1e-11;
    for (int i = 0; i < count; ++i) {
        const double host = solve_ray_radius(original[i],levels[i],config);
        if (expected[i] == 0) {
            if (radius[i] != 0 || batched[i] != 0 || host != 0 ||
                !std::isinf(residual[i]) || !std::isinf(batch_error[i])) return 4;
        } else if (!std::isfinite(radius[i]) || !std::isfinite(batched[i]) ||
                   !std::isfinite(residual[i]) || !std::isfinite(batch_error[i]) ||
                   std::abs(radius[i]-expected[i]) > 2e-6 ||
                   std::abs(batched[i]-expected[i]) > 2e-6 ||
                   std::abs(host-expected[i]) > 2e-6 ||
                   residual[i] > 2e-6 || batch_error[i] > 2e-6) {
            std::cerr << "ray " << i << ": " << radius[i] << " " << batched[i]
                      << " " << host << " expected " << expected[i] << "\n";
            return 5;
        }
    }
    cudaFree(dp);cudaFree(dl);cudaFree(dr);cudaFree(de);cudaFree(br);cudaFree(be);
    std::cout << "production formal/local GPU and host first-branch regression passed: "
              << count << " rays\n";
}
