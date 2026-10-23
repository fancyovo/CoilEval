cmake_minimum_required(VERSION 3.22)
project(coil_eval_native LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

option(SGPU_ENABLE_NVTX "Enable profiling-only NVTX ranges" OFF)
option(COIL_EVAL_ENABLE_TORSION "Build the optional torsion geometry extension" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(coil_eval_native SHARED
    src/field/preconditioner_export.cpp
    src/field/state.cpp
    src/field/segments.cu
    src/field/vector_potential.cu
    src/field/biot_savart.cu
    src/field/tracing.cu
    src/field/field_adjoint.cu
    src/field/psi_design.cu
    src/field/psi_fit.cu
    src/field/psi_batch.cu
    src/field/surface.cu
    src/pipeline/common.cu
    src/pipeline/coil_geometry.cu
    src/pipeline/axis.cu
    src/pipeline/psi.cu
    src/pipeline/surface.cu
    src/pipeline/policy.cu
    src/pipeline/stokes.cu
    src/pipeline/flux.cu
    src/pipeline/volume.cu
    src/pipeline/alpha.cu
    src/pipeline/qs.cu
    src/pipeline/evaluate.cu
    src/pipeline/gradients.cu
    src/pipeline/local_kernels.cu
    src/pipeline/local_batch.cu
    src/pipeline/target_volume.cu
    src/pipeline/session_state.cu
    src/pipeline/reverse_capture.cu
    src/pipeline/api.cu
    src/pipeline/legacy_abi.cu
)
if(COIL_EVAL_ENABLE_TORSION)
    target_sources(coil_eval_native PRIVATE src/coil_torsion.cpp)
endif()

target_include_directories(coil_eval_native PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(coil_eval_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(coil_eval_native PUBLIC CUDA::cublas CUDA::cusolver)
target_link_options(coil_eval_native PRIVATE -static-libstdc++ -static-libgcc)

set(SGPU_WARPS_PER_BLOCK "8" CACHE STRING "Warps per block for warp-per-line kernels")
set(SGPU_SEG_TILE "256" CACHE STRING "Segment tile size")
set(SGPU_COIL_CURVATURE_P95_SCALE "10.0" CACHE STRING
    "Coil curvature p95 score scale in inverse metres")
target_compile_definitions(coil_eval_native PRIVATE
    SGPU_WARPS_PER_BLOCK=${SGPU_WARPS_PER_BLOCK}
    SGPU_SEG_TILE=${SGPU_SEG_TILE}
    SGPU_COIL_CURVATURE_P95_SCALE=${SGPU_COIL_CURVATURE_P95_SCALE}
)
if(SGPU_ENABLE_NVTX)
    target_compile_definitions(coil_eval_native PRIVATE SGPU_ENABLE_NVTX=1)
    target_link_libraries(coil_eval_native PRIVATE CUDA::nvtx3)
endif()
set_target_properties(coil_eval_native PROPERTIES
    OUTPUT_NAME coil_eval_native
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(coil_eval_autograd SHARED
    src/autograd/field.cu
    src/autograd/trace.cu
    src/autograd/batch_field.cu
)
target_link_libraries(coil_eval_autograd PRIVATE CUDA::cudart)
target_link_options(coil_eval_autograd PRIVATE -static-libstdc++ -static-libgcc)
