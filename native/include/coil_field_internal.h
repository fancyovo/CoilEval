#pragma once

extern "C" {

int sgpu_internal_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int n_points
);

int sgpu_internal_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int n_points
);

int sgpu_internal_batch_query_count(void* handle);

int sgpu_internal_batch_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int points_per_query
);

int sgpu_internal_batch_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int points_per_query
);

int sgpu_internal_B_grad_segment_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_segment_position_device,
    float* adj_segment_weight_device
);

int sgpu_internal_B_grad_point_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_xyz_device
);

void sgpu_internal_set_error(const char* message);

}
