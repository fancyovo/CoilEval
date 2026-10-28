#pragma once
namespace coil_eval::cpu {
struct VjpBuffers {
    const void *p, *w, *q, *a, *h;
    void *dp, *dw, *dq;
    int segments, points, threads;
};
// Return the processed prefix; the original scalar code handles the tail.
int autograd_vjp_f32(const VjpBuffers&, bool potential, bool segment_outputs);
int autograd_vjp_f64(const VjpBuffers&, bool potential, bool segment_outputs);
}
