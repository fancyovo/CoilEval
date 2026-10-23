#include "field/device_field.cuh"
#include "field/psi_design.cuh"


namespace coil_eval::field {

__device__ inline void periodic_hermite_uniform(
    double phi,
    const double* values,
    const double* derivatives,
    int n,
    double period,
    double& value,
    double& derivative
) {
    double p = fmod(phi, period);
    if (p < 0.0) p += period;
    double pos = p * static_cast<double>(n) / period;
    int i0 = static_cast<int>(floor(pos));
    double t = pos - static_cast<double>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    double h = period / static_cast<double>(n);
    double t2 = t * t;
    double t3 = t2 * t;
    double y0 = values[i0];
    double y1 = values[i1];
    double d0 = derivatives[i0];
    double d1 = derivatives[i1];
    value = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
          + (t3 - 2.0 * t2 + t) * h * d0
          + (-2.0 * t3 + 3.0 * t2) * y1
          + (t3 - t2) * h * d1;
    derivative = (6.0 * t2 - 6.0 * t) * y0 / h
               + (3.0 * t2 - 4.0 * t + 1.0) * d0
               + (-6.0 * t2 + 6.0 * t) * y1 / h
               + (3.0 * t2 - 2.0 * t) * d1;
}

__device__ inline void periodic_hermite_uniform_f32(
    float phi,
    const float* values,
    const float* derivatives,
    int n,
    float period,
    float& value,
    float& derivative
) {
    float p = fmodf(phi, period);
    if (p < 0.0f) p += period;
    float pos = p * static_cast<float>(n) / period;
    int i0 = static_cast<int>(floorf(pos));
    float t = pos - static_cast<float>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    float h = period / static_cast<float>(n);
    float t2 = t * t;
    float t3 = t2 * t;
    float y0 = values[i0];
    float y1 = values[i1];
    float d0 = derivatives[i0];
    float d1 = derivatives[i1];
    value = (2.0f * t3 - 3.0f * t2 + 1.0f) * y0
          + (t3 - 2.0f * t2 + t) * h * d0
          + (-2.0f * t3 + 3.0f * t2) * y1
          + (t3 - t2) * h * d1;
    derivative = (6.0f * t2 - 6.0f * t) * y0 / h
               + (3.0f * t2 - 4.0f * t + 1.0f) * d0
               + (-6.0f * t2 + 6.0f * t) * y1 / h
               + (3.0f * t2 - 2.0f * t) * d1;
}

__global__ void build_psi_common_basis_kernel(
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int coefficient_count,
    int nfp,
    float radius_scale,
    float* __restrict__ derivative_R,
    float* __restrict__ derivative_Z,
    float* __restrict__ derivative_phi
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(point_count) * coefficient_count;
    if (flat >= count) return;
    const int column = static_cast<int>(flat / point_count);
    const int row = static_cast<int>(flat - static_cast<size_t>(column) * point_count);
    const int a = mode_a[column];
    const int b = mode_b[column];
    const int m = mode_m[column];
    const int kind = mode_kind[column];
    const float X = normalized_R[row];
    const float Y = normalized_Z[row];
    float x_power = 1.0f;
    float y_power = 1.0f;
    float x_derivative_power = 1.0f;
    float y_derivative_power = 1.0f;
    for (int power = 0; power < a; ++power) x_power *= X;
    for (int power = 0; power < b; ++power) y_power *= Y;
    for (int power = 1; power < a; ++power) x_derivative_power *= X;
    for (int power = 1; power < b; ++power) y_derivative_power *= Y;
    const float mono = x_power * y_power;
    const float mono_R = a == 0 ? 0.0f :
        static_cast<float>(a) * x_derivative_power * y_power / radius_scale;
    const float mono_Z = b == 0 ? 0.0f :
        static_cast<float>(b) * x_power * y_derivative_power / radius_scale;
    const float argument = static_cast<float>(m * nfp) * phi[row];
    const float cosine = m == 0 ? 1.0f : cosf(argument);
    const float sine = m == 0 ? 0.0f : sinf(argument);
    const float trig = m == 0 ? 1.0f : (kind == 0 ? cosine : sine);
    const float trig_phi = m == 0 ? 0.0f : static_cast<float>(m * nfp) *
        (kind == 0 ? -sine : cosine);
    derivative_R[flat] = mono_R * trig;
    derivative_Z[flat] = mono_Z * trig;
    derivative_phi[flat] = mono * trig_phi;
}

__global__ void build_psi_batch_features_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int segments_per_query,
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    int nfp,
    float radius_scale,
    float* __restrict__ feature_R,
    float* __restrict__ feature_Z,
    float* __restrict__ feature_phi,
    float* __restrict__ rhs
) {
    extern __shared__ float shared[];
    const size_t flat = blockIdx.x;
    const int query = static_cast<int>(flat / point_count);
    const int row = static_cast<int>(flat - static_cast<size_t>(query) * point_count);
    const size_t segment_offset = static_cast<size_t>(query) * segments_per_query;
    const size_t axis_offset = static_cast<size_t>(query) * axis_count;
    const float phii = phi[row];
    const float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float axis_R_value = 0.0f, axis_Z_value = 0.0f;
    float axis_R_derivative = 0.0f, axis_Z_derivative = 0.0f;
    if (threadIdx.x == 0) {
        periodic_hermite_uniform_f32(
            phii, axis_R + axis_offset, axis_R_phi + axis_offset,
            axis_count, period, axis_R_value, axis_R_derivative
        );
        periodic_hermite_uniform_f32(
            phii, axis_Z + axis_offset, axis_Z_phi + axis_offset,
            axis_count, period, axis_Z_value, axis_Z_derivative
        );
        shared[SEG_TILE * 6 + blockDim.x * 3] = axis_R_value;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 1] = axis_Z_value;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 2] = axis_R_derivative;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 3] = axis_Z_derivative;
    }
    __syncthreads();
    axis_R_value = shared[SEG_TILE * 6 + blockDim.x * 3];
    axis_Z_value = shared[SEG_TILE * 6 + blockDim.x * 3 + 1];
    axis_R_derivative = shared[SEG_TILE * 6 + blockDim.x * 3 + 2];
    axis_Z_derivative = shared[SEG_TILE * 6 + blockDim.x * 3 + 3];
    const float R = axis_R_value + radius_scale * normalized_R[row];
    const float Z = axis_Z_value + radius_scale * normalized_Z[row];
    const float cosine = cosf(phii);
    const float sine = sinf(phii);
    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    eval_B_block_f32(
        R * cosine, R * sine, Z,
        seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
        seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
        segments_per_query, shared, bx, by, bz
    );
    if (threadIdx.x == 0) {
        const float bR = bx * cosine + by * sine;
        const float bphi = -bx * sine + by * cosine;
        const float cphi = bphi / R;
        const float cR = bR - cphi * axis_R_derivative;
        const float cZ = bz - cphi * axis_Z_derivative;
        feature_R[flat] = cR;
        feature_Z[flat] = cZ;
        feature_phi[flat] = cphi;
        rhs[flat] = -2.0f * normalized_R[row] * cR / radius_scale;
    }
}

__global__ void psi_fill_matrix_kernel_f64(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R,
    const double* __restrict__ Z,
    const double* __restrict__ phi,
    const double* __restrict__ axis_R,
    const double* __restrict__ axis_Z,
    const double* __restrict__ axis_R_phi,
    const double* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double* __restrict__ mat,
    double* __restrict__ rhs
) {
    extern __shared__ unsigned char shbuf[];
    int row = blockIdx.x;
    if (row >= gridDim.x) return;
    double* sh = reinterpret_cast<double*>(shbuf);
    int base_count = SEG_TILE * 6 + blockDim.x * 3;
    double* aux = sh + base_count;
    double* xpow = aux;
    double* zpow = xpow + (MAX_PSI_DEGREE + 1);
    double* cosv = zpow + (MAX_PSI_DEGREE + 1);
    double* sinv = cosv + (MAX_PSI_MTOR + 1);
    double* scalar = sinv + (MAX_PSI_MTOR + 1);

    double Ri = R[row];
    double Zi = Z[row];
    double phii = phi[row];

    double cp = cos(phii);
    double sp = sin(phii);
    double bx, by, bz;
    eval_B_block(Ri * cp, Ri * sp, Zi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);

    if (threadIdx.x == 0) {
        double period = TWOPI / static_cast<double>(nfp);
        double ra, za, rap, zap;
        periodic_hermite_uniform(phii, axis_R, axis_R_phi, n_axis, period, ra, rap);
        periodic_hermite_uniform(phii, axis_Z, axis_Z_phi, n_axis, period, za, zap);
        double X = (Ri - ra) / a;
        double Zc = (Zi - za) / a;
        double br = bx * cp + by * sp;
        double bphi = -bx * sp + by * cp;
        double cphi = bphi / Ri;
        double cR = br - cphi * rap;
        double cZ = bz - cphi * zap;
        xpow[0] = 1.0;
        zpow[0] = 1.0;
        for (int k = 1; k <= poly_degree; ++k) {
            xpow[k] = xpow[k - 1] * X;
            zpow[k] = zpow[k - 1] * Zc;
        }
        cosv[0] = 1.0;
        sinv[0] = 0.0;
        for (int m = 1; m <= m_tor; ++m) {
            double arg = static_cast<double>(m * nfp) * phii;
            cosv[m] = cos(arg);
            sinv[m] = sin(arg);
        }
        scalar[0] = cR;
        scalar[1] = cZ;
        scalar[2] = cphi;
        scalar[3] = -2.0 * X * cR / a;
    }
    __syncthreads();

    if (threadIdx.x == 0) rhs[row] = scalar[3];
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        int ax = mode_a[col];
        int bz_exp = mode_b[col];
        int m = mode_m[col];
        int kind = mode_kind[col];
        double mono = xpow[ax] * zpow[bz_exp];
        double mono_x = (ax == 0) ? 0.0 : static_cast<double>(ax) * xpow[ax - 1] * zpow[bz_exp];
        double mono_z = (bz_exp == 0) ? 0.0 : static_cast<double>(bz_exp) * xpow[ax] * zpow[bz_exp - 1];
        double spatial = (scalar[0] * mono_x + scalar[1] * mono_z) / a;
        double trig = (m == 0) ? 1.0 : ((kind == 0) ? cosv[m] : sinv[m]);
        double trig_phi = 0.0;
        if (m != 0) {
            double fac = static_cast<double>(m * nfp);
            trig_phi = (kind == 0) ? (-fac * sinv[m]) : (fac * cosv[m]);
        }
        mat[static_cast<size_t>(row) * n_cols + col] = spatial * trig + scalar[2] * mono * trig_phi;
    }
}

__global__ void psi_fill_matrix_kernel_f32(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ R,
    const float* __restrict__ Z,
    const float* __restrict__ phi,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    float a,
    int poly_degree,
    int m_tor,
    float* __restrict__ mat,
    float* __restrict__ rhs
) {
    extern __shared__ unsigned char shbuf[];
    int row = blockIdx.x;
    if (row >= gridDim.x) return;
    float* sh = reinterpret_cast<float*>(shbuf);
    int base_count = SEG_TILE * 6 + blockDim.x * 3;
    float* aux = sh + base_count;
    float* xpow = aux;
    float* zpow = xpow + (MAX_PSI_DEGREE + 1);
    float* cosv = zpow + (MAX_PSI_DEGREE + 1);
    float* sinv = cosv + (MAX_PSI_MTOR + 1);
    float* scalar = sinv + (MAX_PSI_MTOR + 1);

    float Ri = R[row];
    float Zi = Z[row];
    float phii = phi[row];

    float cp = cosf(phii);
    float sp = sinf(phii);
    float bx, by, bz;
    eval_B_block_f32(Ri * cp, Ri * sp, Zi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);

    if (threadIdx.x == 0) {
        float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
        float ra, za, rap, zap;
        periodic_hermite_uniform_f32(phii, axis_R, axis_R_phi, n_axis, period, ra, rap);
        periodic_hermite_uniform_f32(phii, axis_Z, axis_Z_phi, n_axis, period, za, zap);
        float X = (Ri - ra) / a;
        float Zc = (Zi - za) / a;
        float br = bx * cp + by * sp;
        float bphi = -bx * sp + by * cp;
        float cphi = bphi / Ri;
        float cR = br - cphi * rap;
        float cZ = bz - cphi * zap;
        xpow[0] = 1.0f;
        zpow[0] = 1.0f;
        for (int k = 1; k <= poly_degree; ++k) {
            xpow[k] = xpow[k - 1] * X;
            zpow[k] = zpow[k - 1] * Zc;
        }
        cosv[0] = 1.0f;
        sinv[0] = 0.0f;
        for (int m = 1; m <= m_tor; ++m) {
            float arg = static_cast<float>(m * nfp) * phii;
            cosv[m] = cosf(arg);
            sinv[m] = sinf(arg);
        }
        scalar[0] = cR;
        scalar[1] = cZ;
        scalar[2] = cphi;
        scalar[3] = -2.0f * X * cR / a;
    }
    __syncthreads();

    if (threadIdx.x == 0) rhs[row] = scalar[3];
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        int ax = mode_a[col];
        int bz_exp = mode_b[col];
        int m = mode_m[col];
        int kind = mode_kind[col];
        float mono = xpow[ax] * zpow[bz_exp];
        float mono_x = (ax == 0) ? 0.0f : static_cast<float>(ax) * xpow[ax - 1] * zpow[bz_exp];
        float mono_z = (bz_exp == 0) ? 0.0f : static_cast<float>(bz_exp) * xpow[ax] * zpow[bz_exp - 1];
        float spatial = (scalar[0] * mono_x + scalar[1] * mono_z) / a;
        float trig = (m == 0) ? 1.0f : ((kind == 0) ? cosv[m] : sinv[m]);
        float trig_phi = 0.0f;
        if (m != 0) {
            float fac = static_cast<float>(m * nfp);
            trig_phi = (kind == 0) ? (-fac * sinv[m]) : (fac * cosv[m]);
        }
        mat[static_cast<size_t>(row) * n_cols + col] = spatial * trig + scalar[2] * mono * trig_phi;
    }
}

} // namespace coil_eval::field
