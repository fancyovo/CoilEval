#include "field/state.h"
#include "coil_field.h"
#include "coil_field_internal.h"


namespace coil_eval::field {

// The only definitions: lifetime and thread affinity match the original TU.
thread_local std::string g_last_error;
thread_local PsiWarmPreconditionerCache g_psi_warm_preconditioner;

void set_error(const char* msg) { g_last_error = msg ? msg : ""; }

void set_error(const std::string& msg) { g_last_error = msg; }

int cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        set_error(std::string(where) + ": " + cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int cublas_check(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_error(std::string(where) + ": cublas status " + std::to_string(static_cast<int>(status)));
        return 1;
    }
    return 0;
}

int cusolver_check(cusolverStatus_t status, const char* where) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        set_error(std::string(where) + ": cusolver status " + std::to_string(static_cast<int>(status)));
        return 1;
    }
    return 0;
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_set_psi_warm_preconditioner_capture(int enabled) {
    g_psi_warm_preconditioner.capture_enabled = enabled != 0;
    if (enabled) {
        g_psi_warm_preconditioner.coefficient_count = 0;
        g_psi_warm_preconditioner.upper_factor.clear();
        g_psi_warm_preconditioner.center_scale.clear();
    }
    set_error("");
    return 0;
}

int sgpu_has_psi_warm_preconditioner(int coefficient_count) {
    return coefficient_count > 0 &&
        g_psi_warm_preconditioner.coefficient_count == coefficient_count &&
        g_psi_warm_preconditioner.upper_factor.size() ==
            static_cast<size_t>(coefficient_count) * coefficient_count &&
        g_psi_warm_preconditioner.center_scale.size() == static_cast<size_t>(coefficient_count);
}

void sgpu_clear_psi_warm_preconditioner() {
    g_psi_warm_preconditioner = PsiWarmPreconditionerCache{};
}

void sgpu_internal_set_error(const char* message) {
    set_error(message);
}

const char* sgpu_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
