#include "field/state.h"
#include "coil_field.h"
using namespace coil_eval::field;
extern "C" int sgpu_copy_psi_warm_preconditioner(double* upper, double* scales, int count) {
    if (!upper || !scales || !sgpu_has_psi_warm_preconditioner(count)) {
        set_error("invalid psi preconditioner export");
        return 1;
    }
    for (int row = 0; row < count; ++row) {
        scales[row] = g_psi_warm_preconditioner.center_scale[row];
        for (int column = 0; column < count; ++column) {
            upper[static_cast<size_t>(row) * count + column] = row <= column
                ? g_psi_warm_preconditioner.upper_factor[static_cast<size_t>(column) * count + row]
                : 0.0;
        }
    }
    set_error("");
    return 0;
}
