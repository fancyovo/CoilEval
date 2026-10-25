// CPU-only regression for ownership and exact physical/config/grid identity.
#include "pipeline/axis_coarse_cache.cuh"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace sgpu_axis_coarse;
    SgpuScoreConfig cfg{};
    cfg.device_id = 0; cfg.segments_per_coil = 256; cfg.axis_grid = 8;
    cfg.axis_trace_steps = 240; cfg.axis_span = .5; cfg.axis_r_floor = 1e-4;
    Entry entry;
    entry.available = true; entry.nbase = 1; entry.ncoeff = 3; entry.nfp = 4;
    entry.config = cfg;
    entry.x = {1., .2, .3}; entry.y = {.1, .2, .3}; entry.z = {0., .2, .3};
    entry.currents = {1e5}; entry.R0 = {1., 2.}; entry.Z0 = {0., .1};
    entry.R1 = {1.1, 2.1}; entry.Z1 = {0.01, .11};
    auto matches = [&](const Entry& e, const SgpuScoreConfig& c) {
        return e.matches(entry.x.data(), entry.y.data(), entry.z.data(), entry.currents.data(), 1, 3, 4, &c);
    };
    assert(matches(entry, cfg));
    auto changed = cfg;
    changed.axis_trace_steps += 1; assert(!matches(entry, changed));
    changed = cfg; changed.axis_grid += 1; assert(!matches(entry, changed));
    changed = cfg; changed.axis_span = .6; assert(!matches(entry, changed));
    changed = cfg; changed.axis_r_floor = 1e-5; assert(!matches(entry, changed));
    changed = cfg; changed.device_id = 1; assert(!matches(entry, changed));
    changed = cfg; changed.segments_per_coil = 128; assert(!matches(entry, changed));
    changed = cfg; changed.axis_hint_enabled = 1; assert(!matches(entry, changed));
    // Downstream controls do not change the coarse field-line map.
    changed = cfg; changed.target_volume = .2; assert(matches(entry, changed));
    auto altered = entry;
    altered.currents[0] = std::nextafter(altered.currents[0], 0.);
    assert(!matches(altered, cfg));
    altered = entry; altered.z[0] = -0.; assert(!matches(altered, cfg));
    altered = entry; altered.nfp = 3; assert(!matches(altered, cfg));
    assert(entry.grid_matches(entry.R0, entry.Z0));
    auto shifted = entry.R0; shifted[0] = std::nextafter(shifted[0], 2.);
    assert(!entry.grid_matches(shifted, entry.Z0));
    Slot slot;
    assert(slot.arm(entry)); assert(slot.status == 1);
    assert(!slot.arm(altered));  // Cannot silently overwrite a pending query.
    auto owned = slot.take();
    assert(slot.status == 4 && !slot.pending.available && owned.available);
    assert(!slot.take().available);  // An early scoring failure cannot reuse it.
    assert(slot.arm(entry));
    owned.x[0] = 999.; assert(matches(slot.pending, cfg));  // Independent copies.
    slot.clear(); assert(slot.status == 0 && !slot.pending.available);
    entry.config.axis_max_candidates = 64; entry.config.axis_newton_iters = 6;
    entry.config.axis_tolerance = 1e-7; entry.config.axis_fd_absolute = 1e-6;
    entry.config.axis_fd_relative = 1e-4;
    entry.refinement_available = true;
    cfg = entry.config;
    assert(matches(entry, cfg));
    changed = cfg; changed.axis_max_candidates += 1; assert(!matches(entry, changed));
    changed = cfg; changed.axis_newton_iters += 1; assert(!matches(entry, changed));
    changed = cfg; changed.axis_tolerance *= 2.; assert(!matches(entry, changed));
    changed = cfg; changed.axis_fd_absolute *= 2.; assert(!matches(entry, changed));
    changed = cfg; changed.axis_fd_relative *= 2.; assert(!matches(entry, changed));
    assert(slot.arm(entry));
    slot.refinement_status = 1;
    owned = slot.take();
    assert(owned.refinement_available && slot.refinement_status == 4 && !slot.pending.available);
    slot.clear(); assert(slot.refinement_status == 0);
    Entry full16 = entry;
    full16.nbase = 4; full16.ncoeff = 33;
    full16.x.resize(132, .01); full16.y.resize(132, .02); full16.z.resize(132, .03);
    full16.currents.resize(4, 1e5);
    assert(full16.matches(full16.x.data(), full16.y.data(), full16.z.data(),
                          full16.currents.data(), 4, 33, 4, &cfg));
    auto shifted16 = full16.x;
    shifted16.back() = std::nextafter(shifted16.back(), 1.);
    assert(!full16.matches(shifted16.data(), full16.y.data(), full16.z.data(),
                           full16.currents.data(), 4, 33, 4, &cfg));
    std::cout << "axis coarse cache CPU ownership/key contracts passed\n";
}
