#pragma once

// CPU-only ownership/key contract. No CUDA or solver state is retained here.
#include "coil_field.h"
#include <cstring>
#include <utility>
#include <vector>

namespace sgpu_axis_coarse {

struct Entry {
    bool available = false;
    bool refinement_available = false;
    int nbase = 0, ncoeff = 0, nfp = 0;
    SgpuScoreConfig config{};
    std::vector<double> x, y, z, currents, R0, Z0, R1, Z1;
    // Packed seed R/Z/residual and refined R/Z/residual, six doubles per row.
    std::vector<double> refinement;
    std::vector<int> winding;

    bool coarse_config_matches(const SgpuScoreConfig& cfg) const {
        return cfg.device_id == config.device_id && cfg.segments_per_coil == config.segments_per_coil &&
            cfg.axis_grid == config.axis_grid && cfg.axis_trace_steps == config.axis_trace_steps &&
            cfg.axis_span == config.axis_span && cfg.axis_r_floor == config.axis_r_floor && cfg.axis_hint_enabled == 0;
    }

    bool refinement_config_matches(const SgpuScoreConfig& cfg) const {
        return cfg.axis_max_candidates == config.axis_max_candidates &&
            cfg.axis_newton_iters == config.axis_newton_iters &&
            cfg.axis_tolerance == config.axis_tolerance &&
            cfg.axis_fd_absolute == config.axis_fd_absolute && cfg.axis_fd_relative == config.axis_fd_relative;
    }

    static bool equal(const std::vector<double>& saved, const double* values, std::size_t count) {
        return values && saved.size() == count &&
            std::memcmp(saved.data(), values, count * sizeof(double)) == 0;
    }

    bool matches(const double* cx, const double* cy, const double* cz, const double* ca,
                 int bases, int coefficients, int periods, const SgpuScoreConfig* cfg) const {
        if (!available || !cfg || bases != nbase || coefficients != ncoeff || periods != nfp) return false;
        // These are exactly the coarse map/domain controls, not downstream
        // score controls. Hints are excluded so the ordinary selector always runs.
        if (!coarse_config_matches(*cfg)) return false;
        if (refinement_available && !refinement_config_matches(*cfg)) return false;
        const auto count = static_cast<std::size_t>(nbase) * ncoeff;
        return equal(x, cx, count) && equal(y, cy, count) && equal(z, cz, count) &&
            equal(currents, ca, static_cast<std::size_t>(nbase));
    }

    bool grid_matches(const std::vector<double>& r, const std::vector<double>& z0) const {
        return available && equal(R0, r.data(), r.size()) && equal(Z0, z0.data(), z0.size());
    }
};

struct Slot {
    Entry pending;
    // 0 idle, 1 armed, 2 consumed, 3 rejected key/grid, 4 unused by scoring.
    int status = 0;
    int refinement_status = 0;
    bool arm(Entry entry) {
        if (pending.available) return false;
        entry.available = true;
        pending = std::move(entry);
        status = 1;
        refinement_status = 0;
        return true;
    }
    Entry take() {
        Entry entry = std::move(pending);
        pending = Entry{};
        status = entry.available ? 4 : 0;
        refinement_status = entry.refinement_available ? 4 : 0;
        return entry;
    }
    void clear() { pending = Entry{}; status = 0; refinement_status = 0; }
};

}  // namespace sgpu_axis_coarse

namespace coil_eval::pipeline {
extern thread_local sgpu_axis_coarse::Slot g_axis_coarse_slot;
}
