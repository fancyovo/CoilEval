// Exact coarse-map and candidate-refinement batching for ordinary axis selection.
#include "pipeline/axis.cuh"
#include "pipeline/axis_coarse_cache.cuh"
#include "pipeline/axis_refinement_batch.cuh"
#include "pipeline/common.cuh"
#include "pipeline/policy.cuh"
#include "coil_field_internal.h"
#include "coil_eval_axis_batch.h"
#include <exception>
namespace coil_eval::pipeline {
thread_local sgpu_axis_coarse::Slot g_axis_coarse_slot;
std::vector<AxisCandidate> candidates_from_coarse_endpoints(
    const double* R0, const double* Z0, const double* R1, const double* Z1,
    int grid, int maximum, AxisDomain& domain
) {
    std::vector<double> rs(grid), zs(grid), dR(static_cast<size_t>(grid) * grid), dZ(dR.size());
    for (int i = 0; i < grid; ++i) { rs[i] = R0[i]; zs[i] = Z0[static_cast<size_t>(i) * grid]; }
    for (size_t i = 0; i < dR.size(); ++i) { dR[i] = R1[i] - R0[i]; dZ[i] = Z1[i] - Z0[i]; }
    domain = {rs.front(), rs.back(), zs.front(), zs.back()};
    return find_grid_candidates(rs, zs, dR, dZ, maximum);
}
} // namespace coil_eval::pipeline
using namespace coil_eval::pipeline;
extern "C" {
int sgpu_axis_coarse_grid(const double* x, const double* y, const double* z,
                         int nbase, int ncoeff, const SgpuScoreConfig* config,
                         double* R0, double* Z0, int capacity) {
    if (!x || !y || !z || !config || !R0 || !Z0 || nbase <= 0 || ncoeff < 3 ||
        config->abi_version != SGPU_SCORE_ABI_VERSION || config->struct_size != sizeof(SgpuScoreConfig) ||
        (ncoeff % 2) == 0 || config->axis_grid < 2 ||
        static_cast<std::int64_t>(config->axis_grid) * config->axis_grid > capacity ||
        !std::isfinite(config->axis_span) || !std::isfinite(config->axis_r_floor) ||
        config->axis_span <= 0.0 || config->axis_r_floor <= 0.0) {
        sgpu_internal_set_error("invalid axis coarse grid arguments");
        return 1;
    }
    const auto count = static_cast<size_t>(nbase) * ncoeff;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i])) {
            sgpu_internal_set_error("axis coarse grid requires finite coil coefficients");
            return 1;
        }
    }
    try {
        const auto domain = build_axis_domain(x, y, z, nbase, ncoeff, *config);
        const int grid = config->axis_grid;
        for (int j = 0; j < grid; ++j) {
            for (int i = 0; i < grid; ++i) {
                const size_t k = static_cast<size_t>(j) * grid + i;
                // Same operation order as search_axis_grid, including division.
                R0[k] = domain.r_min + (domain.r_max - domain.r_min) * i / std::max(grid - 1, 1);
                Z0[k] = domain.z_min + (domain.z_max - domain.z_min) * j / std::max(grid - 1, 1);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}

int sgpu_axis_coarse_cache_set(const double* x, const double* y, const double* z,
    const double* currents, int nbase, int ncoeff, int nfp,
    const SgpuScoreConfig* config, const double* R0, const double* Z0,
    const double* R1, const double* Z1, int count) {
    if (g_axis_coarse_slot.pending.available) {
        sgpu_internal_set_error("axis coarse cache already armed; consume or clear it explicitly");
        return 1;
    }
    if (!x || !y || !z || !currents || !config || !R0 || !Z0 || !R1 || !Z1 ||
        nbase <= 0 || ncoeff < 3 || (ncoeff % 2) == 0 || nfp <= 0 ||
        config->abi_version != SGPU_SCORE_ABI_VERSION || config->struct_size != sizeof(SgpuScoreConfig) ||
        config->device_id < 0 || config->segments_per_coil <= 0 || config->axis_grid < 2 ||
        config->axis_trace_steps <= 0 || config->axis_hint_enabled != 0 ||
        static_cast<std::int64_t>(config->axis_grid) * config->axis_grid != count) {
        sgpu_internal_set_error("invalid one-shot axis coarse cache arguments");
        return 1;
    }
    for (int i = 0; i < nbase; ++i) {
        if (!std::isfinite(currents[i])) {
            sgpu_internal_set_error("axis coarse cache requires finite currents");
            return 1;
        }
    }
    try {
        sgpu_axis_coarse::Entry entry;
        entry.R0.resize(count); entry.Z0.resize(count);
        if (sgpu_axis_coarse_grid(x, y, z, nbase, ncoeff, config,
                                  entry.R0.data(), entry.Z0.data(), count)) return 1;
        if (!sgpu_axis_coarse::Entry::equal(entry.R0, R0, count) ||
            !sgpu_axis_coarse::Entry::equal(entry.Z0, Z0, count)) {
            sgpu_internal_set_error("axis coarse cache starts differ from native grid");
            return 1;
        }
        const auto coefficients = static_cast<size_t>(nbase) * ncoeff;
        entry.x.assign(x, x + coefficients); entry.y.assign(y, y + coefficients);
        entry.z.assign(z, z + coefficients); entry.currents.assign(currents, currents + nbase);
        // Nonfinite endpoints are retained: the original coarse map may produce
        // escaped lines, and native candidate extraction must see the same data.
        entry.R1.assign(R1, R1 + count); entry.Z1.assign(Z1, Z1 + count);
        entry.nbase = nbase; entry.ncoeff = ncoeff; entry.nfp = nfp; entry.config = *config;
        if (!g_axis_coarse_slot.arm(std::move(entry))) return 1;
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}

int sgpu_axis_coarse_domain(const double* x, const double* y, const double* z,
    int nbase, int ncoeff, const SgpuScoreConfig* config, double* output) {
    if (!x || !y || !z || !config || !output || nbase <= 0 || ncoeff < 3 || (ncoeff % 2) == 0 ||
        config->abi_version != SGPU_SCORE_ABI_VERSION || config->struct_size != sizeof(SgpuScoreConfig) ||
        !std::isfinite(config->axis_span) || !std::isfinite(config->axis_r_floor) ||
        config->axis_span <= 0.0 || config->axis_r_floor <= 0.0) {
        sgpu_internal_set_error("invalid exact axis-domain arguments");
        return 1;
    }
    const size_t count = static_cast<size_t>(nbase) * ncoeff;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i])) {
            sgpu_internal_set_error("exact axis domain requires finite coil coefficients");
            return 1;
        }
    }
    try {
        const auto domain = build_axis_domain(x, y, z, nbase, ncoeff, *config);
        output[0] = domain.r_min; output[1] = domain.r_max;
        output[2] = domain.z_min; output[3] = domain.z_max;
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}

int sgpu_axis_coarse_cache_clear() { g_axis_coarse_slot.clear(); return 0; }
int sgpu_axis_coarse_cache_status() { return g_axis_coarse_slot.status; }

int sgpu_axis_refinement_batch(void* batch, const SgpuScoreConfig* configs,
    int queries, int nfp, const double* R0, const double* Z0, const double* R1,
    const double* Z1, const double* exact_domains, int grid, int capacity,
    double* payload, int* winding, int* counts) {
    if (!batch || !configs || !R0 || !Z0 || !R1 || !Z1 || !exact_domains || !payload || !winding || !counts ||
        queries <= 0 || queries != sgpu_internal_batch_query_count(batch) || nfp <= 0 || grid < 2 || capacity <= 0) {
        sgpu_internal_set_error("invalid batched axis refinement arguments");
        return 1;
    }
    try {
        std::vector<SgpuScoreConfig> cfg(configs, configs + queries);
        std::vector<AxisDomain> domains(queries);
        std::vector<std::vector<AxisCandidate>> candidates(queries);
        const size_t grid_size = static_cast<size_t>(grid) * grid;
        const size_t slots = static_cast<size_t>(queries) * capacity;
        std::fill(payload, payload + slots * 6, std::numeric_limits<double>::quiet_NaN());
        std::fill(winding, winding + slots, -1);
        std::fill(counts, counts + queries, 0);
        for (int q = 0; q < queries; ++q) {
            std::string reason;
            if (!validate_config(cfg[q], reason) || cfg[q].axis_grid != grid ||
                cfg[q].axis_max_candidates <= 0 || cfg[q].axis_max_candidates > capacity ||
                cfg[q].axis_newton_iters < 0 || cfg[q].axis_trace_steps <= 0 ||
                cfg[q].axis_trace_steps != cfg[0].axis_trace_steps || cfg[q].device_id != cfg[0].device_id ||
                cfg[q].segments_per_coil != cfg[0].segments_per_coil || cfg[q].axis_hint_enabled != 0) {
                sgpu_internal_set_error("incompatible batched axis refinement configuration");
                return 1;
            }
            const size_t offset = static_cast<size_t>(q) * grid_size;
            candidates[q] = candidates_from_coarse_endpoints(R0 + offset, Z0 + offset,
                R1 + offset, Z1 + offset, grid, cfg[q].axis_max_candidates, domains[q]);
            // A reconstructed raster endpoint can differ by one ULP from the
            // original domain boundary. Newton uses the original boundary.
            const double* bounds = exact_domains + static_cast<size_t>(q) * 4;
            if (!(std::isfinite(bounds[0]) && std::isfinite(bounds[1]) && std::isfinite(bounds[2]) &&
                  std::isfinite(bounds[3]) && bounds[1] > bounds[0] && bounds[3] > bounds[2])) {
                sgpu_internal_set_error("invalid exact domain for batched axis refinement");
                return 1;
            }
            domains[q] = {bounds[0], bounds[1], bounds[2], bounds[3]};
            counts[q] = static_cast<int>(candidates[q].size());
            for (size_t i = 0; i < candidates[q].size(); ++i) {
                const size_t slot = static_cast<size_t>(q) * capacity + i;
                payload[6 * slot] = candidates[q][i].R;
                payload[6 * slot + 1] = candidates[q][i].Z;
                payload[6 * slot + 2] = candidates[q][i].residual;
                winding[slot] = candidates[q][i].winding ? 1 : 0;
            }
        }
        auto trace = [&](const std::vector<std::vector<double>>& r,
                         const std::vector<std::vector<double>>& z,
                         std::vector<std::vector<double>>& re,
                         std::vector<std::vector<double>>& ze) {
            size_t lines = 0;
            for (const auto& query : r) lines = std::max(lines, query.size());
            re.resize(queries); ze.resize(queries);
            if (!lines) return true;
            std::vector<double> flat_r(static_cast<size_t>(queries) * lines), flat_z(flat_r.size());
            std::vector<double> flat_re(flat_r.size()), flat_ze(flat_r.size());
            for (int q = 0; q < queries; ++q) {
                const size_t offset = static_cast<size_t>(q) * lines;
                std::fill(flat_r.begin() + offset, flat_r.begin() + offset + lines,
                          .5 * (domains[q].r_min + domains[q].r_max));
                std::fill(flat_z.begin() + offset, flat_z.begin() + offset + lines,
                          .5 * (domains[q].z_min + domains[q].z_max));
                std::copy(r[q].begin(), r[q].end(), flat_r.begin() + offset);
                std::copy(z[q].begin(), z[q].end(), flat_z.begin() + offset);
            }
            if (sgpu_batch_trace_period_mixed(batch, flat_r.data(), flat_z.data(), flat_re.data(),
                    flat_ze.data(), static_cast<int>(lines), nfp, cfg[0].axis_trace_steps)) return false;
            for (int q = 0; q < queries; ++q) {
                const size_t offset = static_cast<size_t>(q) * lines;
                re[q].assign(flat_re.begin() + offset, flat_re.begin() + offset + r[q].size());
                ze[q].assign(flat_ze.begin() + offset, flat_ze.begin() + offset + z[q].size());
            }
            return true;
        };
        if (!sgpu_axis_refinement::refine(candidates, domains, cfg, trace)) return 1;
        for (int q = 0; q < queries; ++q) {
            for (size_t i = 0; i < candidates[q].size(); ++i) {
                const size_t slot = static_cast<size_t>(q) * capacity + i;
                payload[6 * slot + 3] = candidates[q][i].R;
                payload[6 * slot + 4] = candidates[q][i].Z;
                payload[6 * slot + 5] = candidates[q][i].residual;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}

int sgpu_axis_refinement_cache_set(const SgpuScoreConfig* config,
                                  const double* payload, const int* winding, int count) {
    auto& entry = g_axis_coarse_slot.pending;
    if (!entry.available || entry.refinement_available || !config || count < 0 ||
        (count && (!payload || !winding)) || !entry.coarse_config_matches(*config) ||
        !entry.refinement_config_matches(*config) ||
        config->abi_version != SGPU_SCORE_ABI_VERSION || config->struct_size != sizeof(SgpuScoreConfig)) {
        sgpu_internal_set_error("axis refinement requires a compatible pending coarse cache");
        return 1;
    }
    try {
        AxisDomain domain;
        const auto seeds = candidates_from_coarse_endpoints(entry.R0.data(), entry.Z0.data(),
            entry.R1.data(), entry.Z1.data(), entry.config.axis_grid, entry.config.axis_max_candidates, domain);
        bool matches = seeds.size() == static_cast<size_t>(count);
        for (size_t i = 0; matches && i < seeds.size(); ++i) {
            const double row[] = {seeds[i].R, seeds[i].Z, seeds[i].residual};
            matches = std::memcmp(row, payload + 6 * i, 3 * sizeof(double)) == 0 &&
                winding[i] == int(seeds[i].winding);
        }
        if (!matches) {
            sgpu_internal_set_error("axis refinement payload differs from native coarse candidates");
            return 1;
        }
        entry.refinement.clear(); entry.winding.clear();
        if (count) {
            entry.refinement.assign(payload, payload + static_cast<size_t>(count) * 6);
            entry.winding.assign(winding, winding + count);
        }
        entry.refinement_available = true;
        g_axis_coarse_slot.refinement_status = 1;
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}

int sgpu_axis_refinement_cache_status() { return g_axis_coarse_slot.refinement_status; }

int sgpu_axis_refinement_reference(void* field, const SgpuScoreConfig* config,
    int nfp, const double* domain, const double* payload, int count, double* refined) {
    if (!field || !config || !domain || nfp <= 0 || count < 0 ||
        (count && (!payload || !refined)) || config->abi_version != SGPU_SCORE_ABI_VERSION ||
        config->struct_size != sizeof(SgpuScoreConfig)) {
        sgpu_internal_set_error("invalid reference axis refinement input");
        return 1;
    }
    try {
        std::vector<AxisCandidate> candidates(count);
        for (int i = 0; i < count; ++i) {
            candidates[i].R = payload[6 * i]; candidates[i].Z = payload[6 * i + 1];
            candidates[i].residual = payload[6 * i + 2];
        }
        const AxisDomain bounds{domain[0], domain[1], domain[2], domain[3]};
        if (!refine_axis_candidates(field, candidates, bounds, nfp, *config, config->axis_newton_iters)) return 1;
        for (int i = 0; i < count; ++i) {
            refined[3 * i] = candidates[i].R; refined[3 * i + 1] = candidates[i].Z;
            refined[3 * i + 2] = candidates[i].residual;
        }
        return 0;
    } catch (const std::exception& error) {
        sgpu_internal_set_error(error.what());
        return 1;
    }
}


} // extern "C"
