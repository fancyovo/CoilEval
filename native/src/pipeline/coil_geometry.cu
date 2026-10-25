// Sampled physical coil geometry and its analytical coefficient derivatives.
#include "pipeline/types.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/coil_distance.cuh"
#include "pipeline/common.cuh"

namespace coil_eval::pipeline {

namespace {

struct GeometrySample {
    Vec3d point;
    Vec3d first;
    Vec3d second;
    double curvature;
};

struct FullGeometryPoint {
    Vec3d point;
    int coil;
    int sample;
    int reflected;
    int period;
    int label;
};

}  // namespace


void eval_fourier_interleaved(
    const double* coeff,
    int n_coeff,
    double t,
    double& value,
    double& derivative,
    double& second_derivative
) {
    const int order = (n_coeff - 1) / 2;
    value = coeff[0];
    derivative = 0.0;
    second_derivative = 0.0;
    for (int mode = 1; mode <= order; ++mode) {
        const double omega = TWOPI * static_cast<double>(mode);
        const double argument = omega * t;
        const double sine = std::sin(argument);
        const double cosine = std::cos(argument);
        const double sine_coeff = coeff[2 * mode - 1];
        const double cosine_coeff = coeff[2 * mode];
        value += sine_coeff * sine + cosine_coeff * cosine;
        derivative += omega * (sine_coeff * cosine - cosine_coeff * sine);
        second_derivative -= omega * omega * (sine_coeff * sine + cosine_coeff * cosine);
    }
}

Vec3d rotate_z(Vec3d point, double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {
        cosine * point.x - sine * point.y,
        sine * point.x + cosine * point.y,
        point.z,
    };
}

CoilMetrics compute_coil_metrics(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp
) {
    constexpr int samples = 160;
    const int order = (n_coeff - 1) / 2;
    std::vector<std::vector<Vec3d>> base_points(n_base_coils);
    std::vector<double> lengths(n_base_coils, 0.0);
    std::vector<double> curvatures;
    curvatures.reserve(static_cast<size_t>(n_base_coils) * samples);
    for (int coil = 0; coil < n_base_coils; ++coil) {
        auto& points = base_points[coil];
        points.reserve(samples);
        for (int sample = 0; sample < samples; ++sample) {
            const double t = static_cast<double>(sample) / samples;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            points.push_back({x, y, z});
            const double speed2 = dx * dx + dy * dy + dz * dz;
            const double speed = std::sqrt(speed2);
            lengths[coil] += speed / samples;
            const double cx = dy * ddz - dz * ddy;
            const double cy = dz * ddx - dx * ddz;
            const double cz = dx * ddy - dy * ddx;
            curvatures.push_back(std::sqrt(cx * cx + cy * cy + cz * cz) /
                                 std::max(speed2 * speed, 1.0e-30));
        }
    }

    struct LabelledPoint {
        Vec3d point;
        int label;
    };
    std::vector<LabelledPoint> full_points;
    full_points.reserve(static_cast<size_t>(n_base_coils) * 2 * nfp * samples);
    int label = 0;
    for (const auto& base : base_points) {
        for (int reflected = 0; reflected < 2; ++reflected) {
            for (int period = 0; period < nfp; ++period) {
                const double angle = TWOPI * static_cast<double>(period) / nfp;
                for (Vec3d point : base) {
                    if (reflected) {
                        point.y = -point.y;
                        point.z = -point.z;
                    }
                    full_points.push_back({rotate_z(point, angle), label});
                }
                ++label;
            }
        }
    }

    CoilMetrics metrics;
    metrics.length_mean = std::accumulate(lengths.begin(), lengths.end(), 0.0) /
                          std::max(static_cast<int>(lengths.size()), 1);
    metrics.curvature_p95 = percentile(curvatures, 0.95);
    metrics.curvature_max = *std::max_element(curvatures.begin(), curvatures.end());
    for (size_t i = 0; i < full_points.size(); ++i) {
        const Vec3d& p = full_points[i].point;
        metrics.min_axis_distance = std::min(metrics.min_axis_distance, std::hypot(p.x, p.y));
    }
    metrics.min_intercoil_distance = geometry_detail::closest_distinct_label_pair(full_points).distance;
    double total_mode_energy = 0.0;
    double high_mode_energy = 0.0;
    const int high_start = std::max(1, static_cast<int>(std::floor(0.6 * order)));
    for (const double* block : {coeffs_x, coeffs_y, coeffs_z}) {
        for (int coil = 0; coil < n_base_coils; ++coil) {
            const double* coeff = block + static_cast<size_t>(coil) * n_coeff;
            for (int mode = 1; mode <= order; ++mode) {
                const double energy = coeff[2 * mode - 1] * coeff[2 * mode - 1] +
                                      coeff[2 * mode] * coeff[2 * mode];
                total_mode_energy += energy;
                if (mode >= high_start) high_mode_energy += energy;
            }
        }
    }
    metrics.high_mode_fraction = total_mode_energy > 0.0 ? high_mode_energy / total_mode_energy : 0.0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        metrics.current_abs_max = std::max(metrics.current_abs_max, std::abs(currents_a[coil]));
    }
    return metrics;
}

double coil_component(const CoilMetrics& metrics) {
    const double length = q_down(metrics.length_mean, 7.0, 1.4, 0.6);
    const double curvature_p95 = q_down(
        metrics.curvature_p95, COIL_CURVATURE_P95_SCALE, 1.3, 0.5);
    const double curvature_max = q_down(metrics.curvature_max, 35.0, 1.2, 0.5);
    const double spacing = q_up(metrics.min_intercoil_distance, 0.08, 1.1, 0.45);
    const double axis_distance = q_up(metrics.min_axis_distance, 0.20, 1.2, 0.45);
    const double high_mode = q_down(metrics.high_mode_fraction, 0.05, 1.0, 0.7);
    const double current = q_down(metrics.current_abs_max, 2.0e6, 1.0, 0.7);
    return blend({
        {0.16, length}, {0.20, curvature_p95}, {0.12, curvature_max},
        {0.20, spacing}, {0.12, axis_distance}, {0.13, high_mode}, {0.07, current},
    });
}

Vec3d vec_sub(Vec3d left, Vec3d right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

double vec_dot(Vec3d left, Vec3d right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3d vec_cross(Vec3d left, Vec3d right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double vec_norm(Vec3d value) {
    return std::sqrt(std::max(vec_dot(value, value), 0.0));
}

void fourier_basis(int n_coeff, double t, int coefficient, double& value, double& first, double& second) {
    value = 0.0;
    first = 0.0;
    second = 0.0;
    if (coefficient == 0) {
        value = 1.0;
        return;
    }
    const int mode = (coefficient + 1) / 2;
    const double omega = TWOPI * mode;
    const double argument = omega * t;
    if (coefficient % 2 == 1) {
        value = std::sin(argument);
        first = omega * std::cos(argument);
        second = -omega * omega * std::sin(argument);
    } else {
        value = std::cos(argument);
        first = -omega * std::sin(argument);
        second = -omega * omega * std::cos(argument);
    }
}

Vec3d coordinate_vector(int coordinate, double value) {
    if (coordinate == 0) return {value, 0.0, 0.0};
    if (coordinate == 1) return {0.0, value, 0.0};
    return {0.0, 0.0, value};
}

double curvature_directional_derivative(
    const GeometrySample& sample,
    Vec3d d_first,
    Vec3d d_second
) {
    const double speed = vec_norm(sample.first);
    const Vec3d cross = vec_cross(sample.first, sample.second);
    const double cross_norm = vec_norm(cross);
    if (speed <= 1.0e-15 || cross_norm <= 1.0e-15) return 0.0;
    const Vec3d d_cross = vec_sub(
        vec_cross(d_first, sample.second),
        vec_cross(d_second, sample.first)
    );
    const double d_cross_norm = vec_dot(cross, d_cross) / cross_norm;
    const double d_speed = vec_dot(sample.first, d_first) / speed;
    return d_cross_norm / (speed * speed * speed) -
        3.0 * cross_norm * d_speed / (speed * speed * speed * speed);
}

Vec3d transformed_basis_vector(
    int coordinate,
    double basis,
    int reflected,
    int period,
    int nfp
) {
    Vec3d value = coordinate_vector(coordinate, basis);
    if (reflected) {
        value.y = -value.y;
        value.z = -value.z;
    }
    return rotate_z(value, TWOPI * static_cast<double>(period) / nfp);
}

double q_down_derivative(double value, double scale, double power) {
    if (!std::isfinite(value) || !(value > 0.0) || !(scale > 0.0)) return 0.0;
    const double x = value / scale;
    const double xp = std::pow(x, power);
    return -power * std::pow(x, power - 1.0) / (scale * (1.0 + xp) * (1.0 + xp));
}

double q_up_derivative(double value, double scale, double power) {
    if (!std::isfinite(value) || !(value > 0.0) || !(scale > 0.0)) return 0.0;
    const double ratio = std::pow(scale / value, power);
    return power * ratio / (value * (1.0 + ratio) * (1.0 + ratio));
}

bool compute_coil_component_gradient_impl(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    CoilComponentGradient& output,
    std::string& error,
    CoilMetricDerivativeOutput* metrics_out
) {
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a ||
        n_base_coils <= 0 || n_coeff < 3 || (n_coeff % 2) == 0 || nfp <= 0) {
        error = "invalid coil-component gradient input";
        return false;
    }
    constexpr int samples = 160;
    const int parameter_count = n_base_coils * n_coeff;
    const int sample_count = n_base_coils * samples;
    std::vector<GeometrySample> geometry(sample_count);
    std::vector<double> curvatures(sample_count);
    std::vector<double> metric_curvatures(sample_count);
    std::vector<double> lengths(n_base_coils, 0.0);
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int sample_index = 0; sample_index < samples; ++sample_index) {
            const double t = static_cast<double>(sample_index) / samples;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            GeometrySample item{{x, y, z}, {dx, dy, dz}, {ddx, ddy, ddz}, 0.0};
            const double speed = vec_norm(item.first);
            item.curvature = vec_norm(vec_cross(item.first, item.second)) /
                std::max(speed * speed * speed, 1.0e-30);
            const int flat = coil * samples + sample_index;
            geometry[flat] = item;
            curvatures[flat] = item.curvature;
            // Values and active curvature derivatives historically use these
            // two denominator evaluation orders. Preserve both roundings.
            const double speed2 = dx * dx + dy * dy + dz * dz;
            const double cx = dy * ddz - dz * ddy;
            const double cy = dz * ddx - dx * ddz;
            const double cz = dx * ddy - dy * ddx;
            metric_curvatures[flat] = std::sqrt(cx * cx + cy * cy + cz * cz) /
                std::max(speed2 * std::sqrt(speed2), 1.0e-30);
            lengths[coil] += speed / samples;
        }
    }

    std::vector<int> curvature_order(sample_count);
    std::iota(curvature_order.begin(), curvature_order.end(), 0);
    std::sort(curvature_order.begin(), curvature_order.end(), [&](int left, int right) {
        return curvatures[left] < curvatures[right];
    });
    const double percentile_position = 0.95 * (sample_count - 1);
    const int percentile_lo = static_cast<int>(std::floor(percentile_position));
    const int percentile_hi = static_cast<int>(std::ceil(percentile_position));
    const double percentile_alpha = percentile_position - percentile_lo;
    std::vector<double> percentile_weights(sample_count, 0.0);
    percentile_weights[curvature_order[percentile_lo]] += 1.0 - percentile_alpha;
    percentile_weights[curvature_order[percentile_hi]] += percentile_alpha;
    const int maximum_curvature_index = static_cast<int>(
        std::max_element(curvatures.begin(), curvatures.end()) - curvatures.begin()
    );

    std::vector<FullGeometryPoint> full_points;
    full_points.reserve(static_cast<size_t>(n_base_coils) * 2 * nfp * samples);
    int label = 0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int reflected = 0; reflected < 2; ++reflected) {
            for (int period = 0; period < nfp; ++period) {
                for (int sample_index = 0; sample_index < samples; ++sample_index) {
                    Vec3d point = geometry[coil * samples + sample_index].point;
                    if (reflected) {
                        point.y = -point.y;
                        point.z = -point.z;
                    }
                    full_points.push_back({
                        rotate_z(point, TWOPI * static_cast<double>(period) / nfp),
                        coil, sample_index, reflected, period, label,
                    });
                }
                ++label;
            }
        }
    }
    size_t radius_point = 0;
    double minimum_radius = std::numeric_limits<double>::infinity();
    for (size_t left = 0; left < full_points.size(); ++left) {
        const Vec3d point = full_points[left].point;
        const double radius = std::hypot(point.x, point.y);
        if (radius < minimum_radius) {
            minimum_radius = radius;
            radius_point = left;
        }
    }
    const auto spacing = geometry_detail::closest_distinct_label_pair(full_points);
    const double minimum_spacing = spacing.distance;
    CoilMetrics metrics;
    metrics.length_mean = std::accumulate(lengths.begin(), lengths.end(), 0.0) /
                          std::max(n_base_coils, 1);
    metrics.curvature_p95 = percentile(metric_curvatures, 0.95);
    metrics.curvature_max = *std::max_element(metric_curvatures.begin(), metric_curvatures.end());
    metrics.min_intercoil_distance = minimum_spacing;
    metrics.min_axis_distance = minimum_radius;
    for (int coil = 0; coil < n_base_coils; ++coil)
        metrics.current_abs_max = std::max(metrics.current_abs_max, std::abs(currents_a[coil]));
    output.x.assign(parameter_count, 0.0);
    output.y.assign(parameter_count, 0.0);
    output.z.assign(parameter_count, 0.0);
    output.current.assign(n_base_coils, 0.0);

    std::vector<double> d_length[3];
    std::vector<double> d_p95[3];
    std::vector<double> d_max[3];
    std::vector<double> d_spacing[3];
    std::vector<double> d_radius[3];
    std::vector<double> d_high_mode[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        d_length[coordinate].assign(parameter_count, 0.0);
        d_p95[coordinate].assign(parameter_count, 0.0);
        d_max[coordinate].assign(parameter_count, 0.0);
        d_spacing[coordinate].assign(parameter_count, 0.0);
        d_radius[coordinate].assign(parameter_count, 0.0);
        d_high_mode[coordinate].assign(parameter_count, 0.0);
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
            const int parameter = coil * n_coeff + coefficient;
            for (int sample_index = 0; sample_index < samples; ++sample_index) {
                const int flat = coil * samples + sample_index;
                const GeometrySample& item = geometry[flat];
                double basis, first, second;
                fourier_basis(n_coeff, static_cast<double>(sample_index) / samples, coefficient, basis, first, second);
                for (int coordinate = 0; coordinate < 3; ++coordinate) {
                    const Vec3d d_first = coordinate_vector(coordinate, first);
                    const Vec3d d_second = coordinate_vector(coordinate, second);
                    const double speed = vec_norm(item.first);
                    if (speed > 1.0e-15) {
                        d_length[coordinate][parameter] +=
                            vec_dot(item.first, d_first) /
                            (speed * samples * n_base_coils);
                    }
                    if (percentile_weights[flat] != 0.0 || flat == maximum_curvature_index) {
                        const double d_curvature = curvature_directional_derivative(item, d_first, d_second);
                        d_p95[coordinate][parameter] += percentile_weights[flat] * d_curvature;
                        if (flat == maximum_curvature_index) d_max[coordinate][parameter] = d_curvature;
                    }
                }
            }
        }
    }

    const FullGeometryPoint& spacing_a = full_points[spacing.left];
    const FullGeometryPoint& spacing_b = full_points[spacing.right];
    const Vec3d spacing_delta = vec_sub(spacing_a.point, spacing_b.point);
    const FullGeometryPoint& radial = full_points[radius_point];
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
            const int parameter = coil * n_coeff + coefficient;
            double basis_a = 0.0, unused_first = 0.0, unused_second = 0.0;
            double basis_b = 0.0;
            double basis_r = 0.0;
            if (coil == spacing_a.coil) {
                fourier_basis(n_coeff, static_cast<double>(spacing_a.sample) / samples, coefficient, basis_a, unused_first, unused_second);
            }
            if (coil == spacing_b.coil) {
                fourier_basis(n_coeff, static_cast<double>(spacing_b.sample) / samples, coefficient, basis_b, unused_first, unused_second);
            }
            if (coil == radial.coil) {
                fourier_basis(n_coeff, static_cast<double>(radial.sample) / samples, coefficient, basis_r, unused_first, unused_second);
            }
            for (int coordinate = 0; coordinate < 3; ++coordinate) {
                Vec3d d_a{0.0, 0.0, 0.0};
                Vec3d d_b{0.0, 0.0, 0.0};
                if (coil == spacing_a.coil) {
                    d_a = transformed_basis_vector(
                        coordinate, basis_a, spacing_a.reflected, spacing_a.period, nfp
                    );
                }
                if (coil == spacing_b.coil) {
                    d_b = transformed_basis_vector(
                        coordinate, basis_b, spacing_b.reflected, spacing_b.period, nfp
                    );
                }
                if (minimum_spacing > 1.0e-15) {
                    d_spacing[coordinate][parameter] =
                        vec_dot(spacing_delta, vec_sub(d_a, d_b)) / minimum_spacing;
                }
                if (coil == radial.coil && minimum_radius > 1.0e-15) {
                    const Vec3d d_point = transformed_basis_vector(
                        coordinate, basis_r, radial.reflected, radial.period, nfp
                    );
                    d_radius[coordinate][parameter] =
                        (radial.point.x * d_point.x + radial.point.y * d_point.y) /
                        minimum_radius;
                }
            }
        }
    }

    const int order = (n_coeff - 1) / 2;
    const int high_start = std::max(1, static_cast<int>(std::floor(0.6 * order)));
    double total_energy = 0.0;
    double high_energy = 0.0;
    const double* blocks[] = {coeffs_x, coeffs_y, coeffs_z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int coil = 0; coil < n_base_coils; ++coil) {
            const double* coefficients = blocks[coordinate] + static_cast<size_t>(coil) * n_coeff;
            for (int mode = 1; mode <= order; ++mode) {
                const double energy = coefficients[2 * mode - 1] * coefficients[2 * mode - 1] +
                    coefficients[2 * mode] * coefficients[2 * mode];
                total_energy += energy;
                if (mode >= high_start) high_energy += energy;
            }
        }
    }
    metrics.high_mode_fraction = total_energy > 0.0 ? high_energy / total_energy : 0.0;
    output.value = 100.0 * coil_component(metrics);
    if (total_energy > 1.0e-30) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            for (int coil = 0; coil < n_base_coils; ++coil) {
                const double* coefficients = blocks[coordinate] + static_cast<size_t>(coil) * n_coeff;
                for (int coefficient = 1; coefficient < n_coeff; ++coefficient) {
                    const int mode = (coefficient + 1) / 2;
                    const double d_total = 2.0 * coefficients[coefficient];
                    const double d_high = mode >= high_start ? d_total : 0.0;
                    d_high_mode[coordinate][coil * n_coeff + coefficient] =
                        (d_high * total_energy - high_energy * d_total) /
                        (total_energy * total_energy);
                }
            }
        }
    }

    const double metric_derivatives[] = {
        0.16 * q_down_derivative(metrics.length_mean, 7.0, 1.4),
        0.20 * q_down_derivative(
            metrics.curvature_p95, COIL_CURVATURE_P95_SCALE, 1.3),
        0.12 * q_down_derivative(metrics.curvature_max, 35.0, 1.2),
        0.20 * q_up_derivative(metrics.min_intercoil_distance, 0.08, 1.1),
        0.12 * q_up_derivative(metrics.min_axis_distance, 0.20, 1.2),
        0.13 * q_down_derivative(metrics.high_mode_fraction, 0.05, 1.0),
    };
    std::vector<double>* outputs[] = {&output.x, &output.y, &output.z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            (*outputs[coordinate])[parameter] = 100.0 * (
                metric_derivatives[0] * d_length[coordinate][parameter] +
                metric_derivatives[1] * d_p95[coordinate][parameter] +
                metric_derivatives[2] * d_max[coordinate][parameter] +
                metric_derivatives[3] * d_spacing[coordinate][parameter] +
                metric_derivatives[4] * d_radius[coordinate][parameter] +
                metric_derivatives[5] * d_high_mode[coordinate][parameter]
            );
        }
    }
    int active_current = 0;
    for (int coil = 1; coil < n_base_coils; ++coil) {
        if (std::abs(currents_a[coil]) > std::abs(currents_a[active_current])) active_current = coil;
    }
    if (currents_a[active_current] != 0.0) {
        output.current[active_current] = 100.0 * 0.07 *
            q_down_derivative(metrics.current_abs_max, 2.0e6, 1.0) *
            std::copysign(1.0, currents_a[active_current]);
    }
    if (metrics_out) {
        metrics_out->length_mean = metrics.length_mean;
        metrics_out->curvature_p95 = metrics.curvature_p95;
        metrics_out->curvature_max = metrics.curvature_max;
        metrics_out->min_intercoil_distance = metrics.min_intercoil_distance;
        metrics_out->min_axis_distance = metrics.min_axis_distance;
        metrics_out->high_mode_fraction = metrics.high_mode_fraction;
        metrics_out->current_abs_max = metrics.current_abs_max;
        metrics_out->spacing_pair[0] = static_cast<int>(spacing_a.coil);
        metrics_out->spacing_pair[1] = static_cast<int>(spacing_b.coil);
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            metrics_out->d_length[coordinate] = d_length[coordinate];
            metrics_out->d_p95[coordinate] = d_p95[coordinate];
            metrics_out->d_max[coordinate] = d_max[coordinate];
            metrics_out->d_spacing[coordinate] = d_spacing[coordinate];
            metrics_out->d_radius[coordinate] = d_radius[coordinate];
            metrics_out->d_high_mode[coordinate] = d_high_mode[coordinate];
        }
    }
    return true;
}


}  // namespace coil_eval::pipeline
