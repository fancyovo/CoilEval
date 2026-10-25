#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace coil_eval::pipeline::geometry_detail {

struct DistanceMinimum {
    double distance = std::numeric_limits<double>::infinity();
    size_t left = 0;
    size_t right = 0;
};

// Exact sampled nearest-neighbour search. Points keep their original indices;
// ordering by (distance,left,right) preserves the first strict-< active pair of
// the original nested loop, including equal rounded square-root distances.
template <class Point>
class CoilDistanceTree {
    struct Node {
        double lower[3], upper[3];
        size_t begin, end, maximum_index;
        int label, left = -1, right = -1;
        bool uniform_label = true;
    };
    const std::vector<Point>& points_;
    std::vector<size_t> order_;
    std::vector<Node> nodes_;

    static double coordinate(const Point& p, int axis) {
        return axis == 0 ? p.point.x : axis == 1 ? p.point.y : p.point.z;
    }

    int build(size_t begin, size_t end) {
        Node node{};
        node.begin = begin; node.end = end;
        node.maximum_index = order_[begin];
        node.label = points_[order_[begin]].label;
        for (int a = 0; a < 3; ++a)
            node.lower[a] = node.upper[a] = coordinate(points_[order_[begin]], a);
        for (size_t i = begin + 1; i < end; ++i) {
            const size_t index = order_[i];
            node.maximum_index = std::max(node.maximum_index, index);
            if (points_[index].label != node.label) node.uniform_label = false;
            for (int a = 0; a < 3; ++a) {
                node.lower[a] = std::min(node.lower[a], coordinate(points_[index], a));
                node.upper[a] = std::max(node.upper[a], coordinate(points_[index], a));
            }
        }
        const int own = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        if (end - begin <= 16) return own;
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (node.upper[a] - node.lower[a] > node.upper[axis] - node.lower[axis]) axis = a;
        const size_t middle = begin + (end - begin) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
            [&](size_t a, size_t b) {
                const double x = coordinate(points_[a], axis), y = coordinate(points_[b], axis);
                return x < y || (x == y && a < b);
            });
        nodes_[own].left = build(begin, middle);
        nodes_[own].right = build(middle, end);
        return own;
    }

    void query(int id, size_t left, const double* lower, const double* upper,
               DistanceMinimum& best) const {
        const Node& node = nodes_[id];
        if (node.maximum_index <= left || (node.uniform_label && node.label == points_[left].label)) return;
        for (int a = 0; a < 3; ++a)
            if (node.lower[a] > upper[a] || node.upper[a] < lower[a]) return;
        if (node.left >= 0) {
            query(node.left, left, lower, upper, best);
            query(node.right, left, lower, upper, best);
            return;
        }
        const auto& p = points_[left].point;
        for (size_t i = node.begin; i < node.end; ++i) {
            const size_t right = order_[i];
            if (right <= left || points_[right].label == points_[left].label) continue;
            const auto& q = points_[right].point;
            const double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
            const double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (distance < best.distance || (distance == best.distance &&
                    (left < best.left || (left == best.left && right < best.right))))
                best = {distance, left, right};
        }
    }

public:
    explicit CoilDistanceTree(const std::vector<Point>& points) : points_(points), order_(points.size()) {
        std::iota(order_.begin(), order_.end(), size_t(0));
        nodes_.reserve(2 * points.size());
        if (!points.empty()) build(0, points.size());
    }

    DistanceMinimum minimum() const {
        DistanceMinimum best;
        const double infinity = std::numeric_limits<double>::infinity();
        for (size_t left = 0; left < points_.size(); ++left) {
            // Enlarge the search cube outwards, so rounding in p +/- radius
            // cannot prune a candidate whose rounded norm ties or beats best.
            const double radius = std::nextafter(best.distance, infinity);
            double lower[3], upper[3];
            for (int a = 0; a < 3; ++a) {
                const double p = coordinate(points_[left], a);
                lower[a] = std::nextafter(p - radius, -infinity);
                upper[a] = std::nextafter(p + radius, infinity);
            }
            query(0, left, lower, upper, best);
            if (best.distance == 0.0) break;
        }
        return best;
    }
};

template <class Point>
DistanceMinimum closest_distinct_label_pair(const std::vector<Point>& points) {
    // Preserve the original nonfinite-input behaviour without ordering NaNs.
    for (const auto& p : points) {
        if (!std::isfinite(p.point.x) || !std::isfinite(p.point.y) || !std::isfinite(p.point.z)) {
            DistanceMinimum best;
            for (size_t i = 0; i < points.size(); ++i)
                for (size_t j = i + 1; j < points.size(); ++j) {
                    if (points[i].label == points[j].label) continue;
                    const double dx = points[i].point.x - points[j].point.x;
                    const double dy = points[i].point.y - points[j].point.y;
                    const double dz = points[i].point.z - points[j].point.z;
                    const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (d < best.distance) best = {d, i, j};
                }
            return best;
        }
    }
    return CoilDistanceTree<Point>(points).minimum();
}

}  // namespace coil_eval::pipeline::geometry_detail
