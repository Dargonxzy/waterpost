// 中等连通簇的漂浮物判定，见 floaters.hpp。

#include "floaters.hpp"

#include "parallel.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace ai3d::waterpost
{
namespace
{

std::uint64_t xy_key(std::int64_t x, std::int64_t y)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::int32_t>(x))) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::int32_t>(y)));
}

} // namespace

std::vector<FloaterCandidate> classify_floaters(const Gaussians& g, const std::vector<std::int32_t>& component,
                                                const std::vector<std::size_t>& sizes, const std::vector<std::uint8_t>& border,
                                                const FloaterParams& params,
                                                const std::function<double(double, double, double, double)>& water_level)
{
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::int32_t> slot(sizes.size(), -1);
    std::vector<FloaterCandidate> out;
    for (std::size_t c = 0; c < sizes.size(); ++c)
    {
        if (sizes[c] >= params.min_points && sizes[c] <= params.max_points)
        {
            slot[c] = static_cast<std::int32_t>(out.size());
            FloaterCandidate candidate;
            candidate.component = static_cast<std::int32_t>(c);
            candidate.points = sizes[c];
            candidate.border = c < border.size() && border[c];
            out.push_back(candidate);
        }
    }
    if (out.empty())
    {
        return out;
    }
    std::vector<std::vector<std::uint32_t>> members(out.size());
    for (std::size_t i = 0; i < g.count; ++i)
    {
        const std::int32_t c = component[i];
        if (c >= 0 && slot[static_cast<std::size_t>(c)] >= 0)
        {
            members[static_cast<std::size_t>(slot[static_cast<std::size_t>(c)])].push_back(static_cast<std::uint32_t>(i));
        }
    }
    // 主体（锚点）：点数 > max_points 的簇。按 1 m XY 格建索引，既用于最近距离，也用于簇下方的柱体支撑。
    const double cell = 1.0;
    auto coord = [&](double v) { return static_cast<std::int64_t>(std::floor(v / cell)); };
    std::vector<std::pair<std::uint64_t, std::uint32_t>> anchors;
    for (std::size_t i = 0; i < g.count; ++i)
    {
        const std::int32_t c = component[i];
        if (c >= 0 && sizes[static_cast<std::size_t>(c)] > params.max_points)
        {
            anchors.emplace_back(xy_key(coord(g.xyz[3 * i]), coord(g.xyz[3 * i + 1])), static_cast<std::uint32_t>(i));
        }
    }
    std::sort(anchors.begin(), anchors.end());
    const double reach = std::max(2 * params.gap, params.column_pad);

    parallel_for_each(out.size(), [&](std::size_t s) {
        FloaterCandidate& f = out[s];
        const std::vector<std::uint32_t>& m = members[s];
        const std::size_t count = m.size();
        f.low = {inf, inf, inf};
        f.high = {-inf, -inf, -inf};
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        double opacity = 0;
        std::map<int, std::size_t> classes;
        for (std::uint32_t i : m)
        {
            const double* p = g.point(i);
            for (int a = 0; a < 3; ++a)
            {
                f.low[a] = std::min(f.low[a], p[a]);
                f.high[a] = std::max(f.high[a], p[a]);
                mean[a] += p[a];
            }
            opacity += g.opacity[i];
            ++classes[g.semantic.empty() ? -1 : g.semantic[2 * i]];
        }
        mean /= static_cast<double>(std::max<std::size_t>(count, 1));
        f.centroid = {mean[0], mean[1], mean[2]};
        f.mean_opacity = opacity / static_cast<double>(std::max<std::size_t>(count, 1));
        std::size_t best_class = 0;
        for (const auto& [label, number] : classes)
        {
            if (number > best_class)
            {
                best_class = number;
                f.semantic = label;
            }
        }
        // 形状：协方差主轴。
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (std::uint32_t i : m)
        {
            const Eigen::Vector3d d = Eigen::Vector3d(g.point(i)[0], g.point(i)[1], g.point(i)[2]) - mean;
            covariance += d * d.transpose();
        }
        covariance /= static_cast<double>(std::max<std::size_t>(count, 1));
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        const Eigen::Vector3d values = solver.eigenvalues().cwiseMax(0.0); // 升序
        f.length = 4 * std::sqrt(values[2]);
        f.elongation = std::sqrt(values[2] / std::max(values[1], 1e-12));
        // 簇内间距：最近邻距离的中位数（簇最多 max_points 个点，直接两两比较）。
        std::vector<double> nearest(count, inf);
        for (std::size_t a = 0; a < count; ++a)
        {
            const double* p = g.point(m[a]);
            for (std::size_t b = a + 1; b < count; ++b)
            {
                const double* q = g.point(m[b]);
                const double d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
                nearest[a] = std::min(nearest[a], d2);
                nearest[b] = std::min(nearest[b], d2);
            }
        }
        if (count > 1)
        {
            std::nth_element(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(count / 2), nearest.end());
            f.spacing = std::sqrt(nearest[count / 2]);
        }
        // 主体距离与柱体支撑：只看簇包围盒 XY 外扩 reach 之内的主体点（更远的点距离必然 > reach）。
        std::vector<std::pair<double, std::uint32_t>> near;
        f.support_top = -inf;
        for (std::int64_t x = coord(f.low[0] - reach); x <= coord(f.high[0] + reach); ++x)
        {
            for (std::int64_t y = coord(f.low[1] - reach); y <= coord(f.high[1] + reach); ++y)
            {
                const std::uint64_t key = xy_key(x, y);
                auto it = std::lower_bound(anchors.begin(), anchors.end(), std::make_pair(key, std::uint32_t{0}));
                for (; it != anchors.end() && it->first == key; ++it)
                {
                    const double* p = g.point(it->second);
                    const bool column = p[0] >= f.low[0] - params.column_pad && p[0] <= f.high[0] + params.column_pad &&
                                        p[1] >= f.low[1] - params.column_pad && p[1] <= f.high[1] + params.column_pad;
                    if (column)
                    {
                        if (p[2] <= f.low[2])
                        {
                            f.support_top = std::max(f.support_top, p[2]);
                        }
                        else if (p[2] >= f.high[2])
                        {
                            f.covered = true;
                        }
                    }
                    double lb = 0;
                    for (int a = 0; a < 3; ++a)
                    {
                        const double d = std::max({f.low[a] - p[a], 0.0, p[a] - f.high[a]});
                        lb += d * d;
                    }
                    if (lb <= reach * reach)
                    {
                        near.emplace_back(lb, it->second);
                    }
                }
            }
        }
        std::sort(near.begin(), near.end());
        double best = inf;
        for (const auto& [lb, i] : near)
        {
            if (lb >= best)
            {
                break;
            }
            const double* p = g.point(i);
            for (std::uint32_t j : m)
            {
                const double* q = g.point(j);
                best = std::min(best, (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
            }
        }
        f.gap = std::sqrt(best);
        if (!std::isfinite(f.support_top))
        {
            f.support_top = nan;
        }
        f.water_level = water_level ? water_level(f.low[0] - params.column_pad, f.low[1] - params.column_pad,
                                                  f.high[0] + params.column_pad, f.high[1] + params.column_pad)
                                    : nan;
        double base = -inf;
        if (std::isfinite(f.support_top)) base = std::max(base, f.support_top);
        if (std::isfinite(f.water_level)) base = std::max(base, f.water_level);
        f.clearance = std::isfinite(base) ? f.low[2] - base : inf;
        if (f.border)
        {
            f.decision = "keep_block_border";
        }
        else if (f.gap < params.gap)
        {
            f.decision = "keep_near_structure";
        }
        else if (f.elongation >= params.elongation && f.length >= params.elongated_length)
        {
            f.decision = "review_elongated";
        }
        else if (f.covered)
        {
            f.decision = "review_detached_covered";
        }
        else if (!std::isfinite(f.clearance))
        {
            // 下方既没有主体点也没有水面：没有观测不等于悬空，交给其他证据。
            f.decision = "review_unobserved_below";
        }
        else if (f.clearance >= params.clearance)
        {
            f.decision = "remove_suspended_floater";
        }
        else
        {
            f.decision = "review_detached_supported";
        }
    });
    return out;
}

} // namespace ai3d::waterpost
