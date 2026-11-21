// apply 第二段：冲突消解与采样矩形（finalize_block）、全局颜色归约（reduce_plans）、
// 体积清理与水面高斯生成（carve_water_volume / _generated），并把结果写回 LOD 最细层。

#include "apply_common.hpp"
#include "error.hpp"
#include "floaters.hpp"
#include "fs_util.hpp"
#include "lod.hpp"
#include "npz_util.hpp"
#include "parallel.hpp"
#include "scene.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iterator>
#include <tuple>
#include <limits>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ai3d::waterpost
{
namespace
{

using CellSet = std::unordered_set<std::uint64_t>;

std::set<int> active_bodies(const Config& config, std::map<int, CellSet>& unsupported)
{
    const Json active = read_json_file(config.work / "apply" / "active" / "active.json");
    std::set<int> result;
    for (auto it = active.at("bodies").begin(); it != active.at("bodies").end(); ++it)
    {
        if (it.value().at("active").get<bool>())
        {
            result.insert(std::stoi(it.key()));
        }
    }
    const NpzReader reader(config.work / "apply" / "active" / "unsupported.npz");
    const std::vector<std::int64_t> cells = npz_get<std::int64_t>(reader, "cells");
    const std::vector<std::int32_t> ids = npz_get<std::int32_t>(reader, "water_ids");
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        unsupported[ids[k]].insert(cell_key(cells[2 * k], cells[2 * k + 1]));
    }
    return result;
}

// _accepted_cells：活跃水体去掉无支持区域，同一格属于多个水体的格子作为冲突丢弃。
std::map<int, std::vector<std::int64_t>> accepted_cells(const Candidates& c, const std::set<int>& active,
                                                        const std::map<int, CellSet>& unsupported,
                                                        std::vector<std::int64_t>* conflicts)
{
    std::unordered_map<std::uint64_t, int> counts;
    std::vector<std::size_t> rows;
    for (std::size_t k = 0; k < c.ids.size(); ++k)
    {
        if (!active.count(c.ids[k]))
        {
            continue;
        }
        const std::uint64_t key = cell_key(c.cells[2 * k], c.cells[2 * k + 1]);
        const auto u = unsupported.find(c.ids[k]);
        if (u != unsupported.end() && u->second.count(key))
        {
            continue;
        }
        rows.push_back(k);
        ++counts[key];
    }
    std::map<int, std::vector<std::int64_t>> result;
    CellSet reported;
    for (std::size_t k : rows)
    {
        const std::uint64_t key = cell_key(c.cells[2 * k], c.cells[2 * k + 1]);
        if (counts[key] == 1)
        {
            result[c.ids[k]].push_back(c.cells[2 * k]);
            result[c.ids[k]].push_back(c.cells[2 * k + 1]);
        }
        else if (conflicts && reported.insert(key).second)
        {
            conflicts->push_back(c.cells[2 * k]);
            conflicts->push_back(c.cells[2 * k + 1]);
        }
    }
    return result;
}

struct Rect
{
    double low[2];
    double high[2];
};

std::vector<Rect> subtract_rectangle(const Rect& rect, const Rect& obstacle)
{
    const double a0 = std::max(rect.low[0], obstacle.low[0]), a1 = std::max(rect.low[1], obstacle.low[1]);
    const double b0 = std::min(rect.high[0], obstacle.high[0]), b1 = std::min(rect.high[1], obstacle.high[1]);
    if (b0 <= a0 || b1 <= a1)
    {
        return {rect};
    }
    const Rect pieces[4] = {{{rect.low[0], rect.low[1]}, {a0, rect.high[1]}},
                            {{b0, rect.low[1]}, {rect.high[0], rect.high[1]}},
                            {{a0, rect.low[1]}, {b0, a1}},
                            {{a0, b1}, {b0, rect.high[1]}}};
    std::vector<Rect> out;
    for (const Rect& p : pieces)
    {
        if (p.high[0] > p.low[0] && p.high[1] > p.low[1])
        {
            out.push_back(p);
        }
    }
    return out;
}

// _regions：先出现的块拥有重叠区（只看 Z 范围包含水位的块）。
std::vector<Rect> owned_regions(const Report& report, const std::string& name, double height)
{
    const BlockRecord& own = report.block(name);
    if (height < own.bounds.minimum[2] || height > own.bounds.maximum[2])
    {
        return {};
    }
    Rect base{{std::max(own.bounds.minimum[0], report.domain.lower[0]), std::max(own.bounds.minimum[1], report.domain.lower[1])},
              {std::min(own.bounds.maximum[0], report.domain.upper[0]), std::min(own.bounds.maximum[1], report.domain.upper[1])}};
    std::vector<Rect> regions;
    if (base.high[0] > base.low[0] && base.high[1] > base.low[1])
    {
        regions.push_back(base);
    }
    for (const BlockRecord& other : report.blocks)
    {
        if (other.name == name)
        {
            break;
        }
        if (other.bounds.minimum[2] <= height && height <= other.bounds.maximum[2])
        {
            const Rect obstacle{{other.bounds.minimum[0], other.bounds.minimum[1]}, {other.bounds.maximum[0], other.bounds.maximum[1]}};
            std::vector<Rect> next;
            for (const Rect& r : regions)
            {
                for (const Rect& piece : subtract_rectangle(r, obstacle))
                {
                    next.push_back(piece);
                }
            }
            regions = std::move(next);
        }
    }
    return regions;
}

int cell_owner(const Report& report, double x, double y, const double* height)
{
    const double qx = std::min(std::max(x, std::nextafter(report.domain.lower[0], report.domain.upper[0])),
                               std::nextafter(report.domain.upper[0], report.domain.lower[0]));
    const double qy = std::min(std::max(y, std::nextafter(report.domain.lower[1], report.domain.upper[1])),
                               std::nextafter(report.domain.upper[1], report.domain.lower[1]));
    for (std::size_t i = 0; i < report.blocks.size(); ++i)
    {
        const Box3& b = report.blocks[i].bounds;
        if (height && !(b.minimum[2] <= *height && *height <= b.maximum[2]))
        {
            continue;
        }
        if (qx >= b.minimum[0] && qx <= b.maximum[0] && qy >= b.minimum[1] && qy <= b.maximum[1])
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void axis_cells(double low, double high, double bound_low, double bound_high, double step, double tolerance,
                std::vector<double>& centers, std::vector<double>& widths)
{
    centers.clear();
    widths.clear();
    double a = std::max(low, bound_low + tolerance);
    const double end = std::min(high, bound_high - tolerance);
    if (end <= a)
    {
        return;
    }
    const double center = (a + end) / 2;
    if (end - a <= std::min(center - bound_low, bound_high - center) / 4 * (1 - 1e-4))
    {
        centers.push_back(center);
        widths.push_back(end - a);
        return;
    }
    while (a < end)
    {
        const double width = std::min({step, (a - bound_low) / 5, (bound_high - a) / 6, end - a});
        if (width <= 0 || a + width <= a)
        {
            fail_value("Water boundary refinement reached numerical resolution");
        }
        centers.push_back(a + width / 2);
        widths.push_back(width);
        a += width;
        if (centers.size() > 256)
        {
            fail_value("Excessive boundary refinement; inspect project coordinate precision");
        }
    }
}

// _sample_rectangles：全局对齐的四叉合并；块边界精确划分覆盖。
void sample_rectangles(const std::vector<std::int64_t>& cells, const GridSpec& grid, double maximum,
                       const std::vector<Rect>& regions, const Domain& domain, std::vector<double>& points,
                       std::vector<double>& coverage)
{
    points.clear();
    coverage.clear();
    const std::size_t n = cells.size() / 2;
    if (!n)
    {
        return;
    }
    const double spacing = grid.spacing;
    const std::int64_t largest = std::int64_t{1} << std::max(0, static_cast<int>(std::floor(std::log2(maximum / spacing))));
    std::int64_t cmin[2] = {INT64_MAX, INT64_MAX}, cmax[2] = {INT64_MIN, INT64_MIN};
    for (std::size_t k = 0; k < n; ++k)
    {
        for (int a = 0; a < 2; ++a)
        {
            cmin[a] = std::min(cmin[a], cells[2 * k + a]);
            cmax[a] = std::max(cmax[a], cells[2 * k + a]);
        }
    }
    auto floordiv = [](std::int64_t a, std::int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    std::int64_t low[2], shape[2];
    for (int a = 0; a < 2; ++a)
    {
        low[a] = floordiv(cmin[a], largest) * largest;
        shape[a] = floordiv(cmax[a] + largest, largest) * largest - low[a];
    }
    cv::Mat mask = cv::Mat::zeros(static_cast<int>(shape[0]), static_cast<int>(shape[1]), CV_8U);
    for (std::size_t k = 0; k < n; ++k)
    {
        mask.at<std::uint8_t>(static_cast<int>(cells[2 * k] - low[0]), static_cast<int>(cells[2 * k + 1] - low[1])) = 1;
    }
    cv::Mat padded, distance;
    cv::copyMakeBorder(mask, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::distanceTransform(padded, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);
    distance = distance(cv::Rect(1, 1, mask.cols, mask.rows)).clone();
    cv::Mat remaining = mask.clone();
    std::vector<std::array<double, 3>> centers; // x, y, width
    for (std::int64_t k = largest; k; k /= 2)
    {
        const int bi = static_cast<int>(shape[0] / k), bj = static_cast<int>(shape[1] / k);
        for (int i = 0; i < bi; ++i)
        {
            for (int j = 0; j < bj; ++j)
            {
                bool good = true;
                float dmin = 1e30f;
                for (int y = 0; y < k && good; ++y)
                {
                    for (int x = 0; x < k; ++x)
                    {
                        const int r = static_cast<int>(i * k + y), c = static_cast<int>(j * k + x);
                        if (!remaining.at<std::uint8_t>(r, c))
                        {
                            good = false;
                            break;
                        }
                        dmin = std::min(dmin, distance.at<float>(r, c));
                    }
                }
                if (good && k > 1)
                {
                    good = static_cast<double>(dmin) * spacing >= 3.0 * static_cast<double>(k) * spacing;
                }
                if (!good)
                {
                    continue;
                }
                centers.push_back({grid.origin[0] + (static_cast<double>(i * k) + (k - 1) / 2.0 + low[0]) * spacing,
                                   grid.origin[1] + (static_cast<double>(j * k) + (k - 1) / 2.0 + low[1]) * spacing,
                                   static_cast<double>(k) * spacing});
                for (int y = 0; y < k; ++y)
                {
                    for (int x = 0; x < k; ++x)
                    {
                        remaining.at<std::uint8_t>(static_cast<int>(i * k + y), static_cast<int>(j * k + x)) = 0;
                    }
                }
            }
        }
    }
    const std::array<double, 2> tolerance = domain.sampling_tolerance(spacing);
    for (int a = 0; a < 2; ++a)
    {
        if (domain.upper[a] - domain.lower[a] <= 2 * tolerance[a])
        {
            fail_value("Project bounds are below the sampling precision");
        }
    }
    for (const Rect& region : regions)
    {
        for (const auto& c : centers)
        {
            double lo[2], hi[2];
            bool valid = true;
            for (int a = 0; a < 2; ++a)
            {
                lo[a] = std::max(c[a] - c[2] / 2, region.low[a]);
                hi[a] = std::min(c[a] + c[2] / 2, region.high[a]);
                valid &= hi[a] > lo[a];
            }
            if (!valid)
            {
                continue;
            }
            // 补水范围（ROI / 瓦片）的边缘不必整齐：矩形中心在范围内就整块保留，否则丢弃，不再拆成小格。
            std::vector<std::array<double, 4>> pieces;
            if (domain.inside((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
            {
                pieces.push_back({lo[0], lo[1], hi[0], hi[1]});
            }
            for (const auto& piece : pieces)
            {
            lo[0] = piece[0];
            lo[1] = piece[1];
            hi[0] = piece[2];
            hi[1] = piece[3];
            const double p[2] = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2};
            const double w[2] = {hi[0] - lo[0], hi[1] - lo[1]};
            bool safe = true;
            for (int a = 0; a < 2; ++a)
            {
                safe &= w[a] <= domain.horizontal_clearance(p[0], p[1], a) / 4 * (1 - 1e-4);
            }
            if (safe)
            {
                points.insert(points.end(), {p[0], p[1]});
                coverage.insert(coverage.end(), {w[0], w[1]});
                continue;
            }
            std::vector<double> cx, wx, cy, wy;
            axis_cells(lo[0], hi[0], domain.lower[0], domain.upper[0], hi[0] - lo[0], tolerance[0], cx, wx);
            axis_cells(lo[1], hi[1], domain.lower[1], domain.upper[1], hi[1] - lo[1], tolerance[1], cy, wy);
            for (std::size_t a = 0; a < cx.size(); ++a)
            {
                for (std::size_t b = 0; b < cy.size(); ++b)
                {
                    points.insert(points.end(), {cx[a], cy[b]});
                    coverage.insert(coverage.end(), {wx[a], wy[b]});
                }
            }
            }
        }
    }
}

bool boxes_near(const double* a_low, const double* a_high, const Json& b, double radius)
{
    if (!b.is_object())
    {
        return false;
    }
    double sum = 0;
    for (int k = 0; k < 2; ++k)
    {
        const double bl = jnum(b.at("minimum")[static_cast<std::size_t>(k)]), bh = jnum(b.at("maximum")[static_cast<std::size_t>(k)]);
        const double d = std::max(std::max(a_low[k] - bh, bl - a_high[k]), 0.0);
        sum += d * d;
    }
    return std::sqrt(sum) <= radius;
}

// 2D 网格近邻（同一水体的采样点都在同一水位上）。
class PlaneIndex
{
public:
    PlaneIndex(const std::vector<double>& points, double cell) : m_points(points), m_cell(cell)
    {
        for (std::size_t i = 0; i < points.size() / 3; ++i)
        {
            m_cells[key(points[3 * i], points[3 * i + 1])].push_back(i);
        }
    }

    // 最近 k 个距离（含自身 0），升序。
    std::vector<double> nearest(double x, double y, std::size_t k) const
    {
        std::vector<double> best;
        const std::int64_t cx = static_cast<std::int64_t>(std::floor(x / m_cell));
        const std::int64_t cy = static_cast<std::int64_t>(std::floor(y / m_cell));
        for (std::int64_t ring = 0;; ++ring)
        {
            for (std::int64_t dx = -ring; dx <= ring; ++dx)
            {
                for (std::int64_t dy = -ring; dy <= ring; ++dy)
                {
                    if (std::max(std::abs(dx), std::abs(dy)) != ring)
                    {
                        continue;
                    }
                    const auto found = m_cells.find(pack(cx + dx, cy + dy));
                    if (found == m_cells.end())
                    {
                        continue;
                    }
                    for (std::size_t i : found->second)
                    {
                        best.push_back(std::hypot(m_points[3 * i] - x, m_points[3 * i + 1] - y));
                    }
                }
            }
            std::sort(best.begin(), best.end());
            if (best.size() > k)
            {
                best.resize(k);
            }
            if ((best.size() >= k && best.back() <= static_cast<double>(ring) * m_cell) || ring > 100000)
            {
                return best;
            }
            if (best.size() >= std::min(k, m_points.size() / 3) && ring * m_cell > 1e6)
            {
                return best;
            }
        }
    }

private:
    static std::uint64_t pack(std::int64_t x, std::int64_t y)
    {
        return cell_key(x, y);
    }
    std::uint64_t key(double x, double y) const
    {
        return pack(static_cast<std::int64_t>(std::floor(x / m_cell)), static_cast<std::int64_t>(std::floor(y / m_cell)));
    }
    const std::vector<double>& m_points;
    double m_cell;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> m_cells;
};

struct Samples
{
    std::vector<double> points;     // 3m
    std::vector<double> coverage;   // 2m
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> footprint_cells;
    std::vector<std::int32_t> footprint_ids;
    std::vector<std::int32_t> source_ids;
    std::vector<std::int64_t> water_indices;
};

Samples load_samples(const Config& config, const std::string& block)
{
    const NpzReader reader(config.work / "apply" / "samples" / utf8_path(block) / "samples.npz");
    Samples s;
    s.points = npz_get<double>(reader, "points");
    s.coverage = npz_get<double>(reader, "coverage_steps");
    s.ids = npz_get<std::int32_t>(reader, "water_ids");
    s.footprint_cells = npz_get<std::int64_t>(reader, "footprint_cells");
    s.footprint_ids = npz_get<std::int32_t>(reader, "footprint_ids");
    s.source_ids = npz_get<std::int32_t>(reader, "source_water_ids");
    s.water_indices = npz_get<std::int64_t>(reader, "water_indices");
    return s;
}

// 漂浮物去噪：剩余高斯按"中心距离 <= denoise_radius"连成簇（并查集），点数少于 denoise_min_points 的小簇
// 视为漂浮物删除。与逐点密度阈值不同，树冠边缘等稀疏但与主体相连的高斯属于大簇，不受影响。
// 碰到块瓦片边界（距离 < 2 * radius）的簇可能连着邻块，保留。
using WaterLevel = std::function<double(double, double, double, double)>;

// 去噪的附带输出：中等簇的判定明细（floaters.json），以及每个剩余高斯所在连通簇的点数（残留水点诊断用）。
struct DenoiseExtras
{
    WaterLevel water_level;
    const Gaussians* context = nullptr;        // 照片核验渲染时带上的邻块最细层高斯
    std::vector<FloaterCandidate> candidates;
    std::vector<std::uint32_t> component_size; // n；已删除的点为 0
};

std::vector<std::int64_t> denoise_cluster(const Gaussians& g, const std::vector<std::int64_t>& removed, const LodBlock& block,
                                          const Config& config, Json& stats, DenoiseExtras& extras)
{
    const Timer timer;
    const std::size_t n = g.count;
    std::vector<std::uint8_t> gone(n, 0);
    for (std::int64_t r : removed)
    {
        gone[static_cast<std::size_t>(r)] = 1;
    }
    const double radius = config.denoise_radius;
    const double r2 = radius * radius;
    const double cell = radius / std::sqrt(3.0); // 同一体素内任意两点距离 <= radius
    auto voxel = [&](const double* p) {
        const std::int64_t x = static_cast<std::int64_t>(std::floor(p[0] / cell));
        const std::int64_t y = static_cast<std::int64_t>(std::floor(p[1] / cell));
        const std::int64_t z = static_cast<std::int64_t>(std::floor(p[2] / cell));
        return std::array<std::int64_t, 3>{x, y, z};
    };
    auto pack = [](std::int64_t x, std::int64_t y, std::int64_t z) {
        return (static_cast<std::uint64_t>(x & 0x1fffff) << 42) | (static_cast<std::uint64_t>(y & 0x1fffff) << 21) |
               static_cast<std::uint64_t>(z & 0x1fffff);
    };
    std::vector<std::pair<std::uint64_t, std::uint32_t>> keys;
    keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!gone[i])
        {
            const auto v = voxel(g.point(i));
            keys.emplace_back(pack(v[0], v[1], v[2]), static_cast<std::uint32_t>(i));
        }
    }
    std::sort(keys.begin(), keys.end());
    // 体素：起止区间。
    std::vector<std::uint64_t> voxel_key;
    std::vector<std::size_t> voxel_start;
    for (std::size_t k = 0; k < keys.size(); ++k)
    {
        if (k == 0 || keys[k].first != keys[k - 1].first)
        {
            voxel_key.push_back(keys[k].first);
            voxel_start.push_back(k);
        }
    }
    voxel_start.push_back(keys.size());
    const std::size_t voxels = voxel_key.size();
    // 并查集按体素：同一体素内的点必然相连。
    std::vector<std::uint32_t> parent(voxels);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](std::uint32_t a) {
        while (parent[a] != a)
        {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    // 相邻体素（偏移 [-2, 2]^3 的前半部分）里有一对点距离 <= radius 即合并。
    std::vector<std::array<int, 3>> offsets;
    for (int dx = -2; dx <= 2; ++dx)
    {
        for (int dy = -2; dy <= 2; ++dy)
        {
            for (int dz = -2; dz <= 2; ++dz)
            {
                if (std::make_tuple(dx, dy, dz) > std::make_tuple(0, 0, 0))
                {
                    offsets.push_back({dx, dy, dz});
                }
            }
        }
    }
    for (std::size_t v = 0; v < voxels; ++v)
    {
        const double* first = g.point(keys[voxel_start[v]].second);
        const auto base = voxel(first);
        for (const auto& o : offsets)
        {
            const std::uint64_t key = pack(base[0] + o[0], base[1] + o[1], base[2] + o[2]);
            const auto it = std::lower_bound(voxel_key.begin(), voxel_key.end(), key);
            if (it == voxel_key.end() || *it != key)
            {
                continue;
            }
            const std::uint32_t u = static_cast<std::uint32_t>(it - voxel_key.begin());
            std::uint32_t a = find(static_cast<std::uint32_t>(v)), b = find(u);
            if (a == b)
            {
                continue;
            }
            bool linked = false;
            for (std::size_t p = voxel_start[v]; p < voxel_start[v + 1] && !linked; ++p)
            {
                const double* x = g.point(keys[p].second);
                for (std::size_t q = voxel_start[u]; q < voxel_start[u + 1]; ++q)
                {
                    const double* y = g.point(keys[q].second);
                    if ((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]) <= r2)
                    {
                        linked = true;
                        break;
                    }
                }
            }
            if (linked)
            {
                parent[std::max(a, b)] = std::min(a, b);
            }
        }
    }
    // 簇大小与是否碰到瓦片边界。
    Box3 tile;
    for (const LodLeaf& leaf : block.leaves)
    {
        tile.expand(leaf.box);
    }
    const double margin = 2 * radius;
    std::unordered_map<std::uint32_t, std::size_t> size;
    std::unordered_set<std::uint32_t> border;
    for (std::size_t v = 0; v < voxels; ++v)
    {
        const std::uint32_t root = find(static_cast<std::uint32_t>(v));
        size[root] += voxel_start[v + 1] - voxel_start[v];
        for (std::size_t p = voxel_start[v]; p < voxel_start[v + 1]; ++p)
        {
            const double* x = g.point(keys[p].second);
            if (x[0] - tile.minimum[0] < margin || tile.maximum[0] - x[0] < margin || x[1] - tile.minimum[1] < margin ||
                tile.maximum[1] - x[1] < margin)
            {
                border.insert(root);
                break;
            }
        }
    }
    std::vector<std::int64_t> floaters;
    std::size_t clusters = 0;
    for (std::size_t v = 0; v < voxels; ++v)
    {
        const std::uint32_t root = find(static_cast<std::uint32_t>(v));
        if (size[root] >= static_cast<std::size_t>(config.denoise_min_points) || border.count(root))
        {
            continue;
        }
        for (std::size_t p = voxel_start[v]; p < voxel_start[v + 1]; ++p)
        {
            floaters.push_back(static_cast<std::int64_t>(keys[p].second));
        }
    }
    for (const auto& [root, count] : size)
    {
        clusters += (count < static_cast<std::size_t>(config.denoise_min_points) && !border.count(root)) ? 1 : 0;
    }
    // 第二步：中等簇（>= min_points）按主体距离与下方支撑判定（floaters.hpp）。总是计算并记录；
    // 只有 floater_check 打开时才删除 remove_* 的簇。
    std::vector<std::int32_t> component(n, -1);
    std::vector<std::size_t> sizes;
    std::vector<std::uint8_t> touches;
    {
        std::unordered_map<std::uint32_t, std::int32_t> dense;
        for (std::size_t v = 0; v < voxels; ++v)
        {
            const std::uint32_t root = find(static_cast<std::uint32_t>(v));
            const auto [it, inserted] = dense.emplace(root, static_cast<std::int32_t>(sizes.size()));
            if (inserted)
            {
                sizes.push_back(size[root]);
                touches.push_back(border.count(root) ? 1 : 0);
            }
            for (std::size_t p = voxel_start[v]; p < voxel_start[v + 1]; ++p)
            {
                component[keys[p].second] = it->second;
            }
        }
    }
    extras.component_size.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (component[i] >= 0)
        {
            extras.component_size[i] = static_cast<std::uint32_t>(sizes[static_cast<std::size_t>(component[i])]);
        }
    }
    FloaterParams params;
    params.min_points = static_cast<std::size_t>(std::max(1, config.denoise_min_points));
    params.max_points = static_cast<std::size_t>(std::max(config.floater_max_points, config.denoise_min_points));
    params.gap = config.floater_gap;
    params.clearance = config.floater_clearance;
    params.elongation = config.floater_elongation;
    params.column_pad = radius;
    const Timer floater_timer;
    extras.candidates = classify_floaters(g, component, sizes, touches, params, extras.water_level);
    Json photo_stats;
    if (config.floater_check == 1)
    {
        // 照片核验：非块边界的中等簇都核验（几何上"贴近主体"的簇也可能是贴着树冠的漂浮物）。
        FloaterPhotoParams photo;
        photo.photos = !config.floater_photos.empty() ? config.floater_photos
                       : config.project.empty()       ? std::filesystem::path()
                                                      : config.project / ".temp" / "undistortion" / "images";
        photo.max_views = static_cast<std::size_t>(std::max(1, config.floater_photo_views));
        photo.min_gain = config.floater_photo_gain;
        photo.gpu = config.gpu;
        std::vector<std::uint8_t> check(extras.candidates.size(), 0);
        for (std::size_t k = 0; k < check.size(); ++k)
        {
            check[k] = extras.candidates[k].border ? 0 : 1;
        }
        std::vector<View> views;
        if (!photo.photos.empty() && std::filesystem::is_directory(photo.photos))
        {
            if (const SceneBlock* scene_block = load_scene(config).block(block.name))
            {
                views = scene_views(config, scene_block->view_keys);
            }
        }
        std::size_t rendered = 0;
        double seconds = 0;
        verify_floaters_with_photos(g, component, extras.candidates, check, views, photo, extras.context, rendered, seconds);
        std::map<std::string, std::size_t> verdicts;
        for (FloaterCandidate& f : extras.candidates)
        {
            combine_floater_evidence(f);
            ++verdicts[f.photo];
        }
        photo_stats["photos"] = path_utf8(photo.photos);
        photo_stats["views_available"] = views.size();
        photo_stats["views_rendered"] = rendered;
        photo_stats["verdicts"] = verdicts;
        photo_stats["seconds"] = seconds;
        photo_stats["context_gaussians"] = extras.context ? extras.context->count : 0;
        log("[FLOATER] %s: photo check of %zu clusters in %zu views (%.1fs)", block.name.c_str(), extras.candidates.size(),
            rendered, seconds);
    }
    else
    {
        for (FloaterCandidate& f : extras.candidates)
        {
            f.geometry = f.decision;
        }
    }
    std::map<std::string, std::size_t> decisions, decision_points;
    std::size_t floater_points = 0;
    auto drops = [&](const FloaterCandidate& f) {
        return config.floater_check == 2 ? !f.border : (config.floater_check == 1 && f.remove());
    };
    for (const FloaterCandidate& f : extras.candidates)
    {
        ++decisions[f.decision];
        decision_points[f.decision] += f.points;
        if (drops(f))
        {
            floater_points += f.points;
        }
    }
    if (config.floater_check)
    {
        std::vector<std::uint8_t> drop(sizes.size(), 0);
        for (const FloaterCandidate& f : extras.candidates)
        {
            drop[static_cast<std::size_t>(f.component)] = drops(f) ? 1 : 0;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (component[i] >= 0 && drop[static_cast<std::size_t>(component[i])])
            {
                floaters.push_back(static_cast<std::int64_t>(i));
            }
        }
    }
    Json floater_stats;
    floater_stats["mode"] = config.floater_check;
    floater_stats["max_points"] = params.max_points;
    floater_stats["gap"] = params.gap;
    floater_stats["clearance"] = params.clearance;
    floater_stats["elongation"] = params.elongation;
    floater_stats["candidates"] = extras.candidates.size();
    floater_stats["decisions"] = decisions;
    floater_stats["decision_points"] = decision_points;
    floater_stats["removed_points"] = floater_points;
    floater_stats["seconds"] = floater_timer.seconds();
    if (!photo_stats.is_null())
    {
        floater_stats["photo_check"] = photo_stats;
    }
    stats["floater_check"] = floater_stats;
    std::sort(floaters.begin(), floaters.end());
    stats["radius"] = radius;
    stats["min_points"] = config.denoise_min_points;
    stats["components"] = size.size();
    stats["removed_clusters"] = clusters;
    stats["removed_points"] = floaters.size();
    stats["seconds"] = timer.seconds();
    log("[DENOISE] %s: %zu floater points in %zu small clusters (of %zu components), %.1fs", block.name.c_str(),
        floaters.size() - floater_points, clusters, size.size(), timer.seconds());
    std::size_t remove_clusters = 0, remove_points = 0, review_clusters = 0;
    for (const auto& [decision, count] : decisions)
    {
        if (decision.rfind("remove_", 0) == 0)
        {
            remove_clusters += count;
            remove_points += decision_points[decision];
        }
        review_clusters += decision.rfind("review_", 0) == 0 ? count : 0;
    }
    log("[FLOATER] %s: %zu medium clusters checked, %zu to remove (%zu points) %s, %zu for review", block.name.c_str(),
        extras.candidates.size(), remove_clusters, remove_points,
        config.floater_check == 1 ? "removed" : config.floater_check == 2 ? "(all candidates dropped: experiment)" : "reported only",
        review_clusters);
    return floaters;
}

// 经典点云滤波（PCL / Open3D 同义）：
//   sor    统计离群点去除：k 近邻平均距离 > 全块均值 + std_ratio * 标准差 即删除；
//   radius 半径离群点去除：半径内邻居少于 min_neighbors 即删除。
// 邻域只取剩余的原始高斯中心；块瓦片边界 2 倍搜索距离内的点不删（邻居可能在邻块）。
std::vector<std::int64_t> denoise_filter(const Gaussians& g, const std::vector<std::int64_t>& removed, const LodBlock& block,
                                         const Config& config, Json& stats)
{
    const Timer timer;
    const std::size_t n = g.count;
    std::vector<std::uint8_t> gone(n, 0);
    for (std::int64_t r : removed)
    {
        gone[static_cast<std::size_t>(r)] = 1;
    }
    const bool sor = config.denoise_method == "sor";
    const double reach = sor ? config.denoise_max_distance : config.denoise_radius;
    const double cell = std::max(0.05, reach / 4);
    auto coord = [&](double v) { return static_cast<std::int64_t>(std::floor(v / cell)); };
    auto pack = [](std::int64_t x, std::int64_t y, std::int64_t z) {
        return (static_cast<std::uint64_t>(x & 0x1fffff) << 42) | (static_cast<std::uint64_t>(y & 0x1fffff) << 21) |
               static_cast<std::uint64_t>(z & 0x1fffff);
    };
    std::vector<std::pair<std::uint64_t, std::uint32_t>> keys;
    std::vector<std::uint32_t> alive;
    keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!gone[i])
        {
            const double* p = g.point(i);
            keys.emplace_back(pack(coord(p[0]), coord(p[1]), coord(p[2])), static_cast<std::uint32_t>(i));
            alive.push_back(static_cast<std::uint32_t>(i));
        }
    }
    std::sort(keys.begin(), keys.end());
    const int k = sor ? config.denoise_k : config.denoise_min_neighbors;
    const int rings = static_cast<int>(std::ceil(reach / cell));
    std::vector<float> value(alive.size(), 0.0f);
    parallel_ranges(alive.size(), 4096, [&](std::size_t begin, std::size_t end) {
        std::vector<double> best;
        for (std::size_t a = begin; a < end; ++a)
        {
            const std::size_t i = alive[a];
            const double* p = g.point(i);
            const std::int64_t cx = coord(p[0]), cy = coord(p[1]), cz = coord(p[2]);
            best.clear();
            int count = 0;
            for (int ring = 0; ring <= rings; ++ring)
            {
                for (int dx = -ring; dx <= ring; ++dx)
                {
                    for (int dy = -ring; dy <= ring; ++dy)
                    {
                        for (int dz = -ring; dz <= ring; ++dz)
                        {
                            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != ring)
                            {
                                continue;
                            }
                            const std::uint64_t key = pack(cx + dx, cy + dy, cz + dz);
                            auto it = std::lower_bound(keys.begin(), keys.end(), std::make_pair(key, std::uint32_t{0}));
                            for (; it != keys.end() && it->first == key; ++it)
                            {
                                if (it->second == i)
                                {
                                    continue;
                                }
                                const double* q = g.point(it->second);
                                const double d = std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) +
                                                           (p[2] - q[2]) * (p[2] - q[2]));
                                if (d > reach)
                                {
                                    continue;
                                }
                                if (sor)
                                {
                                    best.push_back(d);
                                }
                                else
                                {
                                    ++count;
                                }
                            }
                        }
                    }
                }
                if (sor)
                {
                    if (static_cast<int>(best.size()) >= k)
                    {
                        std::nth_element(best.begin(), best.begin() + (k - 1), best.end());
                        best.resize(static_cast<std::size_t>(k));
                        if (*std::max_element(best.begin(), best.end()) <= ring * cell)
                        {
                            break;
                        }
                    }
                }
                else if (count >= k)
                {
                    break;
                }
            }
            if (sor)
            {
                double sum = 0;
                for (double d : best)
                {
                    sum += d;
                }
                // 搜索距离内不足 k 个邻居：缺的按搜索距离计。
                sum += (k - static_cast<int>(std::min<std::size_t>(best.size(), static_cast<std::size_t>(k)))) * reach;
                value[a] = static_cast<float>(sum / k);
            }
            else
            {
                value[a] = static_cast<float>(count);
            }
        }
    });
    double threshold = 0;
    if (sor)
    {
        double mean = 0, square = 0;
        for (float v : value)
        {
            mean += v;
        }
        mean /= std::max<std::size_t>(1, value.size());
        for (float v : value)
        {
            square += (v - mean) * (v - mean);
        }
        const double std_dev = std::sqrt(square / std::max<std::size_t>(1, value.size()));
        threshold = mean + config.denoise_std_ratio * std_dev;
        stats["mean_distance"] = mean;
        stats["std_distance"] = std_dev;
    }
    Box3 tile;
    for (const LodLeaf& leaf : block.leaves)
    {
        tile.expand(leaf.box);
    }
    const double margin = 2 * reach;
    std::vector<std::int64_t> out;
    for (std::size_t a = 0; a < alive.size(); ++a)
    {
        const bool drop = sor ? value[a] > threshold : value[a] < k;
        if (!drop)
        {
            continue;
        }
        const double* x = g.point(alive[a]);
        if (x[0] - tile.minimum[0] < margin || tile.maximum[0] - x[0] < margin || x[1] - tile.minimum[1] < margin ||
            tile.maximum[1] - x[1] < margin)
        {
            continue;
        }
        out.push_back(alive[a]);
    }
    stats["method"] = config.denoise_method;
    stats["threshold"] = threshold;
    stats["removed_points"] = out.size();
    stats["seconds"] = timer.seconds();
    log("[DENOISE] %s: %s removed %zu points, %.1fs", block.name.c_str(), config.denoise_method.c_str(), out.size(), timer.seconds());
    return out;
}

std::vector<std::int64_t> denoise_floaters(const Gaussians& g, const std::vector<std::int64_t>& removed, const LodBlock& block,
                                           const Config& config, Json& stats, DenoiseExtras& extras)
{
    if (config.denoise_method == "cluster")
    {
        stats["method"] = "cluster";
        return denoise_cluster(g, removed, block, config, stats, extras);
    }
    if (config.denoise_method == "sor" || config.denoise_method == "radius")
    {
        return denoise_filter(g, removed, block, config, stats);
    }
    fail_value("Unknown --denoise-method (sor | radius | cluster): " + config.denoise_method);
}

} // namespace

std::vector<std::string> pipeline_blocks(const Config& config)
{
    std::vector<std::string> result;
    for (const SceneBlock& block : load_scene(config).blocks)
    {
        result.push_back(block.name);
    }
    return result;
}

void run_samples(const Config& config, const std::string& name)
{
    const Timer total;
    const Report report = load_report(config);
    const GridSpec& grid = report.grid;
    const double spacing = grid.spacing;
    std::map<int, CellSet> unsupported;
    const std::set<int> active = active_bodies(config, unsupported);
    const Candidates own = load_candidates(config, name);
    std::vector<std::int64_t> conflicts;
    const std::map<int, std::vector<std::int64_t>> accepted = accepted_cells(own, active, unsupported, &conflicts);

    // 原水点（种子 + 删除候选）归给 0.72 格内的接受足迹；多个水体时记 -1。
    const std::filesystem::path dir = detect_dir(config, name);
    const NpzReader water_reader(dir / "seeds.npz");
    const std::vector<double> seed_xyz = npz_get<double>(water_reader, "water_xyz");
    const std::vector<float> seed_rgb = npz_get<float>(water_reader, "water_rgb");
    std::vector<double> deletion_xyz;
    std::vector<std::int64_t> deletion_indices;
    if (std::filesystem::is_regular_file(dir / "deletion_evidence.npz"))
    {
        const NpzReader reader(dir / "deletion_evidence.npz");
        deletion_xyz = npz_get<double>(reader, "deletion_xyz");
        deletion_indices = npz_get<std::int64_t>(reader, "deletion_indices");
    }
    const std::size_t seeds = seed_xyz.size() / 3, total_water = seeds + deletion_xyz.size() / 3;
    auto water_point = [&](std::size_t i) { return i < seeds ? seed_xyz.data() + 3 * i : deletion_xyz.data() + 3 * (i - seeds); };
    std::vector<std::int32_t> source_ids(total_water, 0);
    double wlow[2] = {1e300, 1e300}, whigh[2] = {-1e300, -1e300};
    for (std::size_t i = 0; i < total_water; ++i)
    {
        for (int a = 0; a < 2; ++a)
        {
            wlow[a] = std::min(wlow[a], water_point(i)[a]);
            whigh[a] = std::max(whigh[a], water_point(i)[a]);
        }
    }
    const double radius = spacing * .72;
    std::size_t candidate_files = 1;
    if (total_water)
    {
        for (const BlockRecord& block : report.blocks)
        {
            // 候选格只落在所属块的范围内（± 半格）：范围离本块水点太远的块不可能相关，不读它的 candidates。
            if (block.name != name)
            {
                double gap = 0;
                for (int a = 0; a < 2; ++a)
                {
                    const double d = std::max({wlow[a] - (block.bounds.maximum[a] + spacing), (block.bounds.minimum[a] - spacing) - whigh[a], 0.0});
                    gap += d * d;
                }
                if (std::sqrt(gap) > radius * (1 + 1e-9) + 1e-9)
                {
                    continue;
                }
            }
            candidate_files += block.name == name ? 0 : 1;
            const Candidates other = block.name == name ? Candidates() : load_candidates(config, block.name);
            const Json& bodies = (block.name == name ? own : other).metadata.at("bodies");
            std::vector<int> relevant;
            for (int wid : active)
            {
                const std::string key = std::to_string(wid);
                if (bodies.contains(key) && boxes_near(wlow, whigh, bodies.at(key).at("bbox"), radius))
                {
                    relevant.push_back(wid);
                }
            }
            if (relevant.empty())
            {
                continue;
            }
            const std::map<int, std::vector<std::int64_t>> neighbour =
                block.name == name ? accepted : accepted_cells(other, active, unsupported, nullptr);
            for (int wid : relevant)
            {
                const auto found = neighbour.find(wid);
                if (found == neighbour.end() || found->second.empty())
                {
                    continue;
                }
                CellSet lookup;
                for (std::size_t k = 0; k < found->second.size() / 2; ++k)
                {
                    lookup.insert(cell_key(found->second[2 * k], found->second[2 * k + 1]));
                }
                std::vector<std::uint8_t> take(total_water, 0);
                parallel_ranges(total_water, 65536, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        const double* p = water_point(i);
                        const std::int64_t cx = static_cast<std::int64_t>(std::floor((p[0] - grid.origin[0]) / spacing));
                        const std::int64_t cy = static_cast<std::int64_t>(std::floor((p[1] - grid.origin[1]) / spacing));
                        for (std::int64_t dx = -1; dx <= 2 && !take[i]; ++dx)
                        {
                            for (std::int64_t dy = -1; dy <= 2; ++dy)
                            {
                                if (!lookup.count(cell_key(cx + dx, cy + dy)))
                                {
                                    continue;
                                }
                                if (std::hypot(grid.origin[0] + (cx + dx) * spacing - p[0], grid.origin[1] + (cy + dy) * spacing - p[1]) <= radius)
                                {
                                    take[i] = 1;
                                    break;
                                }
                            }
                        }
                    }
                });
                for (std::size_t i = 0; i < total_water; ++i)
                {
                    if (!take[i])
                    {
                        continue;
                    }
                    if (source_ids[i] == 0)
                    {
                        source_ids[i] = wid;
                    }
                    else if (source_ids[i] != wid)
                    {
                        source_ids[i] = -1;
                    }
                }
            }
        }
    }

    const int owner = report.block_index(name);
    const double maximum = std::max(config.spacing, spacing);
    Samples out;
    Json bodies = Json::object();
    for (const Surface& surface : report.surfaces)
    {
        const int wid = surface.water_id;
        if (!active.count(wid))
        {
            continue;
        }
        const auto found = accepted.find(wid);
        const std::vector<std::int64_t> empty;
        const std::vector<std::int64_t>& cells = found == accepted.end() ? empty : found->second;
        std::vector<double> xy, steps;
        sample_rectangles(cells, grid, maximum, owned_regions(report, name, surface.height), report.domain, xy, steps);
        Box3 bbox;
        for (std::size_t k = 0; k < xy.size() / 2; ++k)
        {
            out.points.insert(out.points.end(), {xy[2 * k], xy[2 * k + 1], surface.height});
            out.coverage.insert(out.coverage.end(), {steps[2 * k], steps[2 * k + 1]});
            out.ids.push_back(wid);
            bbox.expand(Vec3{xy[2 * k], xy[2 * k + 1], surface.height});
        }
        std::size_t owned = 0;
        for (std::size_t k = 0; k < cells.size() / 2; ++k)
        {
            if (cell_owner(report, grid.origin[0] + cells[2 * k] * spacing, grid.origin[1] + cells[2 * k + 1] * spacing, &surface.height) == owner)
            {
                out.footprint_cells.insert(out.footprint_cells.end(), {cells[2 * k], cells[2 * k + 1]});
                out.footprint_ids.push_back(wid);
                ++owned;
            }
        }
        double rgb[3] = {0, 0, 0};
        std::size_t color = 0, original = 0;
        for (std::size_t i = 0; i < seeds; ++i)
        {
            if (source_ids[i] == wid)
            {
                ++color;
                for (int a = 0; a < 3; ++a)
                {
                    rgb[a] += seed_rgb[3 * i + a];
                }
            }
        }
        const std::size_t first = deletion_xyz.empty() ? 0 : seeds;
        for (std::size_t i = first; i < total_water; ++i)
        {
            original += source_ids[i] == wid ? 1 : 0;
        }
        Json body;
        body["new_points"] = xy.size() / 2;
        if (bbox.valid())
        {
            Json b;
            b["minimum"] = bbox.minimum;
            b["maximum"] = bbox.maximum;
            body["bbox"] = b;
        }
        else
        {
            body["bbox"] = nullptr;
        }
        body["original_water"] = original;
        body["color_source_count"] = color;
        body["rgb_sum"] = {rgb[0], rgb[1], rgb[2]};
        body["footprint_cells"] = owned;
        bodies[std::to_string(wid)] = body;
    }
    std::size_t conflict_count = 0;
    for (std::size_t k = 0; k < conflicts.size() / 2; ++k)
    {
        conflict_count += cell_owner(report, grid.origin[0] + conflicts[2 * k] * spacing, grid.origin[1] + conflicts[2 * k + 1] * spacing, nullptr) == owner;
    }
    if (!deletion_xyz.empty())
    {
        out.source_ids.assign(source_ids.begin() + static_cast<std::ptrdiff_t>(seeds), source_ids.end());
        out.water_indices = deletion_indices;
    }
    // 没有删除候选（无水块）时也没有要归属的原水点。
    const std::filesystem::path target = config.work / "apply" / "samples" / utf8_path(name);
    std::filesystem::create_directories(target);
    NpzWriter writer;
    npz_put(writer, "points", out.points, {out.ids.size(), 3});
    npz_put(writer, "coverage_steps", out.coverage, {out.ids.size(), 2});
    npz_put(writer, "water_ids", out.ids);
    npz_put(writer, "footprint_cells", out.footprint_cells, {out.footprint_ids.size(), 2});
    npz_put(writer, "footprint_ids", out.footprint_ids);
    npz_put(writer, "source_water_ids", out.source_ids);
    npz_put(writer, "water_indices", out.water_indices);
    writer.write(target / "samples.npz", true);
    Json metadata;
    metadata["block"] = name;
    metadata["bodies"] = bodies;
    metadata["ambiguous_footprint_cells"] = conflict_count;
    metadata["source_water"] = out.water_indices.size();
    metadata["seconds"] = total.seconds();
    write_json_atomic(target / "metadata.json", metadata, false);
    log("[SAMPLES] %s: %zu new water samples, %zu footprint cells (%.1fs)", name.c_str(), out.ids.size(),
        out.footprint_ids.size(), total.seconds());
    log("[SAMPLES] %s: read candidates of %zu/%zu blocks", name.c_str(), candidate_files, report.blocks.size());
}

void run_model(const Config& config)
{
    const Report report = load_report(config);
    std::vector<Json> plans;
    for (const BlockRecord& block : report.blocks)
    {
        plans.push_back(read_json_file(config.work / "apply" / "samples" / utf8_path(block.name) / "metadata.json"));
    }
    const Json active = read_json_file(config.work / "apply" / "active" / "active.json");
    Json surfaces = Json::array();
    Json skipped = report.json.at("skipped_surfaces");
    Json index = Json::object();
    for (const Surface& surface : report.surfaces)
    {
        const std::string key = std::to_string(surface.water_id);
        std::size_t count = 0, color = 0, points = 0;
        double rgb[3] = {0, 0, 0};
        Json shards = Json::array();
        for (std::size_t b = 0; b < plans.size(); ++b)
        {
            if (!plans[b].at("bodies").contains(key))
            {
                continue;
            }
            const Json& body = plans[b].at("bodies").at(key);
            count += body.at("original_water").get<std::size_t>();
            color += body.at("color_source_count").get<std::size_t>();
            const std::size_t n = body.at("new_points").get<std::size_t>();
            points += n;
            for (int a = 0; a < 3; ++a)
            {
                rgb[a] += jnum(body.at("rgb_sum")[static_cast<std::size_t>(a)]);
            }
            if (n)
            {
                Json shard;
                shard["block"] = report.blocks[b].name;
                shard["count"] = n;
                shard["bbox"] = body.at("bbox");
                shards.push_back(shard);
            }
        }
        if (!count || !points || !color)
        {
            Json item;
            item["water_id"] = surface.water_id;
            const Json& status = active.at("bodies").contains(key) ? active.at("bodies").at(key) : Json::object();
            item["reason"] = status.value("candidate_cells", 0) == 0 ? "empty_own_mask_footprint"
                             : !status.value("active", false)       ? "no_original_water_support_before_conflict_resolution"
                                                                    : "no_unambiguous_source_water_or_footprint";
            skipped.push_back(item);
            continue;
        }
        Json item;
        item["water_id"] = surface.water_id;
        item["height"] = surface.height;
        item["height_sigma"] = surface.sigma;
        item["rgb"] = {rgb[0] / color, rgb[1] / color, rgb[2] / color};
        item["original_water"] = count;
        item["new_points"] = points;
        surfaces.push_back(item);
        index[key] = shards;
    }
    const std::filesystem::path out = config.work / "apply" / "model";
    std::filesystem::create_directories(out);
    Json model;
    model["surfaces"] = surfaces;
    model["skipped_surfaces"] = skipped;
    model["plan_index"] = index;
    write_json_atomic(out / "report.json", model, false);
    log("[MODEL] %zu accepted surfaces, %zu skipped", surfaces.size(), skipped.size());
}

void run_write(const Config& config, const std::string& name)
{
    const Timer total;
    const Report report = load_report(config);
    const GridSpec& grid = report.grid;
    const double spacing = grid.spacing;
    const Json model = read_json_file(config.work / "apply" / "model" / "report.json");
    std::map<int, std::pair<double, std::array<double, 3>>> accepted;
    for (const Json& s : model.at("surfaces"))
    {
        std::array<double, 3> rgb{};
        for (int a = 0; a < 3; ++a)
        {
            rgb[static_cast<std::size_t>(a)] = jnum(s.at("rgb")[static_cast<std::size_t>(a)]);
        }
        accepted[static_cast<int>(jint(s.at("water_id")))] = {jnum(s.at("height")), rgb};
    }
    const Samples samples = load_samples(config, name);
    // 其他块的 samples 只按需读一次（清理窗口与近邻定尺度共用）。
    std::map<std::string, Samples> sample_cache;
    auto samples_of = [&](const std::string& block_name) -> const Samples& {
        if (block_name == name)
        {
            return samples;
        }
        auto found = sample_cache.find(block_name);
        if (found == sample_cache.end())
        {
            found = sample_cache.emplace(block_name, load_samples(config, block_name)).first;
        }
        return found->second;
    };
    std::vector<std::int64_t> voted;
    for (std::size_t k = 0; k < samples.source_ids.size(); ++k)
    {
        if (samples.source_ids[k] > 0 && accepted.count(samples.source_ids[k]))
        {
            voted.push_back(samples.water_indices[k]);
        }
    }
    std::sort(voted.begin(), voted.end());
    const LodIndex lod = read_lod_index(config.lod);
    const LodBlock* block = lod.block(name);
    if (!block)
    {
        fail_value("Block is not present in the LOD: " + name);
    }
    std::vector<std::size_t> leaf_offsets;
    const Gaussians g = read_block_gaussians(lod, *block, &leaf_offsets);
    const std::size_t n = g.count;
    for (std::int64_t r : voted)
    {
        if (r < 0 || static_cast<std::size_t>(r) >= n)
        {
            fail_value("Removal index outside the LOD block: the LOD changed after detect");
        }
    }

    // ---------------- 足迹格网窗口：体积清理、漂浮簇下方水位、新水面岸线约束共用。
    // label：0 无水，-2 工程格网之外，>0 已接受水体的足迹（所有相邻块的 samples 合起来）。
    std::int64_t low[2] = {0, 0}, size[2] = {0, 0};
    cv::Mat label;
    if (!accepted.empty() && n)
    {
        // 窗口：本块点的 XY 范围 + 腐蚀与可达余量。
        double xlo[2] = {1e300, 1e300}, xhi[2] = {-1e300, -1e300};
        for (std::size_t i = 0; i < n; ++i)
        {
            for (int a = 0; a < 2; ++a)
            {
                xlo[a] = std::min(xlo[a], g.xyz[3 * i + a]);
                xhi[a] = std::max(xhi[a], g.xyz[3 * i + a]);
            }
        }
        const int reach_cells = static_cast<int>(std::ceil(config.carve_below_reach / spacing));
        const int margin = std::max(config.carve_erosion_cells, reach_cells) + 2;
        for (int a = 0; a < 2; ++a)
        {
            low[a] = static_cast<std::int64_t>(std::floor((xlo[a] - grid.origin[a]) / spacing)) - margin;
            size[a] = static_cast<std::int64_t>(std::ceil((xhi[a] - grid.origin[a]) / spacing)) + margin + 1 - low[a];
        }
        if (size[0] * size[1] > 400000000LL)
        {
            fail_value("Water carve window is too large");
        }
        label = cv::Mat(static_cast<int>(size[0]), static_cast<int>(size[1]), CV_32S, cv::Scalar(0));
        for (int x = 0; x < label.rows; ++x)
        {
            const std::int64_t gx = low[0] + x;
            for (int y = 0; y < label.cols; ++y)
            {
                const std::int64_t gy = low[1] + y;
                if (gx < 0 || gx >= grid.shape[0] || gy < 0 || gy >= grid.shape[1])
                {
                    label.at<int>(x, y) = -2;
                }
            }
        }
        for (const BlockRecord& other : report.blocks)
        {
            // 足迹格在所属块的范围内（± 一格）：范围与清理窗口不相交的块跳过，不读它的 samples。
            bool overlaps = true;
            for (int a = 0; a < 2; ++a)
            {
                const std::int64_t first = static_cast<std::int64_t>(std::floor((other.bounds.minimum[a] - spacing - grid.origin[a]) / spacing)) - 1;
                const std::int64_t last = static_cast<std::int64_t>(std::ceil((other.bounds.maximum[a] + spacing - grid.origin[a]) / spacing)) + 1;
                overlaps &= last >= low[a] && first < low[a] + size[a];
            }
            if (!overlaps)
            {
                continue;
            }
            const Samples& s = samples_of(other.name);
            for (std::size_t k = 0; k < s.footprint_ids.size(); ++k)
            {
                const std::int64_t lx = s.footprint_cells[2 * k] - low[0], ly = s.footprint_cells[2 * k + 1] - low[1];
                if (accepted.count(s.footprint_ids[k]) && lx >= 0 && ly >= 0 && lx < size[0] && ly < size[1])
                {
                    label.at<int>(static_cast<int>(lx), static_cast<int>(ly)) = s.footprint_ids[k];
                }
            }
        }
    }
    auto footprint_at = [&](double x, double y) -> int {
        if (label.empty())
        {
            return 0;
        }
        const std::int64_t cx = static_cast<std::int64_t>(std::nearbyint((x - grid.origin[0]) / spacing)) - low[0];
        const std::int64_t cy = static_cast<std::int64_t>(std::nearbyint((y - grid.origin[1]) / spacing)) - low[1];
        if (cx < 0 || cy < 0 || cx >= size[0] || cy >= size[1])
        {
            return 0;
        }
        return label.at<int>(static_cast<int>(cx), static_cast<int>(cy));
    };

    // ---------------- 体积清理
    Json carve_stats;
    std::vector<std::int64_t> carved;
    // 残留水点诊断（问题 B）：带水证据（语义为水，或可见渲染 / 几何水份额达到阈值）却既未被投票删除、也未被体积清理
    // 选中的原始点，记下第一条没满足的条件。只诊断，不改变删除结果。
    static const char* const RESIDUAL_REASONS[] = {
        "guarded_on_land",          // 已被选中，又被"渲染非水占优"保护撤销
        "shore_protected",          // 已被选中，又被岸线锚点保护撤销（不在腐蚀内部区）
        "no_water_in_reach",        // 可达距离内没有已接受水体
        "below_too_deep",           // 低于水位超过 carve_max_depth
        "bank_without_evidence",    // 岸下深处，但既不在足迹内也没有岸边渲染水证据
        "near_level_outside_interior", // 水位附近、在腐蚀内部区之外（岸边带）
        "above_band_land",          // 内部区水面上方 above_band 内，但属于可靠陆地格网
        "above_not_in_footprint",   // 水面上方，不在任何足迹格内
        "above_low_visible",        // 水面上方、在足迹内，但可见水份额低于 carve_float_visible_water
        "above_too_high",           // 水面上方、有可见水证据，但超出 float_band / float_max
        "other"};
    constexpr int RESIDUAL_REASON_COUNT = static_cast<int>(sizeof(RESIDUAL_REASONS) / sizeof(RESIDUAL_REASONS[0]));
    std::vector<std::int64_t> residual;
    std::vector<std::uint8_t> residual_reason, residual_semantic;
    bool has_semantic = lod.semantic;
    for (std::size_t i = 0; i < n && !has_semantic; ++i)
    {
        has_semantic = g.semantic[2 * i] != 0 || g.semantic[2 * i + 1] != 0;
    }
    std::vector<std::uint8_t> voted_flag(n, 0);
    for (std::int64_t r : voted)
    {
        voted_flag[static_cast<std::size_t>(r)] = 1;
    }
    const std::filesystem::path dir = detect_dir(config, name);
    if (config.water_carve && !accepted.empty() && n)
    {
        // detect 交接文件里的下标都指向本块最细层高斯：LOD 在 detect 之后变了就会越界，先检查。
        auto check_rows = [&](const std::vector<std::int64_t>& rows, const char* what) {
            for (std::int64_t r : rows)
            {
                if (r < 0 || static_cast<std::size_t>(r) >= n)
                {
                    fail_value(std::string(what) + " index outside the LOD block " + name + ": the LOD changed after detect");
                }
            }
        };
        const std::vector<std::int64_t> shore = npz_get<std::int64_t>(NpzReader(dir / "components.npz"), "shore_indices");
        check_rows(shore, "Shoreline");
        std::vector<std::uint8_t> land(n, 0), bank_rendered(n, 0), float_rendered(n, 0), whole_on_water(n, 0), on_land(n, 0);
        bool has_land = false, has_visible = false;
        if (std::filesystem::is_regular_file(dir / "carve_evidence.npz"))
        {
            const NpzReader reader(dir / "carve_evidence.npz");
            const std::vector<std::int64_t> land_rows = npz_get<std::int64_t>(reader, "land_indices");
            check_rows(land_rows, "Land");
            for (std::int64_t r : land_rows)
            {
                land[static_cast<std::size_t>(r)] = 1;
            }
            has_land = true;
            const std::vector<std::int64_t> rows = npz_get<std::int64_t>(reader, "visible_water_indices");
            check_rows(rows, "Visible-water");
            const std::vector<std::uint8_t> fraction = npz_get<std::uint8_t>(reader, "visible_water_fraction");
            const std::vector<std::uint8_t> geometry = npz_get<std::uint8_t>(reader, "visible_water_geometry");
            for (std::size_t k = 0; k < rows.size(); ++k)
            {
                const std::size_t r = static_cast<std::size_t>(rows[k]);
                bank_rendered[r] = fraction[k] / 255.0 >= config.carve_bank_visible_water;
                float_rendered[r] = fraction[k] / 255.0 >= config.carve_float_visible_water;
                whole_on_water[r] = geometry[k] / 255.0 >= config.carve_float_geometry_water;
            }
            const std::vector<std::uint8_t> share = npz_get<std::uint8_t>(reader, "visible_water_u8");
            if (share.size() != n || fraction.size() != rows.size() || geometry.size() != rows.size())
            {
                fail_value("Carve evidence does not match the LOD block " + name + ": the LOD changed after detect");
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                on_land[i] = share[i] != 255 && share[i] < 127;
            }
            has_visible = true;
        }
        cv::Mat carve(label.size(), CV_32S, cv::Scalar(0));
        std::set<int> present;
        for (int x = 0; x < label.rows; ++x)
        {
            for (int y = 0; y < label.cols; ++y)
            {
                if (label.at<int>(x, y) > 0)
                {
                    present.insert(label.at<int>(x, y));
                }
            }
        }
        for (int wid : present)
        {
            cv::Mat body = (label == wid) | (label == -2);
            if (config.carve_erosion_cells)
            {
                cv::erode(body, body, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), config.carve_erosion_cells,
                          cv::BORDER_CONSTANT, cv::Scalar(255));
            }
            carve.setTo(wid, body & (label == wid));
        }
        // 可达范围内最近的水体。
        cv::Mat near(label.size(), CV_32S, cv::Scalar(0));
        if (!present.empty())
        {
            const cv::Mat outside = label <= 0;
            cv::Mat distance, nearest;
            cv::distanceTransform(outside, distance, nearest, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
            std::unordered_map<int, int> lut;
            for (int x = 0; x < label.rows; ++x)
            {
                for (int y = 0; y < label.cols; ++y)
                {
                    if (label.at<int>(x, y) > 0)
                    {
                        lut[nearest.at<int>(x, y)] = label.at<int>(x, y);
                    }
                }
            }
            for (int x = 0; x < label.rows; ++x)
            {
                for (int y = 0; y < label.cols; ++y)
                {
                    if (distance.at<float>(x, y) * spacing <= config.carve_below_reach)
                    {
                        const auto found = lut.find(nearest.at<int>(x, y));
                        near.at<int>(x, y) = found == lut.end() ? 0 : found->second;
                    }
                }
            }
        }
        std::vector<std::uint8_t> protect(n, 0);
        for (std::int64_t r : shore)
        {
            protect[static_cast<std::size_t>(r)] = 1;
        }
        std::size_t c_below = 0, c_deep = 0, c_floating = 0, c_above = 0, c_edge = 0, guarded = 0, interior = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double* p = g.point(i);
            const int cx = static_cast<int>(static_cast<std::int64_t>(std::nearbyint((p[0] - grid.origin[0]) / spacing)) - low[0]);
            const int cy = static_cast<int>(static_cast<std::int64_t>(std::nearbyint((p[1] - grid.origin[1]) / spacing)) - low[1]);
            const int wid = carve.at<int>(cx, cy);
            const int reach = near.at<int>(cx, cy);
            const bool inside = wid > 0;
            interior += inside;
            const double level = inside ? accepted.at(wid).first : std::numeric_limits<double>::quiet_NaN();
            const double reach_level = reach > 0 ? accepted.at(reach).first : std::numeric_limits<double>::quiet_NaN();
            const double dz = p[2] - level, reach_dz = p[2] - reach_level;
            const bool in_footprint = label.at<int>(cx, cy) > 0;
            bool below = inside && dz <= config.carve_surface_tolerance && dz >= -config.carve_max_depth;
            const bool bank_ok = in_footprint || bank_rendered[i];
            const bool deep = reach > 0 && bank_ok && reach_dz < -config.carve_below_margin && reach_dz >= -config.carve_max_depth;
            below = below || deep;
            const bool above = has_land && inside && dz > config.carve_surface_tolerance && dz <= config.carve_above_band && !land[i];
            bool floating = false;
            if (has_visible)
            {
                const bool rendered = in_footprint && float_rendered[i] && reach_dz > config.carve_surface_tolerance;
                floating = (rendered && reach_dz <= config.carve_float_band) ||
                           (rendered && whole_on_water[i] && reach_dz <= config.carve_float_max);
            }
            const bool edge_band = config.carve_edge_band && !inside && reach > 0 && in_footprint && bank_rendered[i] &&
                                   reach_dz >= -config.carve_below_margin && reach_dz <= config.carve_surface_tolerance &&
                                   !(has_land && land[i]);
            bool selected = below || above || floating || edge_band;
            bool was_guarded = false, was_shore = false;
            if (selected && has_visible && on_land[i])
            {
                ++guarded;
                selected = false;
                was_guarded = true;
            }
            if (selected && protect[i] && !inside)
            {
                selected = false;
                was_shore = true;
            }
            if (!selected && !voted_flag[i])
            {
                const bool semantic_water = has_semantic && g.semantic[2 * i] == config.water_class;
                const bool evidence = bank_rendered[i] || float_rendered[i] || whole_on_water[i];
                if (semantic_water || evidence)
                {
                    const double tol = config.carve_surface_tolerance;
                    int why = RESIDUAL_REASON_COUNT - 1;
                    if (was_guarded) why = 0;
                    else if (was_shore) why = 1;
                    else if (!inside && reach <= 0) why = 2;
                    else if ((inside && dz < -config.carve_max_depth) || (reach > 0 && reach_dz < -config.carve_max_depth)) why = 3;
                    else if (!inside && reach_dz < -config.carve_below_margin && !bank_ok) why = 4;
                    else if (!inside && reach_dz >= -config.carve_below_margin && reach_dz <= tol) why = 5;
                    else if (inside && dz > tol && dz <= config.carve_above_band) why = 6;
                    else if (reach_dz > tol && !in_footprint) why = 7;
                    else if (reach_dz > tol && !float_rendered[i]) why = 8;
                    else if (reach_dz > tol) why = 9;
                    residual.push_back(static_cast<std::int64_t>(i));
                    residual_reason.push_back(static_cast<std::uint8_t>(why));
                    residual_semantic.push_back(semantic_water ? 1 : 0);
                }
            }
            if (selected)
            {
                carved.push_back(static_cast<std::int64_t>(i));
                c_below += below;
                c_deep += deep && !inside;
                c_floating += floating && !(below || above);
                c_above += above;
                c_edge += edge_band && !(below || above || floating);
            }
        }
        carve_stats["carved"] = carved.size();
        carve_stats["below"] = c_below;
        carve_stats["deep_rim_or_bank"] = c_deep;
        carve_stats["floating"] = c_floating;
        carve_stats["above"] = c_above;
        carve_stats["edge_band"] = c_edge;
        carve_stats["rendered_land_guarded"] = guarded;
        carve_stats["interior_footprint_points"] = interior;
    }
    std::vector<std::int64_t> removed;
    std::set_union(voted.begin(), voted.end(), carved.begin(), carved.end(), std::back_inserter(removed));
    removed.erase(std::unique(removed.begin(), removed.end()), removed.end());

    // ---------------- 通用漂浮物去噪（与水无关，整块最细层）
    Json denoise_stats;
    DenoiseExtras extras;
    // 簇下方的水位：XY 范围内任一已接受足迹格的最高水位。
    extras.water_level = [&](double x0, double y0, double x1, double y1) {
        double level = std::numeric_limits<double>::quiet_NaN();
        if (label.empty())
        {
            return level;
        }
        for (double x = x0; x <= x1 + spacing / 2; x += spacing / 2)
        {
            for (double y = y0; y <= y1 + spacing / 2; y += spacing / 2)
            {
                const int wid = footprint_at(std::min(x, x1), std::min(y, y1));
                if (wid > 0)
                {
                    const double h = accepted.at(wid).first;
                    level = std::isnan(level) ? h : std::max(level, h);
                }
            }
        }
        return level;
    };
    // 照片核验的渲染背景：XY 范围与本块相距 50 m 以内的其他块的最细层高斯（只读一次，只在 floater_check = 1 时）。
    Gaussians context;
    if (config.denoise && n && config.floater_check == 1 && config.denoise_method == "cluster")
    {
        Box3 own;
        for (const LodLeaf& leaf : block->leaves)
        {
            own.expand(leaf.box);
        }
        for (const LodBlock& other : lod.blocks)
        {
            if (other.name == name)
            {
                continue;
            }
            Box3 box;
            for (const LodLeaf& leaf : other.leaves)
            {
                box.expand(leaf.box);
            }
            const double dx = std::max({own.minimum[0] - box.maximum[0], box.minimum[0] - own.maximum[0], 0.0});
            const double dy = std::max({own.minimum[1] - box.maximum[1], box.minimum[1] - own.maximum[1], 0.0});
            if (!box.valid() || std::hypot(dx, dy) > 50.0)
            {
                continue;
            }
            const Gaussians h = read_block_gaussians(lod, other, nullptr);
            context.count += h.count;
            context.xyz.insert(context.xyz.end(), h.xyz.begin(), h.xyz.end());
            context.cov.insert(context.cov.end(), h.cov.begin(), h.cov.end());
            context.opacity.insert(context.opacity.end(), h.opacity.begin(), h.opacity.end());
            context.rgba.insert(context.rgba.end(), h.rgba.begin(), h.rgba.end());
        }
        extras.context = &context;
    }
    if (config.denoise && n)
    {
        const std::vector<std::int64_t> floaters = denoise_floaters(g, removed, *block, config, denoise_stats, extras);
        std::vector<std::int64_t> merged;
        std::set_union(removed.begin(), removed.end(), floaters.begin(), floaters.end(), std::back_inserter(merged));
        removed = std::move(merged);
    }

    // ---------------- 新水面高斯（同一水体全局 k<=4 近邻定尺度）
    const Json& index = model.at("plan_index");
    // 新水面岸线诊断（问题 C）：沿 X / Y 到已接受足迹边界的距离 e，与 2 sigma 比较；water_edge_sigma = k > 0 时
    // 把该轴尺度收紧到 max(e / k, 5 cm)。e 从中心所在格沿轴向走到最后一个足迹格的外沿，最多走 4 sigma。
    auto edge_distance = [&](double x, double y, int axis, double limit) -> double {
        const double inf = std::numeric_limits<double>::infinity();
        if (label.empty())
        {
            return inf;
        }
        const double centre = axis == 0 ? x : y;
        std::int64_t c[2] = {static_cast<std::int64_t>(std::nearbyint((x - grid.origin[0]) / spacing)) - low[0],
                             static_cast<std::int64_t>(std::nearbyint((y - grid.origin[1]) / spacing)) - low[1]};
        auto inside = [&](std::int64_t a, std::int64_t b) {
            return a >= 0 && b >= 0 && a < size[0] && b < size[1] && label.at<int>(static_cast<int>(a), static_cast<int>(b)) > 0;
        };
        if (!inside(c[0], c[1]))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const std::int64_t steps = static_cast<std::int64_t>(std::ceil(limit / spacing)) + 1;
        double best = inf;
        for (int direction : {-1, 1})
        {
            std::int64_t at[2] = {c[0], c[1]};
            std::int64_t k = 0;
            for (; k < steps; ++k)
            {
                std::int64_t next[2] = {at[0], at[1]};
                next[axis] += direction;
                if (!inside(next[0], next[1]))
                {
                    break;
                }
                at[0] = next[0];
                at[1] = next[1];
            }
            if (k < steps)
            {
                const double boundary = grid.origin[axis] + (static_cast<double>(low[axis] + at[axis]) + 0.5 * direction) * spacing;
                best = std::min(best, std::abs(boundary - centre));
            }
        }
        return best;
    };
    std::size_t edge_unknown = 0, edge_limited = 0, edge_clamped = 0, overflow_before = 0, overflow_after = 0;
    double overflow_before_sum = 0, overflow_after_sum = 0, overflow_before_max = 0, overflow_after_max = 0;
    constexpr double OVERFLOW_TOLERANCE = 0.2; // 2 sigma 越过足迹边界超过 0.2 m 才计入
    std::vector<Vec3> new_position, new_scale;
    std::vector<std::array<double, 3>> new_rgb;
    std::size_t reference_shards_read = 0, reference_shards_total = 0;
    for (const auto& [wid, value] : accepted)
    {
        std::vector<std::size_t> rows;
        for (std::size_t k = 0; k < samples.ids.size(); ++k)
        {
            if (samples.ids[k] == wid)
            {
                rows.push_back(k);
            }
        }
        if (rows.empty())
        {
            continue;
        }
        // 近邻参照只需要本块点附近的分片：先只用本块自己的点算出每点第 kk 近邻距离的上界 reach（加入更多点只会
        // 让近邻更近），离本块点包围盒超过 reach 的分片不可能含有近邻，不读。本块点不足 kk 个时退回读全部分片。
        const Json& shards = index.at(std::to_string(wid));
        std::size_t total_points = 0;
        for (const Json& shard : shards)
        {
            total_points += static_cast<std::size_t>(jint(shard.at("count")));
        }
        const std::size_t kk = std::min<std::size_t>(4, total_points);
        const double cell = std::max(config.spacing, spacing);
        double reach = std::numeric_limits<double>::infinity();
        double own_low[2] = {1e300, 1e300}, own_high[2] = {-1e300, -1e300};
        if (kk > 0 && rows.size() >= kk)
        {
            std::vector<double> own(3 * rows.size());
            for (std::size_t q = 0; q < rows.size(); ++q)
            {
                for (int a = 0; a < 3; ++a)
                {
                    own[3 * q + a] = samples.points[3 * rows[q] + a];
                }
                for (int a = 0; a < 2; ++a)
                {
                    own_low[a] = std::min(own_low[a], own[3 * q + a]);
                    own_high[a] = std::max(own_high[a], own[3 * q + a]);
                }
            }
            const PlaneIndex own_plane(own, cell);
            std::vector<double> worst(rows.size(), 0.0);
            parallel_ranges(rows.size(), 4096, [&](std::size_t begin, std::size_t end) {
                for (std::size_t q = begin; q < end; ++q)
                {
                    worst[q] = own_plane.nearest(own[3 * q], own[3 * q + 1], kk).back();
                }
            });
            reach = *std::max_element(worst.begin(), worst.end());
        }
        std::vector<double> reference;
        std::size_t shards_read = 0;
        for (const Json& shard : shards)
        {
            const std::string other = shard.at("block").get<std::string>();
            if (other != name && std::isfinite(reach))
            {
                const Json& b = shard.at("bbox");
                double gap = 0;
                for (int a = 0; a < 2; ++a)
                {
                    const double lo = jnum(b.at("minimum")[static_cast<std::size_t>(a)]), hi = jnum(b.at("maximum")[static_cast<std::size_t>(a)]);
                    const double d = std::max({own_low[a] - hi, lo - own_high[a], 0.0});
                    gap += d * d;
                }
                if (std::sqrt(gap) > reach * (1 + 1e-9) + 1e-9)
                {
                    continue;
                }
            }
            ++shards_read;
            const Samples& s = samples_of(other);
            for (std::size_t k = 0; k < s.ids.size(); ++k)
            {
                if (s.ids[k] == wid)
                {
                    reference.insert(reference.end(), {s.points[3 * k], s.points[3 * k + 1], s.points[3 * k + 2]});
                }
            }
        }
        reference_shards_read += shards_read;
        reference_shards_total += shards.size();
        const PlaneIndex plane(reference, cell);
        std::vector<Vec3> scales(rows.size());
        std::vector<double> edge(2 * rows.size()), before(2 * rows.size());
        parallel_ranges(rows.size(), 4096, [&](std::size_t begin, std::size_t end) {
            for (std::size_t q = begin; q < end; ++q)
            {
                const std::size_t k = rows[q];
                const double x = samples.points[3 * k], y = samples.points[3 * k + 1], z = samples.points[3 * k + 2];
                const std::vector<double> d = plane.nearest(x, y, kk);
                double radius = std::sqrt(1e-7);
                if (d.size() > 1)
                {
                    double sum = 0;
                    for (std::size_t j = 1; j < d.size(); ++j)
                    {
                        sum += d[j] * d[j];
                    }
                    radius = std::sqrt(std::max(sum / static_cast<double>(d.size() - 1), 1e-7));
                }
                Vec3 s{std::max(radius, samples.coverage[2 * k] * (1 + 1e-6)), std::max(radius, samples.coverage[2 * k + 1] * (1 + 1e-6)),
                       std::min(radius, .0005)};
                // Domain.clip_scales：4 sigma 支撑留在工程范围内。
                for (int a = 0; a < 2; ++a)
                {
                    const double clearance = report.domain.horizontal_clearance(x, y, a);
                    if (clearance <= 0)
                    {
                        fail_value("Generated water centre outside the project bounds");
                    }
                    s[a] = std::min(s[a], clearance / 4 * (1 - 1e-5));
                    if (s[a] < samples.coverage[2 * k + a] * (1 - 2e-6))
                    {
                        fail_value("Boundary clipping violates planned water coverage; resample the boundary");
                    }
                }
                const double zcap = std::min(z - report.domain.lower[2], report.domain.upper[2] - z) / 4;
                if (zcap <= 0)
                {
                    fail_value("Water height outside project bbox");
                }
                s[2] = std::min(s[2], zcap);
                for (int a = 0; a < 2; ++a)
                {
                    const double e = edge_distance(x, y, a, 4 * s[a]);
                    edge[2 * q + a] = e;
                    before[2 * q + a] = s[a];
                    if (config.water_edge_sigma > 0 && std::isfinite(e))
                    {
                        s[a] = std::min(s[a], std::max(e / config.water_edge_sigma, 0.05));
                    }
                }
                scales[q] = s;
            }
        });
        for (std::size_t q = 0; q < rows.size(); ++q)
        {
            bool unknown = false, limited = false, clamped = false;
            double worst_before = -1e300, worst_after = -1e300;
            for (int a = 0; a < 2; ++a)
            {
                const double e = edge[2 * q + a];
                if (std::isnan(e))
                {
                    unknown = true;
                    continue;
                }
                if (!std::isfinite(e))
                {
                    continue;
                }
                limited = true;
                clamped |= scales[q][static_cast<std::size_t>(a)] < before[2 * q + a];
                worst_before = std::max(worst_before, 2 * before[2 * q + a] - e);
                worst_after = std::max(worst_after, 2 * scales[q][static_cast<std::size_t>(a)] - e);
            }
            edge_unknown += unknown;
            edge_limited += limited;
            edge_clamped += clamped;
            if (worst_before > OVERFLOW_TOLERANCE)
            {
                ++overflow_before;
                overflow_before_sum += worst_before;
                overflow_before_max = std::max(overflow_before_max, worst_before);
            }
            if (worst_after > OVERFLOW_TOLERANCE)
            {
                ++overflow_after;
                overflow_after_sum += worst_after;
                overflow_after_max = std::max(overflow_after_max, worst_after);
            }
        }
        for (std::size_t q = 0; q < rows.size(); ++q)
        {
            const std::size_t k = rows[q];
            new_position.push_back(Vec3{samples.points[3 * k], samples.points[3 * k + 1], samples.points[3 * k + 2]});
            new_scale.push_back(scales[q]);
            new_rgb.push_back(value.second);
        }
    }

    // ---------------- 写回 LOD 最细层叶节点
    const std::size_t leaves = block->leaves.size();
    std::vector<std::vector<std::size_t>> appended(leaves);
    for (std::size_t q = 0; q < new_position.size(); ++q)
    {
        const Vec3& p = new_position[q];
        // 叶节点按 XY 与 Z 一起划分空间（split_points 也会沿 Z 切）：先找完整包含该点的叶，
        // 否则取 XY 包含、Z 最近的叶，再否则取 XY 最近的叶。
        std::size_t best = 0;
        double best_xy = 1e300, best_z = 1e300;
        for (std::size_t l = 0; l < leaves; ++l)
        {
            const Box3& b = block->leaves[l].box;
            const double dx = std::max({b.minimum[0] - p[0], 0.0, p[0] - b.maximum[0]});
            const double dy = std::max({b.minimum[1] - p[1], 0.0, p[1] - b.maximum[1]});
            const double dz = std::max({b.minimum[2] - p[2], 0.0, p[2] - b.maximum[2]});
            const double dxy = dx * dx + dy * dy;
            if (dxy < best_xy || (dxy == best_xy && dz < best_z))
            {
                best_xy = dxy;
                best_z = dz;
                best = l;
                if (dxy == 0 && dz == 0)
                {
                    break;
                }
            }
        }
        appended[best].push_back(q);
    }
    std::vector<std::vector<std::int64_t>> removed_per_leaf(leaves);
    for (std::int64_t r : removed)
    {
        const std::size_t leaf = static_cast<std::size_t>(std::upper_bound(leaf_offsets.begin(), leaf_offsets.end(), static_cast<std::size_t>(r)) - leaf_offsets.begin()) - 1;
        removed_per_leaf[leaf].push_back(r - static_cast<std::int64_t>(leaf_offsets[leaf]));
    }
    Json changes = Json::array();
    std::vector<std::size_t> touched;
    for (std::size_t l = 0; l < leaves; ++l)
    {
        if (!appended[l].empty() || !removed_per_leaf[l].empty())
        {
            touched.push_back(l);
        }
    }
    // 每个叶节点的容量：本块原始最细层叶节点的最大记录数（导出时的 max_splat_num）。超出的水面点放进
    // 同一父节点下新增的兄弟叶节点，避免查看器加载不了超容量的节点、整片停在父层。
    std::uint32_t capacity = GSSP_PAD;
    for (const LodLeaf& leaf : block->leaves)
    {
        capacity = std::max(capacity, read_leaf_padded(lod.root / utf8_path(leaf.uri)));
    }
    std::vector<std::vector<Json>> change_items(touched.size());
    auto box_json6 = [](const Box3& box) {
        return Json{box.minimum[0], box.minimum[1], box.minimum[2], box.maximum[0], box.maximum[1], box.maximum[2]};
    };
    parallel_for_each(touched.size(), [&](std::size_t t) {
        const std::size_t l = touched[t];
        const LodLeaf& leaf = block->leaves[l];
        const LeafContent content = read_leaf(lod.root / utf8_path(leaf.uri));
        const bool semantic = content.semantic || lod.semantic;
        std::vector<std::uint8_t> keep(content.count, 1);
        for (std::int64_t r : removed_per_leaf[l])
        {
            keep[static_cast<std::size_t>(r)] = 0;
        }
        std::vector<std::uint8_t> records;
        records.reserve((content.count + appended[l].size()) * GSSP_RECORD);
        std::uint32_t count = 0;
        for (std::uint32_t r = 0; r < content.count; ++r)
        {
            if (keep[r])
            {
                records.insert(records.end(), content.records.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * GSSP_RECORD),
                               content.records.begin() + static_cast<std::ptrdiff_t>(std::size_t{r + 1} * GSSP_RECORD));
                ++count;
            }
        }
        Box3 box = leaf.box;
        std::size_t next = 0;
        const std::vector<std::size_t>& water = appended[l];
        for (; next < water.size() && count < capacity; ++next)
        {
            std::uint8_t record[GSSP_RECORD];
            const std::size_t q = water[next];
            encode_water_record(record, new_position[q], new_scale[q], new_rgb[q], 0.9, semantic, config.water_class);
            records.insert(records.end(), record, record + GSSP_RECORD);
            box.expand(new_position[q]);
            ++count;
        }
        write_leaf(config.output / utf8_path(leaf.uri), records, count, box, semantic);
        Json item;
        item["uri"] = leaf.uri;
        item["count"] = count;
        item["padded"] = padded_leaf_count(count);
        item["removed"] = removed_per_leaf[l].size();
        item["added"] = next;
        item["box"] = box_json6(box);
        change_items[t].push_back(item);
        const std::string stem = leaf.uri.substr(0, leaf.uri.size() - 4);
        for (int part = 1; next < water.size(); ++part)
        {
            std::vector<std::uint8_t> extra;
            std::uint32_t extra_count = 0;
            Box3 extra_box;
            for (; next < water.size() && extra_count < capacity; ++next)
            {
                std::uint8_t record[GSSP_RECORD];
                const std::size_t q = water[next];
                encode_water_record(record, new_position[q], new_scale[q], new_rgb[q], 0.9, semantic, config.water_class);
                extra.insert(extra.end(), record, record + GSSP_RECORD);
                extra_box.expand(new_position[q]);
                ++extra_count;
            }
            // 水面点同高：给包围盒一点厚度。
            extra_box.minimum[2] -= 0.01;
            extra_box.maximum[2] += 0.01;
            const std::string uri = stem + "_water" + std::to_string(part) + ".bin";
            write_leaf(config.output / utf8_path(uri), extra, extra_count, extra_box, semantic);
            Json sibling;
            sibling["uri"] = uri;
            sibling["sibling_of"] = leaf.uri;
            sibling["count"] = extra_count;
            sibling["padded"] = padded_leaf_count(extra_count);
            sibling["removed"] = 0;
            sibling["added"] = extra_count;
            sibling["box"] = box_json6(extra_box);
            change_items[t].push_back(sibling);
        }
    });
    std::size_t siblings = 0;
    for (std::vector<Json>& items : change_items)
    {
        siblings += items.size() - 1;
        for (Json& item : items)
        {
            changes.push_back(std::move(item));
        }
    }
    if (siblings)
    {
        log("[WRITE] %s: %zu sibling leaves added for water above the %u-point leaf capacity", name.c_str(), siblings, capacity);
    }
    const std::filesystem::path out = config.work / "apply" / "write" / utf8_path(name);
    std::filesystem::create_directories(out);
    if (config.debug)
    {
        NpzWriter writer;
        npz_put(writer, "removed_indices", removed);
        npz_put(writer, "voted_removed_indices", voted);
        npz_put(writer, "carved_indices", carved);
        writer.write(out / "applied.npz", true);
    }
    // 漂浮簇判定明细。
    {
        Json list = Json::array();
        for (const FloaterCandidate& f : extras.candidates)
        {
            auto num = [](double v) { return std::isfinite(v) ? Json(v) : Json(nullptr); };
            Json item;
            item["component"] = f.component;
            item["points"] = f.points;
            item["decision"] = f.decision;
            item["geometry"] = f.geometry;
            item["photo"] = f.photo;
            item["photo_views"] = f.photo_views;
            item["photo_differ_views"] = f.photo_differ;
            item["photo_drop_wins"] = f.photo_wins;
            item["photo_keep_psnr"] = num(f.photo_keep_psnr);
            item["photo_drop_psnr"] = num(f.photo_drop_psnr);
            item["centroid"] = Json{f.centroid[0], f.centroid[1], f.centroid[2]};
            item["low"] = Json{f.low[0], f.low[1], f.low[2]};
            item["high"] = Json{f.high[0], f.high[1], f.high[2]};
            item["gap"] = num(f.gap);
            item["clearance"] = num(f.clearance);
            item["support_top"] = num(f.support_top);
            item["water_level"] = num(f.water_level);
            item["covered"] = f.covered;
            item["spacing"] = f.spacing;
            item["elongation"] = f.elongation;
            item["length"] = f.length;
            item["mean_opacity"] = f.mean_opacity;
            item["semantic"] = f.semantic;
            list.push_back(item);
        }
        Json doc;
        doc["block"] = name;
        doc["mode"] = config.floater_check;
        doc["candidates"] = list;
        write_json_atomic(out / "floaters.json", doc, false);
    }
    // 残留水点按原因汇总；同时看它们最后落在多大的连通簇里（>= floater_max_points 视为与主体粘连）。
    Json residual_stats;
    {
        std::vector<std::uint8_t> gone(n, 0);
        for (std::int64_t r : removed)
        {
            gone[static_cast<std::size_t>(r)] = 1;
        }
        Json reasons = Json::object();
        std::size_t total = 0, total_semantic = 0, total_attached = 0, total_denoised = 0;
        std::vector<std::uint32_t> residual_component(residual.size(), 0);
        for (int why = 0; why < RESIDUAL_REASON_COUNT; ++why)
        {
            std::size_t count = 0, semantic = 0, attached = 0, denoised = 0;
            for (std::size_t k = 0; k < residual.size(); ++k)
            {
                if (residual_reason[k] != why)
                {
                    continue;
                }
                const std::size_t i = static_cast<std::size_t>(residual[k]);
                const std::uint32_t component = extras.component_size.empty() ? 0 : extras.component_size[i];
                residual_component[k] = component;
                ++count;
                semantic += residual_semantic[k];
                if (gone[i])
                {
                    ++denoised;
                }
                else if (component > static_cast<std::uint32_t>(config.floater_max_points))
                {
                    ++attached;
                }
            }
            if (count)
            {
                reasons[RESIDUAL_REASONS[why]] = Json{{"points", count}, {"semantic_water", semantic},
                                                      {"attached_to_large_component", attached}, {"removed_by_denoise", denoised}};
            }
            total += count;
            total_semantic += semantic;
            total_attached += attached;
            total_denoised += denoised;
        }
        residual_stats["points"] = total;
        residual_stats["semantic_water"] = total_semantic;
        residual_stats["attached_to_large_component"] = total_attached;
        residual_stats["removed_by_denoise"] = total_denoised;
        residual_stats["semantic_available"] = has_semantic;
        residual_stats["by_reason"] = reasons;
        if (config.debug)
        {
            NpzWriter writer;
            std::vector<double> position(3 * residual.size());
            for (std::size_t k = 0; k < residual.size(); ++k)
            {
                for (int a = 0; a < 3; ++a)
                {
                    position[3 * k + static_cast<std::size_t>(a)] = g.xyz[3 * static_cast<std::size_t>(residual[k]) + static_cast<std::size_t>(a)];
                }
            }
            npz_put(writer, "indices", residual);
            npz_put(writer, "reason", residual_reason);
            npz_put(writer, "semantic_water", residual_semantic);
            npz_put(writer, "component_size", residual_component);
            npz_put(writer, "xyz", position);
            writer.write(out / "water_residual.npz", true);
        }
    }
    Json edge_stats;
    edge_stats["sigma_limit"] = config.water_edge_sigma;
    edge_stats["generated"] = new_position.size();
    edge_stats["centre_outside_footprint"] = edge_unknown;
    edge_stats["near_edge"] = edge_limited;
    edge_stats["clamped"] = edge_clamped;
    edge_stats["overflow_2sigma_before"] = overflow_before;
    edge_stats["overflow_2sigma_before_mean"] = overflow_before ? overflow_before_sum / static_cast<double>(overflow_before) : 0.0;
    edge_stats["overflow_2sigma_before_max"] = overflow_before_max;
    edge_stats["overflow_2sigma_after"] = overflow_after;
    edge_stats["overflow_2sigma_after_mean"] = overflow_after ? overflow_after_sum / static_cast<double>(overflow_after) : 0.0;
    edge_stats["overflow_2sigma_after_max"] = overflow_after_max;
    Json record;
    record["block"] = name;
    record["count"] = n;
    record["removed"] = removed.size();
    record["voted_removed"] = voted.size();
    record["added"] = new_position.size();
    record["changed_leaves"] = touched.size();
    record["water_carve"] = carve_stats;
    record["denoise"] = denoise_stats;
    record["water_residual"] = residual_stats;
    record["water_edge"] = edge_stats;
    record["changes"] = changes;
    record["reference_shards_read"] = reference_shards_read;
    record["reference_shards_total"] = reference_shards_total;
    record["samples_files_read"] = sample_cache.size() + 1;
    record["seconds"] = total.seconds();
    write_json_atomic(out / "record.json", record, false);
    log("[WRITE] %s: removed %zu (voted %zu, carved %zu), added %zu, %zu/%zu leaves rewritten (%.1fs)", name.c_str(),
        removed.size(), voted.size(), carved.size(), new_position.size(), touched.size(), leaves, total.seconds());
    log("[WRITE] %s: read samples of %zu/%zu blocks, %zu/%zu reference shards", name.c_str(), sample_cache.size() + 1,
        report.blocks.size(), reference_shards_read, reference_shards_total);
}

void run_finalize(const Config& config)
{
    const Timer total;
    std::vector<LeafChange> changes;
    for (const std::string& name : pipeline_blocks(config))
    {
        const Json record = read_json_file(config.work / "apply" / "write" / utf8_path(name) / "record.json");
        for (const Json& item : record.at("changes"))
        {
            LeafChange change;
            change.uri = item.at("uri").get<std::string>();
            change.count = item.at("count").get<std::uint32_t>();
            change.padded = item.at("padded").get<std::uint32_t>();
            change.sibling_of = item.value("sibling_of", std::string());
            for (int k = 0; k < 3; ++k)
            {
                change.box.minimum[k] = jnum(item.at("box")[static_cast<std::size_t>(k)]);
                change.box.maximum[k] = jnum(item.at("box")[static_cast<std::size_t>(3 + k)]);
            }
            changes.push_back(change);
        }
    }
    finalize_lod(config.lod, config.output, changes, config.link_output);
    log("[FINALIZE] %zu rewritten contents, output %s (%.1fs)", changes.size(), path_utf8(config.output).c_str(), total.seconds());
}

} // namespace ai3d::waterpost
