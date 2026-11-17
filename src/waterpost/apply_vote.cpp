// apply 第一段：每块逐水体的足迹投票（distributed_fill.prepare_candidates + body_vote.grid_evidence
// + rebuild.calibrate_views / evidence_hits / _footprint_cells_from_hits），以及全局连通区域支持（reduce_candidates）。

#include "apply_common.hpp"
#include "error.hpp"
#include "fs_util.hpp"
#include "gpu.hpp"
#include "masks.hpp"
#include "npz_util.hpp"
#include "parallel.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ai3d::waterpost
{
namespace
{

constexpr double STRONG_LLR = 2.9444389791664403; // log(19)
constexpr std::size_t MIN_REFERENCE_CELLS = 30;

struct Calibration
{
    std::string mode;
    double p_w = 0, p_l = 0;
    std::size_t water_cells = 0, land_cells = 0;
    Json json() const
    {
        Json v;
        v["mode"] = mode;
        if (mode != "uncalibrated_majority" && mode != "skipped_no_possible_support")
        {
            v["p_w"] = p_w;
            v["p_l"] = p_l;
        }
        v["water_cells"] = water_cells;
        v["land_cells"] = land_cells;
        return v;
    }
};

Calibration calibrate(const std::vector<std::uint16_t>& positive, const std::vector<std::uint16_t>& negative,
                      const std::vector<std::uint8_t>& water_reference, const std::vector<std::uint8_t>& land_reference)
{
    auto rate = [&](const std::vector<std::uint8_t>& rows, std::size_t& cells, double& value) {
        double p = 0, n = 0;
        cells = 0;
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            if (rows[i])
            {
                ++cells;
                p += positive[i];
                n += negative[i];
            }
        }
        if (cells < MIN_REFERENCE_CELLS)
        {
            return false;
        }
        value = (p + 1) / (p + n + 2);
        return true;
    };
    Calibration c;
    const bool has_w = rate(water_reference, c.water_cells, c.p_w);
    const bool has_l = rate(land_reference, c.land_cells, c.p_l);
    if (!has_w && !has_l)
    {
        c.mode = "uncalibrated_majority";
        return c;
    }
    c.mode = "calibrated";
    if (!has_w)
    {
        c.p_w = 1 - c.p_l;
        c.mode = "land_reference_only";
    }
    if (!has_l)
    {
        c.p_l = 1 - c.p_w;
        c.mode = "water_reference_only";
    }
    if (!(c.p_w > c.p_l))
    {
        c.mode = "no_separation";
    }
    return c;
}

std::vector<std::uint8_t> evidence_hits(const std::vector<std::uint16_t>& positive, const std::vector<std::uint16_t>& negative,
                                        const std::vector<std::uint8_t>& barrier, const Calibration& c)
{
    std::vector<std::uint8_t> hits(positive.size(), 0);
    if (c.mode == "no_separation")
    {
        return hits;
    }
    const double pw = std::min(std::max(c.p_w, 1e-6), 1 - 1e-6), pl = std::min(std::max(c.p_l, 1e-6), 1 - 1e-6);
    const double a = std::log(pw / pl), b = std::log((1 - pw) / (1 - pl));
    for (std::size_t i = 0; i < positive.size(); ++i)
    {
        const double pos = positive[i], neg = negative[i];
        bool strong, weak;
        if (c.mode == "uncalibrated_majority")
        {
            strong = pos >= 3 && pos > neg;
            weak = pos >= 2 && pos > neg;
        }
        else
        {
            const double llr = pos * a + neg * b;
            strong = llr >= STRONG_LLR;
            weak = llr > 0;
        }
        hits[i] = barrier[i] ? 0 : (strong ? 3 : (weak ? 2 : 0));
    }
    return hits;
}

// 3x3 二值腐蚀/膨胀，界外按 0。
cv::Mat morph(const cv::Mat& mask, bool dilate)
{
    cv::Mat out;
    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    if (dilate)
    {
        cv::dilate(mask, out, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    else
    {
        cv::erode(mask, out, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    return out;
}

std::vector<double> relevant_occluders(const std::vector<const LandGrid*>& grids, double height, double tolerance,
                                       double spacing, const std::vector<double>& points, const std::vector<View>& views)
{
    std::vector<double> out;
    if (points.empty())
    {
        return out;
    }
    struct Column
    {
        double x, y, zmin, zmax;
    };
    std::vector<Column> columns;
    for (const LandGrid* grid : grids)
    {
        for (std::size_t i = 0; i < grid->count.size(); ++i)
        {
            if (grid->zmax[i] > height + tolerance && grid->count[i] >= 2)
            {
                columns.push_back({static_cast<double>(grid->cells[2 * i]) * spacing,
                                   static_cast<double>(grid->cells[2 * i + 1]) * spacing, grid->zmin[i], grid->zmax[i]});
            }
        }
    }
    if (columns.empty())
    {
        return out;
    }
    Vec3 low{1e300, 1e300, 1e300}, high{-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < points.size() / 3; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            low[k] = std::min(low[k], points[3 * i + k]);
            high[k] = std::max(high[k], points[3 * i + k]);
        }
    }
    double zmax_all = -1e300;
    for (const Column& c : columns)
    {
        zmax_all = std::max(zmax_all, c.zmax);
    }
    bool any_seeing = false, below = false;
    double slack = 0, min_slope = 1e300;
    for (const View& view : views)
    {
        if (!view_sees_box(view, low, high, .25))
        {
            continue;
        }
        any_seeing = true;
        double horizontal = 0;
        for (int c = 0; c < 4; ++c)
        {
            const double x = (c & 1) ? high[0] : low[0], y = (c & 2) ? high[1] : low[1];
            horizontal = std::max(horizontal, std::hypot(view.center[0] - x, view.center[1] - y));
        }
        const double rise = view.center[2] - height;
        const double focal = view.model_id == 1 ? std::min(view.params[0], view.params[1]) : view.params[0];
        const double distance = std::sqrt(horizontal * horizontal + rise * rise);
        slack = std::max(slack, 2 * 12 * distance / std::max(focal, 1e-6));
        if (rise <= 0)
        {
            below = true;
        }
        min_slope = std::min(min_slope, rise / std::max(horizontal, 1e-6));
    }
    if (!any_seeing)
    {
        return out;
    }
    double reach = std::numeric_limits<double>::infinity();
    if (!below)
    {
        reach = (zmax_all - height + slack) / min_slope + slack;
    }
    for (const Column& c : columns)
    {
        if (std::isfinite(reach) && !(c.x >= low[0] - reach - spacing && c.x <= high[0] + reach + spacing &&
                                      c.y >= low[1] - reach - spacing && c.y <= high[1] + reach + spacing))
        {
            continue;
        }
        const double top = c.zmax, bottom = std::max(c.zmin, height + tolerance);
        const long long steps = std::min<long long>(63, std::max<long long>(0, static_cast<long long>(std::ceil((top - bottom) / spacing)))) + 1;
        const double step = std::max((top - bottom) / std::max<long long>(steps - 1, 1), 0.0);
        for (long long j = 0; j < steps; ++j)
        {
            out.push_back(c.x);
            out.push_back(c.y);
            out.push_back(top - static_cast<double>(j) * step);
        }
    }
    return out;
}

// 签名视图的标签图：1 = 该水体的分量，0 = 非水，2 = 其他水。
std::vector<std::uint8_t> signature_image(MaskCache& cache, const View& view, const std::vector<int>& wanted, bool exact_only)
{
    const Labels labels = cache.labels(view);
    std::vector<std::uint8_t> lut(labels.areas.size() + 1, 2);
    lut[0] = 0; // 背景（非水）
    for (int label : wanted)
    {
        if (label >= 0 && static_cast<std::size_t>(label) < lut.size())
        {
            lut[static_cast<std::size_t>(label)] = 1;
        }
    }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(view.width) * view.height);
    for (int y = 0; y < view.height; ++y)
    {
        const int* row = labels.labels.ptr<int>(y);
        for (int x = 0; x < view.width; ++x)
        {
            const int label = row[x];
            image[static_cast<std::size_t>(y) * view.width + x] =
                static_cast<std::size_t>(label) < lut.size() ? lut[static_cast<std::size_t>(label)] : 2;
        }
    }
    return image;
}


// Pass 1 的 CPU 预筛：把水位平面上的投票区域（矩形 lo-hi，z = lo[2]）投到视图里，外包矩形（外扩一个占据格）
// 覆盖的占据格网里没有水格时，这张视图对任何格子都不会投出"可能正票"，可以不上 GPU。平面矩形投影是凸四边形，
// 区域内每个格子的投影都在四个角点投影的外包矩形内；有角点在相机后方时不剪（返回 true）。
bool occupancy_may_support(const View& view, const Occupancy& occupancy, const Vec3& lo, const Vec3& hi)
{
    double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
    for (int c = 0; c < 4; ++c)
    {
        const double p[3] = {(c & 1) ? hi[0] : lo[0], (c & 2) ? hi[1] : lo[1], lo[2]};
        double u = 0, v = 0;
        if (!project_point(view, p, u, v))
        {
            return true;
        }
        umin = std::min(umin, u);
        umax = std::max(umax, u);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    // 近地平线时投影坐标可能极大：先夹住再转 int。
    umin = std::clamp(umin, -1e7, 1e7);
    umax = std::clamp(umax, -1e7, 1e7);
    vmin = std::clamp(vmin, -1e7, 1e7);
    vmax = std::clamp(vmax, -1e7, 1e7);
    const int c0 = std::max(0, static_cast<int>(std::floor(umin / OCCUPANCY_CELL)) - 1);
    const int c1 = std::min(occupancy.cols - 1, static_cast<int>(std::floor(umax / OCCUPANCY_CELL)) + 1);
    const int r0 = std::max(0, static_cast<int>(std::floor(vmin / OCCUPANCY_CELL)) - 1);
    const int r1 = std::min(occupancy.rows - 1, static_cast<int>(std::floor(vmax / OCCUPANCY_CELL)) + 1);
    for (int r = r0; r <= r1; ++r)
    {
        for (int c = c0; c <= c1; ++c)
        {
            if (occupancy.at(r, c))
            {
                return true;
            }
        }
    }
    return false;
}


// 精确复核的 CPU 预筛：精确票要求格子投到该水体分量的像素上，那必然是水像素，所在 32 像素格的占据位为 1。
// 任何一个格子投影后（看所在占据格及其 3x3 邻域，吸收 CPU 双精度与 GPU 单精度投影的差别）都碰不到水格时，
// 这张视图的精确票必然为 0，不必解码掩码、生成标签图。
bool rows_may_hit_water(const View& view, const Occupancy& occupancy, const std::vector<double>& points)
{
    for (std::size_t k = 0; k < points.size() / 3; ++k)
    {
        double u = 0, v = 0;
        if (!project_point(view, points.data() + 3 * k, u, v))
        {
            continue;
        }
        if (!(u > -2 && v > -2 && u < view.width + 2 && v < view.height + 2))
        {
            continue;
        }
        const int c = static_cast<int>(std::floor(u / OCCUPANCY_CELL)), r = static_cast<int>(std::floor(v / OCCUPANCY_CELL));
        for (int dr = -1; dr <= 1; ++dr)
        {
            for (int dc = -1; dc <= 1; ++dc)
            {
                const int rr = r + dr, cc = c + dc;
                if (rr >= 0 && cc >= 0 && rr < occupancy.rows && cc < occupancy.cols && occupancy.at(rr, cc))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

void run_candidates(const Config& config, const std::string& name)
{
    const Timer total;
    const Report report = load_report(config);
    const BlockRecord& record = report.block(name);
    const GridSpec& grid = report.grid;
    const double spacing = grid.spacing;
    const std::filesystem::path out = config.work / "apply" / "candidates" / utf8_path(name);
    std::filesystem::create_directories(out);
    const std::vector<View> views = load_catalog(config);

    double low[2], high[2];
    for (int a = 0; a < 2; ++a)
    {
        low[a] = std::max(record.bounds.minimum[a], report.domain.lower[a]);
        high[a] = std::min(record.bounds.maximum[a], report.domain.upper[a]);
    }
    const bool has_region = high[0] > low[0] && high[1] > low[1];
    std::vector<std::int64_t> all_cells;
    std::vector<std::int32_t> all_ids;
    std::vector<std::uint8_t> all_support;
    Json bodies = Json::object();

    std::vector<double> seeds;
    std::vector<LandGrid> lands;
    std::unordered_map<std::string, Occupancy> occupancy;
    if (has_region && !report.surfaces.empty())
    {
        const double reach = std::max(high[0] - low[0], high[1] - low[1]);
        for (const BlockRecord& other : report.blocks)
        {
            bool far = false;
            for (int a = 0; a < 2; ++a)
            {
                far |= other.bounds.minimum[a] - reach > high[a] + spacing || other.bounds.maximum[a] + reach < low[a] - spacing;
            }
            if (far)
            {
                continue;
            }
            if (other.water)
            {
                const std::vector<double> s = load_seed_xyz(config, other.name);
                seeds.insert(seeds.end(), s.begin(), s.end());
            }
            LandGrid land;
            if (load_land_grid(config, other.name, land))
            {
                lands.push_back(std::move(land));
            }
            load_occupancy(config, other.name, occupancy);
        }
    }
    std::vector<const LandGrid*> land_ptrs;
    for (const LandGrid& l : lands)
    {
        land_ptrs.push_back(&l);
    }
    std::unordered_map<std::string, std::size_t> view_index;
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        view_index[views[i].mask_key] = i;
    }
    MaskCache& cache = shared_mask_cache(config.water_class);
    const std::size_t decodes_before = cache.decodes();
    if (has_region && !report.surfaces.empty())
    {
        gpu_select_device(config.gpu);
    }
    const std::int64_t origin_index[2] = {static_cast<std::int64_t>(std::nearbyint(grid.origin[0] / spacing)),
                                          static_cast<std::int64_t>(std::nearbyint(grid.origin[1] / spacing))};

    std::size_t skipped_bodies = 0, culled_views = 0, total_sig_views = 0, exact_culled = 0, exact_total = 0;
    double prof_pass1 = 0, prof_images = 0, prof_exact = 0;
    for (const Surface& surface : report.surfaces)
    {
        if (!has_region)
        {
            break;
        }
        const Timer body_timer;
        const int wid = surface.water_id;
        const double height = surface.height;
        std::int64_t core_low[2], core_high[2], patch_low[2], patch_high[2], ps[2];
        bool empty = false;
        for (int a = 0; a < 2; ++a)
        {
            core_low[a] = std::max<std::int64_t>(static_cast<std::int64_t>(std::floor((low[a] - spacing / 2 - grid.origin[a]) / spacing)) + 1, 0);
            core_high[a] = std::min<std::int64_t>(static_cast<std::int64_t>(std::ceil((high[a] + spacing / 2 - grid.origin[a]) / spacing)), grid.shape[a]);
            empty |= core_high[a] <= core_low[a];
            patch_low[a] = std::max<std::int64_t>(core_low[a] - 2, 0);
            patch_high[a] = std::min<std::int64_t>(core_high[a] + 2, grid.shape[a]);
            ps[a] = patch_high[a] - patch_low[a];
        }
        if (empty)
        {
            continue;
        }
        // 签名视图（看得到本块、且该水体在其中有分量的视图）：正票只来自它们，一张都没有就不可能有候选格，
        // 整个水体直接跳过（不建屏障与参照格）。只查该水体签名里的视图，不扫全部视图。
        const Vec3 patch_lo{grid.origin[0] + patch_low[0] * spacing, grid.origin[1] + patch_low[1] * spacing, height};
        const Vec3 patch_hi{grid.origin[0] + (patch_high[0] - 1) * spacing, grid.origin[1] + (patch_high[1] - 1) * spacing, height};
        std::vector<std::size_t> sig_views;
        for (const auto& [key, labels] : surface.signature)
        {
            const auto found = view_index.find(key);
            if (!labels.empty() && found != view_index.end() && view_sees_box(views[found->second], patch_lo, patch_hi, .25))
            {
                sig_views.push_back(found->second);
            }
        }
        std::sort(sig_views.begin(), sig_views.end());
        if (sig_views.empty())
        {
            Json skipped;
            skipped["cells"] = 0;
            skipped["bbox"] = nullptr;
            skipped["skipped"] = "no_signature_view_sees_block";
            bodies[std::to_string(wid)] = skipped;
            ++skipped_bodies;
            continue;
        }
        const std::size_t count = static_cast<std::size_t>(ps[0] * ps[1]);
        auto local_row = [&](std::int64_t gx, std::int64_t gy, std::size_t& row) {
            const std::int64_t lx = gx - patch_low[0], ly = gy - patch_low[1];
            if (lx < 0 || ly < 0 || lx >= ps[0] || ly >= ps[1])
            {
                return false;
            }
            row = static_cast<std::size_t>(lx * ps[1] + ly);
            return true;
        };
        auto cell_point = [&](std::size_t row, double* p) {
            p[0] = static_cast<double>(static_cast<std::int64_t>(row) / ps[1] + patch_low[0]) * spacing + grid.origin[0];
            p[1] = static_cast<double>(static_cast<std::int64_t>(row) % ps[1] + patch_low[1]) * spacing + grid.origin[1];
            p[2] = height;
        };
        const double tolerance = std::max(3 * surface.sigma, spacing);
        std::vector<std::uint8_t> barrier(count, 0), water_reference(count, 0);
        for (const LandGrid& land : lands)
        {
            for (std::size_t i = 0; i < land.count.size(); ++i)
            {
                std::size_t row;
                if (land.zmin[i] <= height + tolerance && land.count[i] >= 2 &&
                    local_row(land.cells[2 * i] - origin_index[0], land.cells[2 * i + 1] - origin_index[1], row))
                {
                    barrier[row] = 1;
                }
            }
        }
        for (std::size_t s = 0; s < seeds.size() / 3; ++s)
        {
            std::size_t row;
            if (local_row(static_cast<std::int64_t>(std::nearbyint((seeds[3 * s] - grid.origin[0]) / spacing)),
                          static_cast<std::int64_t>(std::nearbyint((seeds[3 * s + 1] - grid.origin[1]) / spacing)), row))
            {
                water_reference[row] = 1;
            }
        }
        std::vector<std::uint8_t> land_reference(count, 0);
        {
            const cv::Mat b(static_cast<int>(ps[0]), static_cast<int>(ps[1]), CV_8U, barrier.data());
            const cv::Mat interior = morph(b, false);
            std::size_t total_interior = static_cast<std::size_t>(cv::countNonZero(interior));
            if (total_interior >= 30)
            {
                std::memcpy(land_reference.data(), interior.data, count);
            }
            else
            {
                land_reference = barrier;
            }
        }

        // Pass 1：可能的正票（签名视图，占据格网超集，再用精确标签收紧），不做遮挡。
        Json summary;
        std::vector<std::uint8_t> possible(count, 0);
        std::size_t possible_count = 0;
        double support_seconds = 0;
        {
            const Timer t;
            // 先在 CPU 上剪掉对本块不可能投正票的视图（无水，或投票区域落不到水格上）；全部剪掉就不上 GPU。
            std::vector<std::size_t> pass_views;
            for (std::size_t i : sig_views)
            {
                const auto entry = occupancy.find(views[i].mask_key);
                if (entry != occupancy.end() &&
                    (entry->second.water_pixels == 0 || !occupancy_may_support(views[i], entry->second, patch_lo, patch_hi)))
                {
                    continue;
                }
                pass_views.push_back(i);
            }
            culled_views += sig_views.size() - pass_views.size();
            total_sig_views += sig_views.size();
            if (!pass_views.empty())
            {
                std::vector<double> points(3 * count);
                for (std::size_t r = 0; r < count; ++r)
                {
                    cell_point(r, points.data() + 3 * r);
                }
                const Timer t_pass1;
                GridGpu pass(points, {}, 4);
                for (std::size_t i : pass_views)
                {
                    const auto entry = occupancy.find(views[i].mask_key);
                    pass.possible(pinhole(views[i]), entry == occupancy.end() ? nullptr : &entry->second);
                }
                std::vector<std::uint32_t> support, against;
                std::size_t occluded = 0;
                pass.download(support, against, occluded);
                prof_pass1 += t_pass1.seconds();
                std::vector<std::size_t> rows;
                for (std::size_t r = 0; r < count; ++r)
                {
                    if (support[r])
                    {
                        rows.push_back(r);
                    }
                }
                if (!rows.empty())
                {
                    std::vector<double> subset(3 * rows.size());
                    Vec3 lo{1e300, 1e300, height}, hi{-1e300, -1e300, height};
                    for (std::size_t k = 0; k < rows.size(); ++k)
                    {
                        cell_point(rows[k], subset.data() + 3 * k);
                        for (int a = 0; a < 2; ++a)
                        {
                            lo[a] = std::min(lo[a], subset[3 * k + a]);
                            hi[a] = std::max(hi[a], subset[3 * k + a]);
                        }
                    }
                    GridGpu exact(subset, {}, 4);
                    std::vector<std::size_t> near;
                    for (std::size_t i : sig_views)
                    {
                        if (view_sees_box(views[i], lo, hi, .25))
                        {
                            near.push_back(i);
                        }
                    }
                    std::vector<std::uint8_t> keep(near.size(), 1);
                    parallel_for_each(near.size(), [&](std::size_t j) {
                        const auto entry = occupancy.find(views[near[j]].mask_key);
                        if (entry != occupancy.end() && !rows_may_hit_water(views[near[j]], entry->second, subset))
                        {
                            keep[j] = 0;
                        }
                    });
                    std::vector<std::size_t> used;
                    for (std::size_t j = 0; j < near.size(); ++j)
                    {
                        if (keep[j])
                        {
                            used.push_back(near[j]);
                        }
                    }
                    exact_culled += near.size() - used.size();
                    exact_total += near.size();
                    // 标签图在 CPU 上按批并行准备，GPU 逐张计票。
                    for (std::size_t start = 0; start < used.size(); start += 16)
                    {
                        const std::size_t end = std::min(used.size(), start + 16);
                        std::vector<std::vector<std::uint8_t>> images(end - start);
                        const Timer t_images;
                        parallel_for_limited(end - start, 16, [&](std::size_t j) {
                            const View& view = views[used[start + j]];
                            images[j] = signature_image(cache, view, surface.signature.at(view.mask_key), true);
                        });
                        prof_images += t_images.seconds();
                        const Timer t_exact;
                        for (std::size_t j = 0; j < images.size(); ++j)
                        {
                            exact.exact(pinhole(views[used[start + j]]), images[j].data());
                        }
                        prof_exact += t_exact.seconds();
                    }
                    const Timer t_down;
                    exact.download(support, against, occluded);
                    prof_exact += t_down.seconds();
                    for (std::size_t k = 0; k < rows.size(); ++k)
                    {
                        if (support[k])
                        {
                            possible[rows[k]] = 1;
                            ++possible_count;
                        }
                    }
                }
            }
            support_seconds = t.seconds();
        }
        summary["catalog_views"] = views.size();
        summary["support_pass_views"] = sig_views.size();
        summary["possible_cells"] = possible_count;
        std::vector<std::uint16_t> positive(count, 0), negative(count, 0);
        Calibration calibration;
        std::vector<std::int64_t> body_cells;
        std::size_t three_votes = 0, rejected = 0;
        if (!possible_count)
        {
            calibration.mode = "skipped_no_possible_support";
            summary["evaluated_cells"] = 0;
        }
        else
        {
            // Pass 2：可能正票 + 校准参照格，签名视图正反票 + 其他视图占据反票，带遮挡。
            const Timer t;
            std::vector<std::size_t> evaluated;
            for (std::size_t r = 0; r < count; ++r)
            {
                if (possible[r] || water_reference[r] || land_reference[r])
                {
                    evaluated.push_back(r);
                }
            }
            std::vector<double> points(3 * evaluated.size());
            Vec3 lo{1e300, 1e300, height}, hi{-1e300, -1e300, height};
            for (std::size_t k = 0; k < evaluated.size(); ++k)
            {
                cell_point(evaluated[k], points.data() + 3 * k);
                for (int a = 0; a < 2; ++a)
                {
                    lo[a] = std::min(lo[a], points[3 * k + a]);
                    hi[a] = std::max(hi[a], points[3 * k + a]);
                }
            }
            const std::vector<double> occluders = relevant_occluders(land_ptrs, height, tolerance, spacing, points, views);
            GridGpu pass(points, occluders, 4);
            std::size_t sig_used = 0, other_used = 0;
            std::vector<std::size_t> signature_views, other_views;
            for (std::size_t i = 0; i < views.size(); ++i)
            {
                if (!view_sees_box(views[i], lo, hi, .25))
                {
                    continue;
                }
                const auto sig = surface.signature.find(views[i].mask_key);
                (sig != surface.signature.end() && !sig->second.empty() ? signature_views : other_views).push_back(i);
            }
            for (std::size_t start = 0; start < signature_views.size(); start += 16)
            {
                const std::size_t end = std::min(signature_views.size(), start + 16);
                std::vector<std::vector<std::uint8_t>> images(end - start);
                parallel_for_limited(end - start, 16, [&](std::size_t j) {
                    const View& view = views[signature_views[start + j]];
                    images[j] = signature_image(cache, view, surface.signature.at(view.mask_key), false);
                });
                for (std::size_t j = 0; j < images.size(); ++j)
                {
                    pass.signature(pinhole(views[signature_views[start + j]]), images[j].data(), 2 * spacing);
                    ++sig_used;
                }
            }
            for (std::size_t i : other_views)
            {
                const auto entry = occupancy.find(views[i].mask_key);
                if (entry == occupancy.end())
                {
                    continue;
                }
                pass.other(pinhole(views[i]), entry->second.water_pixels == 0, &entry->second, 2 * spacing);
                ++other_used;
            }
            std::vector<std::uint32_t> support, against;
            std::size_t occluded = 0;
            pass.download(support, against, occluded);
            for (std::size_t k = 0; k < evaluated.size(); ++k)
            {
                positive[evaluated[k]] = static_cast<std::uint16_t>(std::min<std::uint32_t>(support[k], 65535));
                negative[evaluated[k]] = static_cast<std::uint16_t>(std::min<std::uint32_t>(against[k], 65535));
            }
            std::vector<std::uint8_t> water_ref_clean(count, 0);
            for (std::size_t r = 0; r < count; ++r)
            {
                water_ref_clean[r] = water_reference[r] && !barrier[r];
            }
            calibration = calibrate(positive, negative, water_ref_clean, land_reference);
            const std::vector<std::uint8_t> hits = evidence_hits(positive, negative, barrier, calibration);
            cv::Mat strong(static_cast<int>(ps[0]), static_cast<int>(ps[1]), CV_8U);
            for (std::size_t r = 0; r < count; ++r)
            {
                strong.data[r] = hits[r] >= 3 ? 1 : 0;
                three_votes += positive[r] >= 3 ? 1 : 0;
                rejected += (positive[r] >= 3 && hits[r] < 3) ? 1 : 0;
            }
            const cv::Mat closed = morph(morph(strong, true), false);
            for (std::size_t r = 0; r < count; ++r)
            {
                if (hits[r] >= 3 || (closed.data[r] && hits[r] >= 2))
                {
                    body_cells.push_back(static_cast<std::int64_t>(r) / ps[1] + patch_low[0]);
                    body_cells.push_back(static_cast<std::int64_t>(r) % ps[1] + patch_low[1]);
                }
            }
            summary["evaluated_cells"] = evaluated.size();
            summary["signature_views"] = sig_used;
            summary["occupancy_views"] = other_used;
            summary["occluded_observations"] = occluded;
            summary["occluder_samples"] = occluders.size() / 3;
            summary["evidence_seconds"] = t.seconds();
        }
        // 核心区与块区域内的格子；每格是否有原水点（0.72 格半径）。
        std::vector<std::int64_t> cells;
        for (std::size_t k = 0; k < body_cells.size() / 2; ++k)
        {
            const std::int64_t gx = body_cells[2 * k], gy = body_cells[2 * k + 1];
            if (gx < core_low[0] || gx >= core_high[0] || gy < core_low[1] || gy >= core_high[1])
            {
                continue;
            }
            const double x = grid.origin[0] + gx * spacing, y = grid.origin[1] + gy * spacing;
            if (!(x + spacing / 2 > low[0] && y + spacing / 2 > low[1] && x - spacing / 2 < high[0] && y - spacing / 2 < high[1]))
            {
                continue;
            }
            if (!report.domain.inside(x, y))
            {
                continue;
            }
            cells.push_back(gx);
            cells.push_back(gy);
        }
        const std::size_t ncells = cells.size() / 2;
        std::vector<std::uint8_t> support(ncells, 0);
        Box3 bbox;
        if (ncells)
        {
            std::unordered_map<std::uint64_t, std::size_t> lookup;
            for (std::size_t k = 0; k < ncells; ++k)
            {
                lookup[cell_key(cells[2 * k], cells[2 * k + 1])] = k;
                bbox.expand(Vec3{grid.origin[0] + cells[2 * k] * spacing, grid.origin[1] + cells[2 * k + 1] * spacing, 0});
            }
            const double radius = spacing * .72;
            for (std::size_t s = 0; s < seeds.size() / 3; ++s)
            {
                const double sx = seeds[3 * s], sy = seeds[3 * s + 1];
                const std::int64_t cx = static_cast<std::int64_t>(std::floor((sx - grid.origin[0]) / spacing));
                const std::int64_t cy = static_cast<std::int64_t>(std::floor((sy - grid.origin[1]) / spacing));
                for (std::int64_t dx = -1; dx <= 2; ++dx)
                {
                    for (std::int64_t dy = -1; dy <= 2; ++dy)
                    {
                        const auto found = lookup.find(cell_key(cx + dx, cy + dy));
                        if (found == lookup.end())
                        {
                            continue;
                        }
                        const double x = grid.origin[0] + (cx + dx) * spacing, y = grid.origin[1] + (cy + dy) * spacing;
                        if (std::hypot(x - sx, y - sy) <= radius)
                        {
                            support[found->second] = 1;
                        }
                    }
                }
            }
        }
        std::size_t supported = 0;
        for (std::uint8_t s : support)
        {
            supported += s;
        }
        summary["cells"] = ncells;
        summary["original_support"] = supported > 0;
        summary["supported_cells"] = supported;
        if (ncells)
        {
            Json b;
            b["minimum"] = {bbox.minimum[0], bbox.minimum[1]};
            b["maximum"] = {bbox.maximum[0], bbox.maximum[1]};
            summary["bbox"] = b;
        }
        else
        {
            summary["bbox"] = nullptr;
        }
        summary["calibration"] = calibration.json();
        summary["barrier_cells"] = static_cast<std::size_t>(std::count(barrier.begin(), barrier.end(), 1));
        summary["three_vote_cells"] = three_votes;
        summary["counter_evidence_rejected_cells"] = rejected;
        summary["level_tolerance"] = tolerance;
        summary["support_seconds"] = support_seconds;
        summary["seconds"] = body_timer.seconds();
        bodies[std::to_string(wid)] = summary;
        all_cells.insert(all_cells.end(), cells.begin(), cells.end());
        all_ids.insert(all_ids.end(), ncells, wid);
        all_support.insert(all_support.end(), support.begin(), support.end());
        log("[CANDIDATES] %s body %d: %zu cells (%s), possible %zu, %.1fs", name.c_str(), wid, ncells,
            calibration.mode.c_str(), possible_count, body_timer.seconds());
    }
    NpzWriter writer;
    npz_put(writer, "cells", all_cells, {all_cells.size() / 2, 2});
    npz_put(writer, "water_ids", all_ids);
    npz_put(writer, "support", all_support);
    writer.write(out / "candidates.npz", true);
    Json metadata;
    metadata["block"] = name;
    metadata["bodies"] = bodies;
    metadata["seconds"] = total.seconds();
    write_json_atomic(out / "metadata.json", metadata, false);
    log("[CANDIDATES] %s complete: %zu cells, %zu of %zu bodies skipped (no signature view sees the block), "
        "views culled on the CPU: support pass %zu/%zu, exact pass %zu/%zu, %.1fs",
        name.c_str(), all_ids.size(), skipped_bodies, report.surfaces.size(), culled_views, total_sig_views, exact_culled,
        exact_total, total.seconds());
    log("[CANDIDATES] %s support pass: occupancy votes %.2fs, label images %.2fs, exact votes %.2fs (decodes %zu)", name.c_str(),
        prof_pass1, prof_images, prof_exact, cache.decodes() - decodes_before);
}

// ---------------------------------------------------------------- active

namespace
{

// 8 连通区域；返回每格的区域号。
std::vector<std::size_t> regions8(const std::vector<std::int64_t>& cells)
{
    const std::size_t n = cells.size() / 2;
    std::unordered_map<std::uint64_t, std::size_t> lookup;
    lookup.reserve(n * 2);
    for (std::size_t k = 0; k < n; ++k)
    {
        lookup[cell_key(cells[2 * k], cells[2 * k + 1])] = k;
    }
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<std::size_t(std::size_t)> find = [&](std::size_t a) {
        while (parent[a] != a)
        {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    const int offsets[4][2] = {{1, -1}, {1, 0}, {1, 1}, {0, 1}};
    for (std::size_t k = 0; k < n; ++k)
    {
        for (const auto& o : offsets)
        {
            const auto found = lookup.find(cell_key(cells[2 * k] + o[0], cells[2 * k + 1] + o[1]));
            if (found != lookup.end())
            {
                const std::size_t a = find(k), b = find(found->second);
                if (a != b)
                {
                    parent[a] = b;
                }
            }
        }
    }
    std::vector<std::size_t> roots(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        roots[k] = find(k);
    }
    return roots;
}

} // namespace

void run_active(const Config& config)
{
    const Timer total;
    const Report report = load_report(config);
    const std::filesystem::path out = config.work / "apply" / "active";
    std::filesystem::create_directories(out);
    std::map<int, std::vector<std::int64_t>> cells;
    std::map<int, std::vector<std::uint8_t>> support;
    std::map<int, std::size_t> candidate_cells;
    for (const BlockRecord& block : report.blocks)
    {
        const Candidates c = load_candidates(config, block.name);
        for (std::size_t k = 0; k < c.ids.size(); ++k)
        {
            cells[c.ids[k]].push_back(c.cells[2 * k]);
            cells[c.ids[k]].push_back(c.cells[2 * k + 1]);
            support[c.ids[k]].push_back(c.support[k]);
            ++candidate_cells[c.ids[k]];
        }
    }
    Json bodies = Json::object();
    std::vector<std::int64_t> drop_cells;
    std::vector<std::int32_t> drop_ids;
    for (const Surface& surface : report.surfaces)
    {
        const int wid = surface.water_id;
        Json value;
        value["candidate_cells"] = candidate_cells[wid];
        if (!cells.count(wid))
        {
            value["active"] = false;
            value["region_rule"] = "connected_region_support";
            bodies[std::to_string(wid)] = value;
            continue;
        }
        // 去重（接缝光环里的重复格），支持取或。
        std::unordered_map<std::uint64_t, std::size_t> unique;
        std::vector<std::int64_t> merged;
        std::vector<std::uint8_t> flags;
        const std::vector<std::int64_t>& c = cells[wid];
        for (std::size_t k = 0; k < c.size() / 2; ++k)
        {
            const auto [it, inserted] = unique.emplace(cell_key(c[2 * k], c[2 * k + 1]), merged.size() / 2);
            if (inserted)
            {
                merged.push_back(c[2 * k]);
                merged.push_back(c[2 * k + 1]);
                flags.push_back(0);
            }
            flags[it->second] |= support[wid][k];
        }
        const std::vector<std::size_t> roots = regions8(merged);
        std::unordered_set<std::size_t> supported;
        for (std::size_t k = 0; k < roots.size(); ++k)
        {
            if (flags[k])
            {
                supported.insert(roots[k]);
            }
        }
        std::size_t keep = 0, drop = 0;
        for (std::size_t k = 0; k < roots.size(); ++k)
        {
            if (supported.count(roots[k]))
            {
                ++keep;
            }
            else
            {
                ++drop;
                drop_cells.push_back(merged[2 * k]);
                drop_cells.push_back(merged[2 * k + 1]);
                drop_ids.push_back(wid);
            }
        }
        value["active"] = keep > 0;
        value["region_rule"] = "connected_region_support";
        value["supported_region_cells"] = keep;
        value["unsupported_region_cells"] = drop;
        bodies[std::to_string(wid)] = value;
    }
    NpzWriter writer;
    npz_put(writer, "cells", drop_cells, {drop_cells.size() / 2, 2});
    npz_put(writer, "water_ids", drop_ids);
    writer.write(out / "unsupported.npz", true);
    Json result;
    result["bodies"] = bodies;
    result["seconds"] = total.seconds();
    write_json_atomic(out / "active.json", result, false);
    std::size_t active = 0;
    for (auto it = bodies.begin(); it != bodies.end(); ++it)
    {
        active += it.value().at("active").get<bool>() ? 1 : 0;
    }
    log("[ACTIVE] %zu/%zu bodies active, %zu unsupported region cells (%.1fs)", active, report.surfaces.size(),
        drop_ids.size(), total.seconds());
}

} // namespace ai3d::waterpost
