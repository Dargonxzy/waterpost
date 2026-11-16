// detect：一个块的多视图二值投票、可见 T*alpha 支持、岸线锚点、可靠陆地格网与占据格网
// （water_postprocessing.workflow_stages.stage_detect + mask_voting + prepared_components）。

#include "error.hpp"
#include "fs_util.hpp"
#include "gpu.hpp"
#include "lod.hpp"
#include "masks.hpp"
#include "npz_util.hpp"
#include "parallel.hpp"
#include "project.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <numeric>

namespace ai3d::waterpost
{
namespace
{

struct VotePolicy
{
    int min_visible_views = 4;
    int min_water_views = 3;
    double water_hit_ratio = .6;
    double seed_hit_ratio = .75;
    int min_land_views = 2;
    double land_hit_ratio = .75;
    double support_water_fraction = .9;
    int min_shore_visible = 3;
    int min_shore_views = 2;
};

// GaussianEnvelope：4x4x4 空间箱，每箱中心点 AABB 与最大方差。
struct Envelope
{
    std::vector<Vec3> low, high;
    std::vector<double> variance;

    explicit Envelope(const Gaussians& g)
    {
        if (!g.count)
        {
            return;
        }
        Vec3 lower{1e300, 1e300, 1e300}, upper{-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < g.count; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                lower[k] = std::min(lower[k], g.xyz[3 * i + k]);
                upper[k] = std::max(upper[k], g.xyz[3 * i + k]);
            }
        }
        const int d = 4;
        std::vector<Vec3> lo(d * d * d, Vec3{1e300, 1e300, 1e300}), hi(d * d * d, Vec3{-1e300, -1e300, -1e300});
        std::vector<double> var(d * d * d, -1.0);
        for (std::size_t i = 0; i < g.count; ++i)
        {
            int cell[3];
            for (int k = 0; k < 3; ++k)
            {
                const double span = std::max(upper[k] - lower[k], 1e-30);
                cell[k] = std::min(static_cast<int>((g.xyz[3 * i + k] - lower[k]) / span * d), d - 1);
            }
            const int id = (cell[0] * d + cell[1]) * d + cell[2];
            for (int k = 0; k < 3; ++k)
            {
                lo[id][k] = std::min(lo[id][k], g.xyz[3 * i + k]);
                hi[id][k] = std::max(hi[id][k], g.xyz[3 * i + k]);
            }
            var[id] = std::max(var[id], static_cast<double>(g.sigma_max[i]) * g.sigma_max[i]);
        }
        for (int id = 0; id < d * d * d; ++id)
        {
            if (var[id] >= 0)
            {
                low.push_back(lo[id]);
                high.push_back(hi[id]);
                variance.push_back(var[id] * 1.001);
            }
        }
    }

    // 返回 false 表示退回整张图。
    bool rectangles(const View& view, std::vector<std::array<double, 4>>& rects) const
    {
        rects.clear();
        const PinholeCamera cam = pinhole(view);
        const double near = 1e-6;
        for (std::size_t b = 0; b < low.size(); ++b)
        {
            double zmin = 1e300, zmax = -1e300;
            double camera_points[8][3];
            for (int c = 0; c < 8; ++c)
            {
                const double p[3] = {(c & 1) ? high[b][0] : low[b][0], (c & 2) ? high[b][1] : low[b][1],
                                     (c & 4) ? high[b][2] : low[b][2]};
                const double d[3] = {p[0] - cam.center[0], p[1] - cam.center[1], p[2] - cam.center[2]};
                for (int a = 0; a < 3; ++a)
                {
                    camera_points[c][a] = cam.r[3 * a] * d[0] + cam.r[3 * a + 1] * d[1] + cam.r[3 * a + 2] * d[2];
                }
                zmin = std::min(zmin, camera_points[c][2]);
                zmax = std::max(zmax, camera_points[c][2]);
            }
            const bool visible = zmax >= near * (1 - 1e-5);
            if (visible && zmin <= near * (1 + 1e-5))
            {
                return false;
            }
            if (!visible)
            {
                continue;
            }
            const double z0 = zmin * (1 - 1e-6);
            double u0 = 1e300, v0 = 1e300, u1 = -1e300, v1 = -1e300;
            for (int c = 0; c < 8; ++c)
            {
                const double u = camera_points[c][0] / camera_points[c][2] * cam.fx + cam.cx - .5;
                const double v = camera_points[c][1] / camera_points[c][2] * cam.fy + cam.cy - .5;
                u0 = std::min(u0, u);
                u1 = std::max(u1, u);
                v0 = std::min(v0, v);
                v1 = std::max(v1, v);
            }
            const double jacobian = (cam.fx * cam.fx + std::pow(1.3 * view.width / 2, 2) + cam.fy * cam.fy +
                                     std::pow(1.3 * view.height / 2, 2)) / (z0 * z0);
            const double radius = std::ceil(3 * std::sqrt(variance[b] * jacobian + std::sqrt(.1))) + 32;
            const std::array<double, 4> rect = {u0 - radius, v0 - radius, u1 + radius, v1 + radius};
            if (rect[0] < view.width && rect[1] < view.height && rect[2] >= 0 && rect[3] >= 0)
            {
                rects.push_back(rect);
            }
        }
        return true;
    }
};

bool roi_hits(const Occupancy& occupancy, const std::vector<std::array<double, 4>>& rects)
{
    for (const auto& r : rects)
    {
        const int left = std::max(0, static_cast<int>(std::floor(r[0] / OCCUPANCY_CELL)));
        const int top = std::max(0, static_cast<int>(std::floor(r[1] / OCCUPANCY_CELL)));
        const int right = std::min(occupancy.cols, static_cast<int>(std::floor(r[2] / OCCUPANCY_CELL)) + 1);
        const int bottom = std::min(occupancy.rows, static_cast<int>(std::floor(r[3] / OCCUPANCY_CELL)) + 1);
        for (int y = top; y < bottom; ++y)
        {
            for (int x = left; x < right; ++x)
            {
                if (occupancy.at(y, x))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

struct PreparedView
{
    bool dry = true;
    std::size_t rejected = 0;
    std::vector<std::uint8_t> state;
    std::vector<float> distance;
    std::vector<std::uint8_t> shore;
};

// _prepare_view：水图 → 天空修正 → 有符号距离场 → 外环岸线。
PreparedView prepare_view(const View& view, MaskCache& cache)
{
    PreparedView out;
    const PinholeCamera cam = pinhole(view);
    const cv::Mat water = cache.water(view);
    const int width = view.width, height = view.height;
    cv::Mat state = water.clone();
    const double r2 = cam.r[2], r5 = cam.r[5], r8 = cam.r[8];
    bool any = false;
    for (int y = 0; y < height; ++y)
    {
        std::uint8_t* row = state.ptr<std::uint8_t>(y);
        const double yy = (y + .5 - cam.cy) / cam.fy;
        for (int x = 0; x < width; ++x)
        {
            if (!row[x])
            {
                continue;
            }
            const double xx = (x + .5 - cam.cx) / cam.fx;
            if (xx * r2 + yy * r5 + r8 >= 0)
            {
                row[x] = 0;
                ++out.rejected;
            }
            else
            {
                any = true;
            }
        }
    }
    if (!any)
    {
        return out;
    }
    out.dry = false;
    cv::Mat wet = state * 255;
    cv::Mat land = (state == 0);
    auto transform = [](const cv::Mat& mask) {
        cv::Mat padded;
        cv::copyMakeBorder(mask, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat distance;
        cv::distanceTransform(padded, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);
        return distance(cv::Rect(1, 1, mask.cols, mask.rows)).clone();
    };
    const cv::Mat d_wet = transform(wet);
    const cv::Mat d_land = transform(land);
    const int radius = boundary_pixels(width, height);
    cv::Mat near;
    cv::dilate(wet, near, ellipse_kernel(radius), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    out.state.resize(pixels);
    out.distance.resize(pixels);
    out.shore.resize(pixels);
    for (int y = 0; y < height; ++y)
    {
        const std::uint8_t* s = state.ptr<std::uint8_t>(y);
        const std::uint8_t* w = water.ptr<std::uint8_t>(y);
        const std::uint8_t* n = near.ptr<std::uint8_t>(y);
        const float* a = d_wet.ptr<float>(y);
        const float* b = d_land.ptr<float>(y);
        for (int x = 0; x < width; ++x)
        {
            const std::size_t at = static_cast<std::size_t>(y) * width + x;
            out.state[at] = s[x];
            out.distance[at] = a[x] - b[x];
            out.shore[at] = (n[x] && !w[x] && !s[x]) ? 1 : 0;
        }
    }
    return out;
}

// stable_shore：低、不透明、局部一致的地表锚点，每个 XY 格一个。
std::vector<std::int64_t> stable_shore(const Gaussians& g, const std::vector<std::uint8_t>& candidate, double cell)
{
    std::vector<std::int64_t> ids;
    for (std::size_t i = 0; i < g.count; ++i)
    {
        if (candidate[i] && g.opacity[i] >= 0.5f && g.normal_z[i] >= 0.7f)
        {
            ids.push_back(static_cast<std::int64_t>(i));
        }
    }
    std::vector<std::pair<std::pair<std::int64_t, std::int64_t>, std::int64_t>> keyed;
    keyed.reserve(ids.size());
    for (std::int64_t id : ids)
    {
        const double* p = g.point(static_cast<std::size_t>(id));
        keyed.push_back({{static_cast<std::int64_t>(std::floor(p[0] / cell)), static_cast<std::int64_t>(std::floor(p[1] / cell))}, id});
    }
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::int64_t> chosen;
    for (std::size_t a = 0; a < keyed.size();)
    {
        std::size_t b = a;
        while (b < keyed.size() && keyed[b].first == keyed[a].first)
        {
            ++b;
        }
        if (b - a >= 3)
        {
            std::vector<double> zz;
            for (std::size_t k = a; k < b; ++k)
            {
                zz.push_back(g.point(static_cast<std::size_t>(keyed[k].second))[2]);
            }
            const double low = quantile(zz, .25);
            const double tolerance = std::max(.10, cell * .25);
            std::vector<std::size_t> near;
            for (std::size_t k = 0; k < zz.size(); ++k)
            {
                if (std::abs(zz[k] - low) <= tolerance)
                {
                    near.push_back(k);
                }
            }
            if (near.size() >= 3)
            {
                std::vector<double> nz;
                for (std::size_t k : near)
                {
                    nz.push_back(zz[k]);
                }
                const double med = median(nz);
                std::size_t best = near[0];
                for (std::size_t k : near)
                {
                    if (std::abs(zz[k] - med) < std::abs(zz[best] - med))
                    {
                        best = k;
                    }
                }
                chosen.push_back(keyed[a + best].second);
            }
        }
        a = b;
    }
    return chosen;
}

std::vector<std::int64_t> flat(const std::vector<std::uint8_t>& mask)
{
    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < mask.size(); ++i)
    {
        if (mask[i])
        {
            out.push_back(static_cast<std::int64_t>(i));
        }
    }
    return out;
}

std::vector<double> gather_xyz(const Gaussians& g, const std::vector<std::int64_t>& rows)
{
    std::vector<double> out(3 * rows.size());
    for (std::size_t k = 0; k < rows.size(); ++k)
    {
        for (int a = 0; a < 3; ++a)
        {
            out[3 * k + a] = g.xyz[3 * static_cast<std::size_t>(rows[k]) + a];
        }
    }
    return out;
}

Json box_json(const Box3& box)
{
    Json v;
    v["minimum"] = box.minimum;
    v["maximum"] = box.maximum;
    return v;
}

} // namespace

void run_detect(const Config& config, const std::string& name)
{
    const Timer total;
    const std::filesystem::path out = config.work / "detect" / utf8_path(name);
    std::filesystem::create_directories(out);
    const LodIndex lod = read_lod_index(config.lod);
    const LodBlock* block = lod.block(name);
    if (!block)
    {
        fail_value("Block is not present in the LOD: " + name);
    }
    std::vector<View> views = block_views(config.project, name, config.mask_dir);
    require_pinhole(views);
    MaskCache cache(config.water_class, std::size_t{1} << 30);
    const VotePolicy policy;

    // 1) 每张图解码一次：占据格网（ROI 预检 + apply 反票）。
    Timer timer;
    std::vector<Occupancy> occupancy(views.size());
    parallel_for_limited(views.size(), 8, [&](std::size_t i) { occupancy[i] = cache.occupancy(views[i]); });
    const double decode_seconds = timer.seconds();

    timer = Timer();
    Gaussians g = read_block_gaussians(lod, *block, nullptr);
    const double read_seconds = timer.seconds();
    log("[DETECT] %s: %zu finest Gaussians in %zu leaves (L%d), %zu views, masks %.1fs, LOD %.1fs", name.c_str(),
        g.count, block->leaves.size(), block->level, views.size(), decode_seconds, read_seconds);

    // 2) ROI 预检：有水且 ROI 碰到水格的视图需要像素级投票，其余只记反票。
    std::vector<std::uint8_t> selected(views.size(), 0);
    std::size_t wet_views = 0, fallbacks = 0;
    {
        const Envelope envelope(g);
        parallel_for_each(views.size(), [&](std::size_t i) {
            if (!occupancy[i].water_pixels)
            {
                return;
            }
            std::vector<std::array<double, 4>> rects;
            if (!envelope.rectangles(views[i], rects))
            {
                selected[i] = 1;
                return;
            }
            selected[i] = roi_hits(occupancy[i], rects) ? 1 : 0;
        });
        for (std::size_t i = 0; i < views.size(); ++i)
        {
            wet_views += occupancy[i].water_pixels ? 1 : 0;
        }
    }
    const std::size_t selected_count = static_cast<std::size_t>(std::count(selected.begin(), selected.end(), 1));
    log("[DETECT] %s: ROI precheck %zu/%zu views selected (%zu with water)", name.c_str(), selected_count, views.size(),
        wet_views);
    (void)fallbacks;

    Json stats;
    stats["block"] = name;
    stats["count"] = g.count;
    stats["views"] = views.size();
    stats["selected_views"] = selected_count;
    stats["mask_decode_seconds"] = decode_seconds;
    stats["lod_read_seconds"] = read_seconds;

    const std::size_t n = g.count;
    std::vector<std::uint8_t> water(n, 0), land(n, 0), seed(n, 0), shore_candidate(n, 0);
    std::vector<double> fraction(n, std::numeric_limits<double>::quiet_NaN()), geometry(n, 0.0);
    std::vector<std::int64_t> shore;
    DetectGpu::Result votes;
    const bool no_water = selected_count == 0;
    if (!no_water)
    {
        timer = Timer();
        gpu_select_device(config.gpu);
        Vec3 low{1e300, 1e300, 1e300}, high{-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < n; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                low[k] = std::min(low[k], g.xyz[3 * i + k]);
                high[k] = std::max(high[k], g.xyz[3 * i + k]);
            }
        }
        const double diagonal = std::sqrt(std::pow(high[0] - low[0], 2) + std::pow(high[1] - low[1], 2) + std::pow(high[2] - low[2], 2));
        const double unit = std::max(diagonal / 1000, 1e-6);
        DetectGpu gpu(g, unit);
        std::size_t projected = 0, negative_views = 0, sky_pixels = 0;
        const std::size_t batch = 6;
        for (std::size_t start = 0; start < views.size(); start += batch)
        {
            const std::size_t end = std::min(views.size(), start + batch);
            std::vector<PreparedView> prepared(end - start);
            std::vector<std::uint8_t> sees(end - start, 0);
            parallel_for_each(end - start, [&](std::size_t j) {
                const std::size_t i = start + j;
                sees[j] = view_sees_box(views[i], low, high, 0.0) ? 1 : 0;
                if (sees[j] && selected[i])
                {
                    prepared[j] = prepare_view(views[i], cache);
                }
            });
            for (std::size_t j = 0; j < end - start; ++j)
            {
                if (!sees[j])
                {
                    continue;
                }
                const std::size_t i = start + j;
                const PinholeCamera cam = pinhole(views[i]);
                PreparedView& p = prepared[j];
                sky_pixels += p.rejected;
                const bool negative = !selected[i] || p.dry;
                const std::size_t inside = gpu.vote(cam, negative, negative ? nullptr : p.state.data(),
                                                    negative ? nullptr : p.distance.data(), negative ? nullptr : p.shore.data());
                if (inside)
                {
                    gpu.render(cam, negative ? nullptr : p.state.data());
                    ++projected;
                    negative_views += negative ? 1 : 0;
                }
            }
            if (start / batch % 25 == 0)
            {
                log("[VOTE] %s: %zu/%zu views, %.1fs", name.c_str(), end, views.size(), timer.seconds());
            }
        }
        votes = gpu.download();
        stats["projected_views"] = projected;
        stats["negative_only_views"] = negative_views;
        stats["sky_water_pixels"] = sky_pixels;
        stats["boundary_checks"] = votes.boundary_checks;
        stats["early_uniform"] = votes.early_uniform;
        stats["subdivided_tiles"] = votes.subdivided_tiles;
        stats["vote_seconds"] = timer.seconds();

        std::size_t geometry_count = 0, observed_count = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const int visible = votes.visible[i], hits = votes.water[i], land_hits = votes.land[i];
            const bool enough = visible >= policy.min_visible_views;
            const bool geometry_water = enough && hits >= policy.min_water_views && hits >= policy.water_hit_ratio * visible;
            seed[i] = geometry_water && hits >= policy.seed_hit_ratio * visible;
            const double mw = votes.mass[3 * i], ml = votes.mass[3 * i + 1];
            const double known = mw + ml;
            const bool observed = enough && votes.water_views[i] >= policy.min_water_views && known > 0 &&
                                  mw >= policy.support_water_fraction * known;
            water[i] = geometry_water || observed;
            land[i] = enough && land_hits >= policy.min_land_views && land_hits >= policy.land_hit_ratio * visible && !water[i];
            shore_candidate[i] = votes.center_visible[i] >= policy.min_shore_visible &&
                                 votes.boundary[i] >= policy.min_shore_views &&
                                 votes.center_water[i] < policy.water_hit_ratio * votes.center_visible[i] && !water[i];
            if (known > 0)
            {
                fraction[i] = mw / known;
            }
            geometry[i] = static_cast<double>(hits) / std::max(visible, 1);
            geometry_count += geometry_water ? 1 : 0;
            observed_count += (observed && !geometry_water) ? 1 : 0;
        }
        shore = stable_shore(g, shore_candidate, config.cell_size);
        std::sort(shore.begin(), shore.end());
        stats["geometry_water"] = geometry_count;
        stats["visibility_recovered_water"] = observed_count;
    }
    const std::vector<std::int64_t> water_rows = flat(water), seed_rows = flat(seed), land_rows_all = flat(land);
    stats["water"] = water_rows.size();
    stats["seed"] = seed_rows.size();
    stats["land"] = land_rows_all.size();
    stats["shore"] = shore.size();
    stats["shore_candidates"] = flat(shore_candidate).size();
    log("[DETECT] %s: water %zu (seed %zu), land %zu, shore anchors %zu", name.c_str(), water_rows.size(),
        seed_rows.size(), land_rows_all.size(), shore.size());

    // 3) 检测结果只用于诊断：仅 --debug 时保存。
    if (config.debug)
    {
        NpzWriter writer;
        npz_put(writer, "water_indices", water_rows);
        npz_put(writer, "shore_indices", shore);
        if (!no_water)
        {
            npz_put(writer, "visible_views", votes.visible);
            npz_put(writer, "water_views", votes.water);
            npz_put(writer, "land_views", votes.land);
            npz_put(writer, "center_views", votes.center_visible);
            npz_put(writer, "center_water_views", votes.center_water);
            npz_put(writer, "boundary_views", votes.boundary);
            npz_put(writer, "observed_water_views", votes.water_views);
            npz_put(writer, "visible_mass", votes.mass, {n, 3});
            npz_put(writer, "water_seed", seed);
            npz_put(writer, "shore_candidate", shore_candidate);
            npz_put(writer, "land", land);
        }
        writer.write(out / "detections.npz", true);
    }

    // 4) prepare_components：岸线锚点在每张图里的外环分量标签。
    timer = Timer();
    std::vector<std::size_t> order(views.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return views[a].mask_key < views[b].mask_key; });
    std::vector<std::vector<std::int32_t>> obs_label(views.size());
    std::vector<std::vector<std::int64_t>> obs_anchor(views.size());
    const std::vector<double> shore_xyz = gather_xyz(g, shore);
    if (!no_water && !shore.empty())
    {
        parallel_for_limited(order.size(), 8, [&](std::size_t r) {
            const View& view = views[order[r]];
            std::vector<int> u, v;
            std::vector<std::int64_t> anchors;
            for (std::size_t a = 0; a < shore.size(); ++a)
            {
                double pu = 0, pv = 0;
                if (!project_point(view, shore_xyz.data() + 3 * a, pu, pv))
                {
                    continue;
                }
                if (!(pu >= 0 && pv >= 0 && pu < view.width && pv < view.height))
                {
                    continue;
                }
                anchors.push_back(static_cast<std::int64_t>(a));
                u.push_back(static_cast<int>(std::floor(pu)));
                v.push_back(static_cast<int>(std::floor(pv)));
            }
            if (anchors.empty())
            {
                return;
            }
            const Labels labels = cache.labels(view);
            const std::vector<int> found = boundary_at_pixels(labels, boundary_pixels(view.width, view.height), u, v);
            for (std::size_t k = 0; k < anchors.size(); ++k)
            {
                if (found[k] > 0)
                {
                    obs_label[r].push_back(found[k]);
                    obs_anchor[r].push_back(anchors[k]);
                }
            }
        });
    }
    std::vector<std::int32_t> all_view, all_label;
    std::vector<std::int64_t> all_anchor;
    for (std::size_t r = 0; r < order.size(); ++r)
    {
        for (std::size_t k = 0; k < obs_label[r].size(); ++k)
        {
            all_view.push_back(static_cast<std::int32_t>(r));
            all_label.push_back(obs_label[r][k]);
            all_anchor.push_back(obs_anchor[r][k]);
        }
    }
    stats["shore_observations"] = all_view.size();
    stats["prepare_seconds"] = timer.seconds();
    {
        NpzWriter writer;
        npz_put(writer, "shore_indices", shore);
        npz_put(writer, "shore_xyz", shore_xyz, {shore.size(), 3});
        npz_put(writer, "obs_view", all_view);
        npz_put(writer, "obs_label", all_label);
        npz_put(writer, "obs_anchor", all_anchor);
        writer.write(out / "components.npz", true);
    }
    // 拟合种子（水参照、原水支持与颜色）；删除候选为全部水判定。
    {
        NpzWriter writer;
        npz_put(writer, "water_xyz", gather_xyz(g, seed_rows), {seed_rows.size(), 3});
        std::vector<float> rgb(3 * seed_rows.size());
        for (std::size_t k = 0; k < seed_rows.size(); ++k)
        {
            for (int a = 0; a < 3; ++a)
            {
                rgb[3 * k + a] = std::min(1.0f, (g.rgba[4 * static_cast<std::size_t>(seed_rows[k]) + a] + 0.5f) / 255.0f);
            }
        }
        npz_put(writer, "water_rgb", rgb, {seed_rows.size(), 3});
        writer.write(out / "seeds.npz", true);
    }
    if (!no_water)
    {
        NpzWriter writer;
        npz_put(writer, "deletion_indices", water_rows);
        npz_put(writer, "deletion_xyz", gather_xyz(g, water_rows), {water_rows.size(), 3});
        writer.write(out / "deletion_evidence.npz", true);
    }
    // 陆地与可见水份额、可靠陆地格网。
    const double land_grid_spacing = config.footprint_spacing;
    if (!no_water)
    {
        std::vector<std::int64_t> land_rows, visible_rows;
        std::vector<std::uint8_t> visible_fraction, visible_geometry, visible_u8(n, 255);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (land[i] && !seed[i])
            {
                land_rows.push_back(static_cast<std::int64_t>(i));
            }
            const bool finite = std::isfinite(fraction[i]);
            if (finite)
            {
                visible_u8[i] = static_cast<std::uint8_t>(std::nearbyint(std::clamp(fraction[i], 0.0, 1.0) * 254));
            }
            if (finite && fraction[i] >= .3 && !seed[i])
            {
                visible_rows.push_back(static_cast<std::int64_t>(i));
                visible_fraction.push_back(static_cast<std::uint8_t>(std::nearbyint(std::clamp(fraction[i], 0.0, 1.0) * 255)));
                visible_geometry.push_back(static_cast<std::uint8_t>(std::nearbyint(std::clamp(geometry[i], 0.0, 1.0) * 255)));
            }
        }
        // 可靠陆地：投成陆地、可见、主要渲染在非水像素上、不透明；每个绝对格取最低/最高面与点数。
        std::vector<std::pair<std::pair<std::int64_t, std::int64_t>, double>> cells;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (land[i] && std::isfinite(fraction[i]) && fraction[i] < .5 && g.opacity[i] >= .5f)
            {
                const double* p = g.point(i);
                cells.push_back({{static_cast<std::int64_t>(std::nearbyint(p[0] / land_grid_spacing)),
                                  static_cast<std::int64_t>(std::nearbyint(p[1] / land_grid_spacing))},
                                 p[2]});
            }
        }
        std::sort(cells.begin(), cells.end());
        std::vector<std::int64_t> land_cells;
        std::vector<float> zmin, zmax;
        std::vector<std::int32_t> counts;
        for (std::size_t a = 0; a < cells.size();)
        {
            std::size_t b = a;
            while (b < cells.size() && cells[b].first == cells[a].first)
            {
                ++b;
            }
            land_cells.push_back(cells[a].first.first);
            land_cells.push_back(cells[a].first.second);
            zmin.push_back(static_cast<float>(cells[a].second));
            zmax.push_back(static_cast<float>(cells[b - 1].second));
            counts.push_back(static_cast<std::int32_t>(b - a));
            a = b;
        }
        // 逐点清理证据：只给本块 write。
        NpzWriter carve;
        npz_put(carve, "land_indices", land_rows);
        npz_put(carve, "visible_water_indices", visible_rows);
        npz_put(carve, "visible_water_fraction", visible_fraction);
        npz_put(carve, "visible_water_geometry", visible_geometry);
        npz_put(carve, "visible_water_u8", visible_u8);
        carve.write(out / "carve_evidence.npz", true);
        // 可靠陆地格网：给本块与邻块的 candidates（屏障、遮挡）。
        NpzWriter land_grid;
        npz_put(land_grid, "land_cells", land_cells, {land_cells.size() / 2, 2});
        npz_put(land_grid, "land_zmin", zmin);
        npz_put(land_grid, "land_zmax", zmax);
        npz_put(land_grid, "land_count", counts);
        land_grid.write(out / "land_grid.npz", true);
        // 占据格网（apply 的反票，不再解码 PNG）。
        NpzWriter occ;
        std::vector<std::uint8_t> blob;
        std::vector<std::int64_t> offsets, bytes, water_pixels;
        std::vector<std::int32_t> rows, cols;
        for (std::size_t r = 0; r < order.size(); ++r)
        {
            const Occupancy& o = occupancy[order[r]];
            offsets.push_back(static_cast<std::int64_t>(blob.size()));
            bytes.push_back(static_cast<std::int64_t>(o.packed.size()));
            blob.insert(blob.end(), o.packed.begin(), o.packed.end());
            rows.push_back(o.rows);
            cols.push_back(o.cols);
            water_pixels.push_back(o.water_pixels);
        }
        npz_put(occ, "occupancy_blob", blob);
        npz_put(occ, "occupancy_offsets", offsets);
        npz_put(occ, "occupancy_bytes", bytes);
        npz_put(occ, "occupancy_rows", rows);
        npz_put(occ, "occupancy_cols", cols);
        npz_put(occ, "occupancy_water_pixels", water_pixels);
        occ.write(out / "occupancy.npz", true);
    }

    // 交接元数据：视图只存掩码键（obs_view 与占据格网按此顺序），相机由 reconcile 统一建目录。
    Json metadata;
    metadata["schema"] = "ai3d-waterpost.prepared/2";
    metadata["block"] = name;
    metadata["count"] = n;
    metadata["water"] = seed_rows.size();
    metadata["deletion_candidates"] = water_rows.size();
    metadata["shore"] = shore.size();
    metadata["bounds"] = box_json(block_core_bounds(config.project, name));
    metadata["no_water"] = no_water;
    Json keys = Json::array();
    if (!no_water)
    {
        for (std::size_t r = 0; r < order.size(); ++r)
        {
            keys.push_back(views[order[r]].mask_key);
        }
    }
    metadata["view_keys"] = keys;
    stats["mask_decodes"] = cache.decodes();
    stats["seconds"] = total.seconds();
    metadata["stats"] = stats;
    write_json_atomic(out / "components.json", metadata, false);
    log("[DETECT] %s complete in %.1fs", name.c_str(), total.seconds());
}

} // namespace ai3d::waterpost
