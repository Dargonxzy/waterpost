// 中等漂浮簇的照片核验：训练照片是独立于重建结果的观测。对每个候选簇，在看得到它的视图里分别渲染
// "原始块"与"去掉全部候选簇的块"，只在该簇包围盒的投影矩形内与照片比较。真实结构去掉后与照片差异变大；
// 漂浮物（照片里不存在）去掉后与照片更接近。多个候选簇互不重叠时一次渲染即可同时检验。

#include "floaters.hpp"

#include "camera.hpp"
#include "fs_util.hpp"
#include "gpu.hpp"
#include "photos.hpp"
#include "project.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ai3d::waterpost
{
namespace
{

struct PixelRect
{
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    bool valid() const
    {
        return x1 >= x0 && y1 >= y0;
    }
};

PixelRect project_box(const PinholeCamera& cam, const Vec3& low, const Vec3& high, int pad)
{
    double u0 = 1e300, v0 = 1e300, u1 = -1e300, v1 = -1e300;
    for (int corner = 0; corner < 8; ++corner)
    {
        const double p[3] = {corner & 1 ? high[0] : low[0], corner & 2 ? high[1] : low[1], corner & 4 ? high[2] : low[2]};
        double c[3];
        for (int row = 0; row < 3; ++row)
        {
            c[row] = cam.r[3 * row] * (p[0] - cam.center[0]) + cam.r[3 * row + 1] * (p[1] - cam.center[1]) +
                     cam.r[3 * row + 2] * (p[2] - cam.center[2]);
        }
        if (c[2] <= 0.5)
        {
            return {};
        }
        const double u = cam.fx * c[0] / c[2] + cam.cx, v = cam.fy * c[1] / c[2] + cam.cy;
        u0 = std::min(u0, u);
        u1 = std::max(u1, u);
        v0 = std::min(v0, v);
        v1 = std::max(v1, v);
    }
    PixelRect r;
    r.x0 = std::max(0, static_cast<int>(std::floor(u0)) - pad);
    r.x1 = std::min(cam.width - 1, static_cast<int>(std::ceil(u1)) + pad);
    r.y0 = std::max(0, static_cast<int>(std::floor(v0)) - pad);
    r.y1 = std::min(cam.height - 1, static_cast<int>(std::ceil(v1)) + pad);
    // 太大的矩形（相机贴着簇）不提供局部证据。
    if (r.valid() && static_cast<double>(r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1) > 0.25 * cam.width * cam.height)
    {
        return {};
    }
    return r;
}

Gaussians subset(const Gaussians& g, const std::vector<std::uint8_t>& drop)
{
    Gaussians out;
    for (std::size_t i = 0; i < g.count; ++i)
    {
        if (drop[i])
        {
            continue;
        }
        ++out.count;
        out.xyz.insert(out.xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * i), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
        out.cov.insert(out.cov.end(), g.cov.begin() + static_cast<std::ptrdiff_t>(6 * i), g.cov.begin() + static_cast<std::ptrdiff_t>(6 * i + 6));
        out.opacity.push_back(g.opacity[i]);
        out.rgba.insert(out.rgba.end(), g.rgba.begin() + static_cast<std::ptrdiff_t>(4 * i), g.rgba.begin() + static_cast<std::ptrdiff_t>(4 * i + 4));
        if (!g.sigma_max.empty()) out.sigma_max.push_back(g.sigma_max[i]);
        if (!g.normal_z.empty()) out.normal_z.push_back(g.normal_z[i]);
        if (!g.semantic.empty()) out.semantic.insert(out.semantic.end(), {g.semantic[2 * i], g.semantic[2 * i + 1]});
    }
    return out;
}

double render_unit(const Gaussians& g)
{
    Vec3 low{1e300, 1e300, 1e300}, high{-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < g.count; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            low[k] = std::min(low[k], g.xyz[3 * i + k]);
            high[k] = std::max(high[k], g.xyz[3 * i + k]);
        }
    }
    return std::max(std::sqrt(std::pow(high[0] - low[0], 2) + std::pow(high[1] - low[1], 2) + std::pow(high[2] - low[2], 2)) / 1000, 1e-6);
}

} // namespace

void verify_floaters_with_photos(const Gaussians& g, const std::vector<std::int32_t>& component,
                                 std::vector<FloaterCandidate>& candidates, const std::vector<std::uint8_t>& check,
                                 const std::vector<View>& views, const FloaterPhotoParams& params, const Gaussians* context,
                                 std::size_t& rendered_views, double& seconds)
{
    const Timer timer;
    rendered_views = 0;
    std::vector<std::size_t> todo;
    for (std::size_t k = 0; k < candidates.size(); ++k)
    {
        if (check[k])
        {
            todo.push_back(k);
        }
    }
    const bool available = !todo.empty() && !views.empty() && std::filesystem::is_directory(params.photos);
    for (std::size_t k : todo)
    {
        candidates[k].photo = available ? "photo_inconclusive" : "photo_unavailable";
    }
    if (!available)
    {
        seconds = timer.seconds();
        return;
    }
    // 选视图：按视图顺序，只要还有可见的候选簇不够 max_views 个视图就选上。
    std::vector<PinholeCamera> cameras(views.size());
    std::vector<std::vector<std::pair<std::size_t, PixelRect>>> visible(views.size());
    std::vector<std::size_t> taken(candidates.size(), 0);
    std::vector<std::size_t> chosen;
    for (std::size_t v = 0; v < views.size(); ++v)
    {
        cameras[v] = pinhole(views[v]);
        bool useful = false;
        std::vector<std::pair<std::size_t, PixelRect>> here;
        for (std::size_t k : todo)
        {
            const PixelRect r = project_box(cameras[v], candidates[k].low, candidates[k].high, params.pad);
            if (r.valid())
            {
                here.emplace_back(k, r);
                useful |= taken[k] < params.max_views;
            }
        }
        if (!useful)
        {
            continue;
        }
        for (const auto& [k, r] : here)
        {
            ++taken[k];
        }
        visible[v] = std::move(here);
        chosen.push_back(v);
    }
    std::vector<std::uint8_t> drop(g.count, 0);
    std::vector<std::uint8_t> checked(candidates.size(), 0);
    for (std::size_t k : todo)
    {
        checked[k] = 1;
    }
    std::vector<std::int32_t> slot_of_component;
    for (std::size_t k = 0; k < candidates.size(); ++k)
    {
        const std::size_t c = static_cast<std::size_t>(candidates[k].component);
        if (slot_of_component.size() <= c)
        {
            slot_of_component.resize(c + 1, -1);
        }
        slot_of_component[c] = static_cast<std::int32_t>(k);
    }
    for (std::size_t i = 0; i < g.count; ++i)
    {
        const std::int32_t c = component[i];
        if (c >= 0 && static_cast<std::size_t>(c) < slot_of_component.size() && slot_of_component[static_cast<std::size_t>(c)] >= 0 &&
            checked[static_cast<std::size_t>(slot_of_component[static_cast<std::size_t>(c)])])
        {
            drop[i] = 1;
        }
    }
    gpu_select_device(params.gpu);
    const double unit = render_unit(g);
    Gaussians with = g;
    Gaussians without = subset(g, drop);
    if (context && context->count)
    {
        for (Gaussians* target : {&with, &without})
        {
            target->count += context->count;
            target->xyz.insert(target->xyz.end(), context->xyz.begin(), context->xyz.end());
            target->cov.insert(target->cov.end(), context->cov.begin(), context->cov.end());
            target->opacity.insert(target->opacity.end(), context->opacity.begin(), context->opacity.end());
            target->rgba.insert(target->rgba.end(), context->rgba.begin(), context->rgba.end());
        }
    }
    ColorRenderer keep(with, unit);
    with = Gaussians();
    ColorRenderer gone(without, unit);
    without = Gaussians();
    std::vector<double> se_keep(candidates.size(), 0.0), se_drop(candidates.size(), 0.0);
    std::vector<std::size_t> pixels(candidates.size(), 0);
    long long cut = -1;
    for (std::size_t v : chosen)
    {
        const View& view = views[v];
        const cv::Mat photo = load_view_photo(params.photos, view, cut);
        if (photo.empty())
        {
            continue;
        }
        ++rendered_views;
        const std::vector<float> a = keep.render(cameras[v]);
        const std::vector<float> b = gone.render(cameras[v]);
        for (const auto& [k, r] : visible[v])
        {
            FloaterCandidate& f = candidates[k];
            // 只比较两次渲染确实不同的像素（簇的实际覆盖）；去掉簇后背后什么都没渲染出来（黑背景：邻块不在本次渲染里、
            // 或背后没有重建）的像素没有证据，跳过——否则任何有颜色的团块都会显得比黑色更接近照片。
            double ek = 0, ed = 0;
            std::size_t used = 0;
            for (int y = r.y0; y <= r.y1; ++y)
            {
                const cv::Vec3b* row = photo.ptr<cv::Vec3b>(y);
                for (int x = r.x0; x <= r.x1; ++x)
                {
                    const std::size_t at = (static_cast<std::size_t>(y) * static_cast<std::size_t>(view.width) + static_cast<std::size_t>(x)) * 3;
                    const bool differ = std::abs(a[at] - b[at]) > 2e-3f || std::abs(a[at + 1] - b[at + 1]) > 2e-3f ||
                                        std::abs(a[at + 2] - b[at + 2]) > 2e-3f;
                    if (!differ || b[at] + b[at + 1] + b[at + 2] < 0.06f)
                    {
                        continue;
                    }
                    ++used;
                    for (int c = 0; c < 3; ++c)
                    {
                        const double p = row[x][2 - c] / 255.0;
                        ek += (a[at + c] - p) * (a[at + c] - p) / 3;
                        ed += (b[at + c] - p) * (b[at + c] - p) / 3;
                    }
                }
            }
            ++f.photo_views;
            if (used)
            {
                se_keep[k] += ek;
                se_drop[k] += ed;
                pixels[k] += used;
                ++f.photo_differ;
                f.photo_wins += ed < ek ? 1 : 0;
            }
        }
    }
    for (std::size_t k : todo)
    {
        FloaterCandidate& f = candidates[k];
        if (!pixels[k])
        {
            f.photo = "photo_unavailable";
            continue;
        }
        const double n = static_cast<double>(pixels[k]);
        f.photo_keep_psnr = 10 * std::log10(1.0 / std::max(se_keep[k] / n, 1e-12));
        f.photo_drop_psnr = 10 * std::log10(1.0 / std::max(se_drop[k] / n, 1e-12));
        const double gain = f.photo_drop_psnr - f.photo_keep_psnr;
        const double ratio = f.photo_differ ? static_cast<double>(f.photo_wins) / static_cast<double>(f.photo_differ) : 0.5;
        if (f.photo_differ < params.min_views)
        {
            f.photo = "photo_inconclusive";
        }
        else if (gain >= params.min_gain && ratio >= params.win_ratio)
        {
            f.photo = "photo_favors_drop";
        }
        else if (gain <= -params.min_gain && ratio <= 1 - params.win_ratio)
        {
            f.photo = "photo_favors_keep";
        }
        else
        {
            f.photo = "photo_inconclusive";
        }
    }
    seconds = timer.seconds();
}

void combine_floater_evidence(FloaterCandidate& f)
{
    f.geometry = f.decision;
    if (f.border || f.geometry == "review_elongated")
    {
        return;
    }
    // 删除需要"强错误证据 + 缺乏可信物体支持"：照片支持删除，且几何上与主体分离。贴近主体的簇即使照片不支持也只标记待核查。
    const bool detached = f.geometry == "remove_suspended_floater" || f.geometry == "review_detached_supported" ||
                          f.geometry == "review_unobserved_below";
    if (f.photo == "photo_favors_drop" && !detached)
    {
        f.decision = "review_near_structure_photo_disputes";
    }
    else if (f.photo == "photo_favors_drop")
    {
        f.decision = "remove_photo_confirmed_floater";
    }
    else if (f.photo == "photo_favors_keep")
    {
        f.decision = "keep_photo_supported";
    }
    else if (f.remove())
    {
        f.decision = f.photo == "photo_unavailable" || f.photo == "photo_not_checked" ? "review_suspended_no_photo"
                                                                                    : "review_suspended_photo_inconclusive";
    }
}

} // namespace ai3d::waterpost
