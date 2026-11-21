// evaluate：把若干个 LOD 的最细层渲染到训练相机下，与原始照片（以及第一个 LOD 的渲染）算 PSNR，
// 按水 / 非水像素（语义掩码）分开统计。只用 bin 里的 0 阶颜色，黑背景。

#include "error.hpp"
#include "fs_util.hpp"
#include "gpu.hpp"
#include "lod.hpp"
#include "masks.hpp"
#include "parallel.hpp"
#include "photos.hpp"
#include "project.hpp"
#include "scene.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

namespace ai3d::waterpost
{
namespace
{

Gaussians all_finest(const std::filesystem::path& root)
{
    const LodIndex lod = read_lod_index(root);
    Gaussians out;
    for (const LodBlock& block : lod.blocks)
    {
        Gaussians g = read_block_gaussians(lod, block, nullptr);
        out.count += g.count;
        out.xyz.insert(out.xyz.end(), g.xyz.begin(), g.xyz.end());
        out.cov.insert(out.cov.end(), g.cov.begin(), g.cov.end());
        out.opacity.insert(out.opacity.end(), g.opacity.begin(), g.opacity.end());
        out.rgba.insert(out.rgba.end(), g.rgba.begin(), g.rgba.end());
    }
    return out;
}

struct Sum
{
    double se = 0;
    double count = 0;
    void add(double e)
    {
        se += e;
        count += 1;
    }
    double psnr() const
    {
        return count > 0 ? 10 * std::log10(1.0 / std::max(se / count, 1e-12)) : std::nan("");
    }
};

} // namespace


// 照片路径：图像名是存在的绝对路径就直接用；否则在照片目录下找 <图像名去掉前 k 级目录>（k 第一次找到后固定）。
std::filesystem::path photo_path(const std::filesystem::path& images, const std::string& name, long long& cut)
{
    std::vector<std::string> parts;
    std::string current;
    for (char c : name)
    {
        if (c == '/' || c == '\\')
        {
            if (!current.empty()) parts.push_back(current);
            current.clear();
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty()) parts.push_back(current);
    auto under = [&](std::size_t skip) {
        std::filesystem::path path = images;
        for (std::size_t k = skip; k < parts.size(); ++k) path /= utf8_path(parts[k]);
        return path;
    };
    if (cut >= 0)
    {
        return under(std::min<std::size_t>(static_cast<std::size_t>(cut), parts.size() - 1));
    }
    for (std::size_t skip = 0; skip + 1 <= parts.size(); ++skip)
    {
        if (std::filesystem::is_regular_file(under(skip)))
        {
            cut = static_cast<long long>(skip);
            return under(skip);
        }
    }
    const std::filesystem::path direct = utf8_path(name);
    return direct.is_absolute() ? direct : images / direct;
}

cv::Mat load_view_photo(const std::filesystem::path& images, const View& view, long long& cut)
{
    const std::filesystem::path image_path = photo_path(images, view.name, cut);
    cv::Mat photo;
    std::ifstream in(image_path, std::ios::binary);
    if (in)
    {
        std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!bytes.empty())
        {
            photo = cv::imdecode(bytes, cv::IMREAD_COLOR);
        }
    }
    if (photo.empty())
    {
        return photo;
    }
    if (photo.cols != view.width || photo.rows != view.height)
    {
        cv::resize(photo, photo, cv::Size(view.width, view.height), 0, 0, cv::INTER_AREA);
    }
    if (view.distorted())
    {
        photo = undistort_bilinear(view, photo); // 渲染是针孔相机，照片先去畸变才能逐像素比
    }
    return photo;
}

void run_evaluate(const std::filesystem::path& project, const std::vector<std::pair<std::string, std::filesystem::path>>& variants,
                  const std::filesystem::path& mask_dir, const std::filesystem::path& images, int step,
                  const std::filesystem::path& csv, int gpu, int water_class, const std::vector<std::string>& only,
                  const Config* poses, const std::vector<EvalRegion>& regions, const std::filesystem::path& region_csv,
                  int region_pad)
{
    gpu_select_device(gpu);
    std::map<std::string, View> unique;
    if (poses)
    {
        for (View& view : read_pose_views(*poses))
        {
            unique.emplace(view.key(), std::move(view));
        }
    }
    else
    {
        for (const std::string& name : project_blocks(project))
        {
            for (View& view : block_views(project, name, mask_dir))
            {
                unique.emplace(view.key(), std::move(view));
            }
        }
    }
    std::vector<View> views;
    std::size_t index = 0;
    for (auto& [key, view] : unique)
    {
        const bool wanted = only.empty() ? index % static_cast<std::size_t>(std::max(1, step)) == 0
                                         : std::find(only.begin(), only.end(), view.name) != only.end();
        ++index;
        if (wanted)
        {
            views.push_back(view);
        }
    }
    require_pinhole(views);
    std::vector<std::unique_ptr<ColorRenderer>> renderers;
    for (const auto& [name, path] : variants)
    {
        const Gaussians g = all_finest(path);
        Vec3 low{1e300, 1e300, 1e300}, high{-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < g.count; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                low[k] = std::min(low[k], g.xyz[3 * i + k]);
                high[k] = std::max(high[k], g.xyz[3 * i + k]);
            }
        }
        const double unit = std::max(std::sqrt(std::pow(high[0] - low[0], 2) + std::pow(high[1] - low[1], 2) + std::pow(high[2] - low[2], 2)) / 1000, 1e-6);
        renderers.push_back(std::make_unique<ColorRenderer>(g, unit));
        log("[EVAL] %s: %zu finest Gaussians", name.c_str(), g.count);
    }
    MaskCache masks(water_class, std::size_t{256} << 20);
    const std::size_t v = variants.size();
    // [variant][0 all, 1 water, 2 land] vs photo; [variant][3 all, 4 land] vs the first variant's render
    // 另加 [5] 岸线带全部、[6] 岸线带水侧、[7] 岸线带陆侧：掩码边界 15 px 以内的像素（新水面越界 / 岸边残留只影响这一带）。
    constexpr int METRICS = 8;
    std::vector<std::array<Sum, METRICS>> totals(v);
    std::vector<std::array<double, METRICS>> mean_psnr(v, std::array<double, METRICS>{});
    std::vector<std::array<std::size_t, METRICS>> mean_count(v, std::array<std::size_t, METRICS>{});
    std::ofstream out(csv);
    out << "view,water_fraction";
    for (const auto& [name, path] : variants)
    {
        out << "," << name << "_all," << name << "_water," << name << "_land," << name << "_vs_ref_all," << name << "_vs_ref_land,"
            << name << "_shore," << name << "_shore_water," << name << "_shore_land";
    }
    out << "\n";
    // 局部区域：[区域][变体] 的平方误差和与像素数；views 为区域落在画面内的视图数；
    // differ 为该区域内各变体渲染不完全相同的视图数（被遮挡时两者一样，不提供证据）；
    // wins[区域][变体] 为该变体比第一个变体更接近照片的视图数（只在 differ 的视图里比较）。
    const std::size_t regions_n = regions.size();
    std::vector<std::vector<double>> region_se(regions_n, std::vector<double>(v, 0.0));
    std::vector<std::vector<std::size_t>> region_wins(regions_n, std::vector<std::size_t>(v, 0));
    std::vector<std::size_t> region_pixels(regions_n, 0), region_views(regions_n, 0), region_differ(regions_n, 0);
    const Timer timer;
    long long photo_cut = -1; // 照片目录下要去掉图像名的前几级目录（第一次找到后固定）
    for (std::size_t k = 0; k < views.size(); ++k)
    {
        const View& view = views[k];
        const cv::Mat photo = load_view_photo(images, view, photo_cut);
        if (photo.empty())
        {
            log("[EVAL] missing image %s", path_utf8(photo_path(images, view.name, photo_cut)).c_str());
            continue;
        }
        const cv::Mat water = masks.water(view);
        const PinholeCamera cam = pinhole(view);
        std::vector<std::vector<float>> renders(v);
        for (std::size_t r = 0; r < v; ++r)
        {
            renders[r] = renderers[r]->render(cam);
        }
        out << view.name << "," << static_cast<double>(cv::countNonZero(water)) / (static_cast<double>(view.width) * view.height);
        if (!only.empty())
        {
            const int w = view.width / 4, h = view.height / 4;
            std::vector<cv::Mat> tiles;
            cv::Mat edge;
            cv::Mat small_mask;
            cv::resize(water, small_mask, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
            cv::morphologyEx(small_mask, edge, cv::MORPH_GRADIENT, cv::Mat::ones(3, 3, CV_8U));
            auto mark = [&](cv::Mat m, const std::string& label) {
                m.setTo(cv::Scalar(0, 0, 255), edge);
                cv::putText(m, label, cv::Point(12, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 2);
                tiles.push_back(m);
            };
            cv::Mat p;
            cv::resize(photo, p, cv::Size(w, h), 0, 0, cv::INTER_AREA);
            mark(p, "photo");
            for (std::size_t r = 0; r < v; ++r)
            {
                cv::Mat full(view.height, view.width, CV_8UC3);
                for (int y = 0; y < view.height; ++y)
                {
                    cv::Vec3b* row = full.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < view.width; ++x)
                    {
                        const std::size_t at = (static_cast<std::size_t>(y) * view.width + x) * 3;
                        for (int c = 0; c < 3; ++c)
                        {
                            row[x][2 - c] = cv::saturate_cast<uchar>(renders[r][at + c] * 255.0f);
                        }
                    }
                }
                cv::Mat s;
                cv::resize(full, s, cv::Size(w, h), 0, 0, cv::INTER_AREA);
                mark(s, variants[r].first);
            }
            cv::Mat sheet;
            cv::hconcat(tiles, sheet);
            std::vector<uchar> buf;
            cv::imencode(".jpg", sheet, buf, {cv::IMWRITE_JPEG_QUALITY, 88});
            std::ofstream(csv.parent_path() / utf8_path(std::filesystem::path(view.name).stem().string() + "_cmp.jpg"), std::ios::binary)
                .write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        }
        cv::Mat shore;
        {
            cv::Mat binary = water != 0, grown, shrunk;
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
            cv::dilate(binary, grown, kernel);
            cv::erode(binary, shrunk, kernel);
            shore = grown & ~shrunk;
        }
        for (std::size_t r = 0; r < v; ++r)
        {
            std::array<Sum, METRICS> local;
            for (int y = 0; y < view.height; ++y)
            {
                const cv::Vec3b* row = photo.ptr<cv::Vec3b>(y);
                const std::uint8_t* w = water.ptr<std::uint8_t>(y);
                const std::uint8_t* band = shore.ptr<std::uint8_t>(y);
                for (int x = 0; x < view.width; ++x)
                {
                    const std::size_t at = (static_cast<std::size_t>(y) * view.width + x) * 3;
                    double e = 0, f = 0;
                    for (int c = 0; c < 3; ++c)
                    {
                        const double d = renders[r][at + c] - row[x][2 - c] / 255.0;
                        e += d * d;
                        const double g = renders[r][at + c] - renders[0][at + c];
                        f += g * g;
                    }
                    e /= 3;
                    f /= 3;
                    local[0].add(e);
                    local[w[x] ? 1 : 2].add(e);
                    local[3].add(f);
                    if (!w[x])
                    {
                        local[4].add(f);
                    }
                    if (band[x])
                    {
                        local[5].add(e);
                        local[w[x] ? 6 : 7].add(e);
                    }
                }
            }
            for (int s = 0; s < METRICS; ++s)
            {
                totals[r][static_cast<std::size_t>(s)].se += local[static_cast<std::size_t>(s)].se;
                totals[r][static_cast<std::size_t>(s)].count += local[static_cast<std::size_t>(s)].count;
                const double p = local[static_cast<std::size_t>(s)].psnr();
                if (std::isfinite(p))
                {
                    mean_psnr[r][static_cast<std::size_t>(s)] += p;
                    ++mean_count[r][static_cast<std::size_t>(s)];
                }
                out << "," << p;
            }
        }
        out << "\n";
        for (std::size_t q = 0; q < regions_n; ++q)
        {
            const EvalRegion& region = regions[q];
            double u0 = 1e300, v0 = 1e300, u1 = -1e300, v1 = -1e300;
            bool behind = false;
            for (int corner = 0; corner < 8 && !behind; ++corner)
            {
                const double p[3] = {corner & 1 ? region.high[0] : region.low[0], corner & 2 ? region.high[1] : region.low[1],
                                     corner & 4 ? region.high[2] : region.low[2]};
                double c[3];
                for (int row = 0; row < 3; ++row)
                {
                    c[row] = cam.r[3 * row] * (p[0] - cam.center[0]) + cam.r[3 * row + 1] * (p[1] - cam.center[1]) +
                             cam.r[3 * row + 2] * (p[2] - cam.center[2]);
                }
                if (c[2] <= 0.5)
                {
                    behind = true;
                    break;
                }
                const double u = cam.fx * c[0] / c[2] + cam.cx, w = cam.fy * c[1] / c[2] + cam.cy;
                u0 = std::min(u0, u);
                u1 = std::max(u1, u);
                v0 = std::min(v0, w);
                v1 = std::max(v1, w);
            }
            if (behind)
            {
                continue;
            }
            const int x0 = std::max(0, static_cast<int>(std::floor(u0)) - region_pad);
            const int x1 = std::min(view.width - 1, static_cast<int>(std::ceil(u1)) + region_pad);
            const int y0 = std::max(0, static_cast<int>(std::floor(v0)) - region_pad);
            const int y1 = std::min(view.height - 1, static_cast<int>(std::ceil(v1)) + region_pad);
            if (x1 < x0 || y1 < y0 || static_cast<double>(x1 - x0 + 1) * (y1 - y0 + 1) > 0.25 * view.width * view.height)
            {
                continue;
            }
            ++region_views[q];
            std::vector<double> se(v, 0.0);
            bool differ = false;
            for (int y = y0; y <= y1; ++y)
            {
                const cv::Vec3b* row = photo.ptr<cv::Vec3b>(y);
                for (int x = x0; x <= x1; ++x)
                {
                    const std::size_t at = (static_cast<std::size_t>(y) * view.width + x) * 3;
                    for (std::size_t r = 0; r < v; ++r)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            const double d = renders[r][at + c] - row[x][2 - c] / 255.0;
                            se[r] += d * d / 3;
                            differ |= std::abs(renders[r][at + c] - renders[0][at + c]) > 1e-4f;
                        }
                    }
                }
            }
            region_pixels[q] += static_cast<std::size_t>(x1 - x0 + 1) * static_cast<std::size_t>(y1 - y0 + 1);
            for (std::size_t r = 0; r < v; ++r)
            {
                region_se[q][r] += se[r];
            }
            if (differ)
            {
                ++region_differ[q];
                for (std::size_t r = 1; r < v; ++r)
                {
                    region_wins[q][r] += se[r] < se[0] ? 1 : 0;
                }
            }
        }
        if (k % 20 == 0)
        {
            log("[EVAL] %zu/%zu views, %.1fs", k + 1, views.size(), timer.seconds());
        }
    }
    if (regions_n)
    {
        std::ofstream rc(region_csv.empty() ? std::filesystem::path(csv).replace_extension(".regions.csv") : region_csv);
        rc << "block,id,decision,points,views,differ_views,pixels";
        for (const auto& [name, path] : variants)
        {
            rc << "," << name << "_psnr";
        }
        for (std::size_t r = 1; r < v; ++r)
        {
            rc << "," << variants[r].first << "_wins";
        }
        rc << "\n";
        for (std::size_t q = 0; q < regions_n; ++q)
        {
            rc << regions[q].block << "," << regions[q].id << "," << regions[q].decision << "," << regions[q].points << ","
               << region_views[q] << "," << region_differ[q] << "," << region_pixels[q];
            for (std::size_t r = 0; r < v; ++r)
            {
                const double mse = region_pixels[q] ? region_se[q][r] / static_cast<double>(region_pixels[q]) : std::nan("");
                rc << "," << 10 * std::log10(1.0 / std::max(mse, 1e-12));
            }
            for (std::size_t r = 1; r < v; ++r)
            {
                rc << "," << region_wins[q][r];
            }
            rc << "\n";
        }
        log("[EVAL] %zu regions written", regions_n);
    }
    log("[EVAL] mean of per-image PSNR (dB) over %zu views; vs_ref = against the first LOD's render", views.size());
    log("%-14s %8s %8s %8s %12s %12s %8s %12s %11s", "lod", "all", "water", "land", "vs_ref_all", "vs_ref_land", "shore",
        "shore_water", "shore_land");
    for (std::size_t r = 0; r < v; ++r)
    {
        double m[METRICS];
        for (int s = 0; s < METRICS; ++s)
        {
            m[s] = mean_count[r][static_cast<std::size_t>(s)] ? mean_psnr[r][static_cast<std::size_t>(s)] / mean_count[r][static_cast<std::size_t>(s)] : std::nan("");
        }
        log("%-14s %8.3f %8.3f %8.3f %12.3f %12.3f %8.3f %12.3f %11.3f", variants[r].first.c_str(), m[0], m[1], m[2], m[3], m[4],
            m[5], m[6], m[7]);
    }
}

} // namespace ai3d::waterpost
