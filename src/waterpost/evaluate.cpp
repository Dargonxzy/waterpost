// evaluate：把若干个 LOD 的最细层渲染到训练相机下，与原始照片（以及第一个 LOD 的渲染）算 PSNR，
// 按水 / 非水像素（语义掩码）分开统计。只用 bin 里的 0 阶颜色，黑背景。

#include "error.hpp"
#include "fs_util.hpp"
#include "gpu.hpp"
#include "lod.hpp"
#include "masks.hpp"
#include "parallel.hpp"
#include "project.hpp"
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

void run_evaluate(const std::filesystem::path& project, const std::vector<std::pair<std::string, std::filesystem::path>>& variants,
                  const std::filesystem::path& mask_dir, const std::filesystem::path& images, int step,
                  const std::filesystem::path& csv, int gpu, int water_class, const std::vector<std::string>& only)
{
    gpu_select_device(gpu);
    std::map<std::string, View> unique;
    for (const std::string& name : project_blocks(project))
    {
        for (View& view : block_views(project, name, mask_dir))
        {
            unique.emplace(view.key(), std::move(view));
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
    std::vector<std::array<Sum, 5>> totals(v);
    std::vector<std::array<double, 5>> mean_psnr(v, std::array<double, 5>{});
    std::vector<std::array<std::size_t, 5>> mean_count(v, std::array<std::size_t, 5>{});
    std::ofstream out(csv);
    out << "view,water_fraction";
    for (const auto& [name, path] : variants)
    {
        out << "," << name << "_all," << name << "_water," << name << "_land," << name << "_vs_ref_all," << name << "_vs_ref_land";
    }
    out << "\n";
    const Timer timer;
    for (std::size_t k = 0; k < views.size(); ++k)
    {
        const View& view = views[k];
        const std::filesystem::path image_path = images / utf8_path(view.name);
        cv::Mat photo;
        {
            std::ifstream in(image_path, std::ios::binary);
            if (in) {
                std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                if (!bytes.empty()) photo = cv::imdecode(bytes, cv::IMREAD_COLOR);
            }
        }
        if (photo.empty())
        {
            log("[EVAL] missing image %s", path_utf8(image_path).c_str());
            continue;
        }
        if (photo.cols != view.width || photo.rows != view.height)
        {
            cv::resize(photo, photo, cv::Size(view.width, view.height), 0, 0, cv::INTER_AREA);
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
        for (std::size_t r = 0; r < v; ++r)
        {
            std::array<Sum, 5> local;
            for (int y = 0; y < view.height; ++y)
            {
                const cv::Vec3b* row = photo.ptr<cv::Vec3b>(y);
                const std::uint8_t* w = water.ptr<std::uint8_t>(y);
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
                }
            }
            for (int s = 0; s < 5; ++s)
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
        if (k % 20 == 0)
        {
            log("[EVAL] %zu/%zu views, %.1fs", k + 1, views.size(), timer.seconds());
        }
    }
    log("[EVAL] mean of per-image PSNR (dB) over %zu views; vs_ref = against the first LOD's render", views.size());
    log("%-14s %8s %8s %8s %12s %12s", "lod", "all", "water", "land", "vs_ref_all", "vs_ref_land");
    for (std::size_t r = 0; r < v; ++r)
    {
        double m[5];
        for (int s = 0; s < 5; ++s)
        {
            m[s] = mean_count[r][static_cast<std::size_t>(s)] ? mean_psnr[r][static_cast<std::size_t>(s)] / mean_count[r][static_cast<std::size_t>(s)] : std::nan("");
        }
        log("%-14s %8.3f %8.3f %8.3f %12.3f %12.3f", variants[r].first.c_str(), m[0], m[1], m[2], m[3], m[4]);
    }
}

} // namespace ai3d::waterpost
