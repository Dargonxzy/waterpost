#include "masks.hpp"

#include "error.hpp"
#include "fs_util.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <zstd.h>

#include <algorithm>
#include <map>
#include <memory>
#include <climits>
#include <cmath>
#include <cstring>

namespace ai3d::waterpost
{
namespace
{

std::vector<std::uint8_t> compress(const void* data, std::size_t size)
{
    std::vector<std::uint8_t> out(ZSTD_compressBound(size));
    const std::size_t written = ZSTD_compress(out.data(), out.size(), data, size, 1);
    if (ZSTD_isError(written))
    {
        fail_runtime("zstd compression failed");
    }
    out.resize(written);
    out.shrink_to_fit();
    return out;
}

void decompress(const std::vector<std::uint8_t>& payload, void* data, std::size_t size)
{
    const std::size_t read = ZSTD_decompress(data, size, payload.data(), payload.size());
    if (ZSTD_isError(read) || read != size)
    {
        fail_runtime("zstd decompression failed");
    }
}

} // namespace

MaskCache::MaskCache(int water_class, std::size_t budget_bytes) : m_water_class(water_class), m_budget(budget_bytes)
{
}

MaskCache& shared_mask_cache(int water_class)
{
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<MaskCache>> caches;
    std::lock_guard<std::mutex> lock(mutex);
    std::unique_ptr<MaskCache>& cache = caches[water_class];
    if (!cache)
    {
        cache = std::make_unique<MaskCache>(water_class, std::size_t{2} << 30);
    }
    return *cache;
}

std::size_t MaskCache::decodes() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_decodes;
}

bool MaskCache::get(const std::string& key, Entry& out)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto found = m_entries.find(key);
    if (found == m_entries.end())
    {
        return false;
    }
    m_order.splice(m_order.end(), m_order, found->second.second);
    out = found->second.first;
    return true;
}

void MaskCache::put(const std::string& key, Entry entry)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (entry.payload.size() > m_budget)
    {
        return;
    }
    const auto found = m_entries.find(key);
    if (found != m_entries.end())
    {
        m_bytes -= found->second.first.payload.size();
        m_order.erase(found->second.second);
        m_entries.erase(found);
    }
    while (!m_order.empty() && m_bytes + entry.payload.size() > m_budget)
    {
        const auto victim = m_entries.find(m_order.front());
        m_bytes -= victim->second.first.payload.size();
        m_entries.erase(victim);
        m_order.pop_front();
    }
    m_bytes += entry.payload.size();
    m_order.push_back(key);
    m_entries.emplace(key, std::make_pair(std::move(entry), std::prev(m_order.end())));
}

cv::Mat MaskCache::water(const View& view)
{
    const std::string key = "w:" + view.mask_key;
    Entry entry;
    if (get(key, entry))
    {
        cv::Mat water(entry.rows, entry.cols, CV_8U);
        std::vector<std::uint8_t> packed((static_cast<std::size_t>(entry.rows) * entry.cols + 7) / 8);
        decompress(entry.payload, packed.data(), packed.size());
        std::uint8_t* out = water.ptr<std::uint8_t>();
        const std::size_t total = static_cast<std::size_t>(entry.rows) * entry.cols;
        for (std::size_t i = 0; i < total; ++i)
        {
            out[i] = (packed[i >> 3] >> (7 - (i & 7))) & 1u;
        }
        return water;
    }
    const std::vector<std::uint8_t> bytes = read_file_bytes(view.mask_path);
    cv::Mat raw = cv::imdecode(cv::Mat(1, static_cast<int>(bytes.size()), CV_8U, const_cast<std::uint8_t*>(bytes.data())),
                               cv::IMREAD_UNCHANGED);
    if (raw.empty())
    {
        fail_value("Cannot decode semantic mask: " + path_utf8(view.mask_path));
    }
    cv::Mat classes;
    if (raw.channels() == 1)
    {
        classes = raw;
    }
    else if (raw.channels() == 2)
    {
        cv::extractChannel(raw, classes, 0);
    }
    else
    {
        // 解码成多通道的 LA PNG（OpenCV 把灰度+透明度展开为 BGRA）：灰度通道就是类别。
        const bool grey_alpha = bytes.size() >= 26 && bytes[25] == 4;
        if (grey_alpha)
        {
            cv::extractChannel(raw, classes, 0);
        }
        else
        {
            cv::Mat grey;
            cv::cvtColor(raw, grey, raw.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
            classes = grey;
        }
    }
    if (classes.depth() != CV_8U)
    {
        fail_value("Semantic mask must store 8-bit class IDs: " + path_utf8(view.mask_path));
    }
    if (classes.rows != view.height || classes.cols != view.width)
    {
        cv::Mat resized;
        cv::resize(classes, resized, cv::Size(view.width, view.height), 0, 0, cv::INTER_NEAREST);
        classes = resized;
    }
    cv::Mat water = (classes == m_water_class) / 255;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_decodes;
    }
    const std::size_t total = static_cast<std::size_t>(water.rows) * water.cols;
    std::vector<std::uint8_t> packed((total + 7) / 8, 0);
    const std::uint8_t* in = water.ptr<std::uint8_t>();
    for (std::size_t i = 0; i < total; ++i)
    {
        if (in[i])
        {
            packed[i >> 3] |= static_cast<std::uint8_t>(0x80u >> (i & 7));
        }
    }
    entry.rows = water.rows;
    entry.cols = water.cols;
    entry.payload = compress(packed.data(), packed.size());
    put(key, std::move(entry));
    return water;
}

Occupancy occupancy_from_water(const cv::Mat& water)
{
    Occupancy result;
    result.rows = (water.rows + OCCUPANCY_CELL - 1) / OCCUPANCY_CELL;
    result.cols = (water.cols + OCCUPANCY_CELL - 1) / OCCUPANCY_CELL;
    std::vector<std::uint8_t> grid(static_cast<std::size_t>(result.rows) * result.cols, 0);
    for (int y = 0; y < water.rows; ++y)
    {
        const std::uint8_t* row = water.ptr<std::uint8_t>(y);
        std::uint8_t* cells = grid.data() + static_cast<std::size_t>(y / OCCUPANCY_CELL) * result.cols;
        for (int x = 0; x < water.cols; ++x)
        {
            if (row[x])
            {
                cells[x / OCCUPANCY_CELL] = 1;
                ++result.water_pixels;
            }
        }
    }
    result.packed.assign((grid.size() + 7) / 8, 0);
    for (std::size_t i = 0; i < grid.size(); ++i)
    {
        if (grid[i])
        {
            result.packed[i >> 3] |= static_cast<std::uint8_t>(0x80u >> (i & 7));
        }
    }
    return result;
}

Occupancy MaskCache::occupancy(const View& view)
{
    return occupancy_from_water(water(view));
}

Labels MaskCache::labels(const View& view, int min_area)
{
    const std::string key = "l" + std::to_string(min_area) + ":" + view.mask_key;
    Entry entry;
    Labels result;
    if (get(key, entry))
    {
        result.labels.create(entry.rows, entry.cols, CV_32S);
        decompress(entry.payload, result.labels.ptr(), static_cast<std::size_t>(entry.rows) * entry.cols * 4);
        result.areas = entry.extra;
        return result;
    }
    const cv::Mat water = this->water(view);
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(water, result.labels, stats, centroids, 8, CV_32S);
    bool all = true;
    bool any = false;
    for (int i = 1; i < count; ++i)
    {
        const bool keep = min_area <= 0 || stats.at<int>(i, cv::CC_STAT_AREA) >= min_area;
        all &= keep;
        any |= keep;
    }
    if (!any)
    {
        result.labels = cv::Mat::zeros(water.size(), CV_32S);
        result.areas = {0};
    }
    else if (all)
    {
        result.areas.assign(static_cast<std::size_t>(count), 0);
        for (int i = 1; i < count; ++i)
        {
            result.areas[static_cast<std::size_t>(i)] = stats.at<int>(i, cv::CC_STAT_AREA);
        }
    }
    else
    {
        cv::Mat filtered(water.size(), CV_8U);
        for (int y = 0; y < water.rows; ++y)
        {
            const int* l = result.labels.ptr<int>(y);
            std::uint8_t* f = filtered.ptr<std::uint8_t>(y);
            for (int x = 0; x < water.cols; ++x)
            {
                f[x] = l[x] > 0 && stats.at<int>(l[x], cv::CC_STAT_AREA) >= min_area ? 1 : 0;
            }
        }
        const int second = cv::connectedComponentsWithStats(filtered, result.labels, stats, centroids, 8, CV_32S);
        result.areas.assign(static_cast<std::size_t>(second), 0);
        for (int i = 1; i < second; ++i)
        {
            result.areas[static_cast<std::size_t>(i)] = stats.at<int>(i, cv::CC_STAT_AREA);
        }
    }
    entry.rows = result.labels.rows;
    entry.cols = result.labels.cols;
    entry.extra = result.areas;
    entry.payload = compress(result.labels.ptr(), static_cast<std::size_t>(entry.rows) * entry.cols * 4);
    put(key, std::move(entry));
    return result;
}

int boundary_pixels(int width, int height)
{
    const int pixels = static_cast<int>(std::nearbyint(std::hypot(static_cast<double>(width), static_cast<double>(height)) * 0.0015));
    return std::max(2, std::min(16, pixels));
}

cv::Mat ellipse_kernel(int radius)
{
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1));
}

std::vector<int> boundary_at_pixels(const Labels& labels, int radius, const std::vector<int>& u, const std::vector<int>& v)
{
    const cv::Mat kernel = ellipse_kernel(radius);
    std::vector<std::pair<int, int>> offsets;
    for (int y = 0; y < kernel.rows; ++y)
    {
        for (int x = 0; x < kernel.cols; ++x)
        {
            if (kernel.at<std::uint8_t>(y, x))
            {
                offsets.emplace_back(y - radius, x - radius);
            }
        }
    }
    const int height = labels.labels.rows;
    const int width = labels.labels.cols;
    std::vector<int> result(u.size(), 0);
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        if (labels.labels.at<int>(v[i], u[i]) != 0)
        {
            continue;
        }
        int closest = INT_MAX;
        for (const auto& [dy, dx] : offsets)
        {
            const int yy = v[i] + dy;
            const int xx = u[i] + dx;
            if (yy < 0 || yy >= height || xx < 0 || xx >= width)
            {
                continue;
            }
            const int label = labels.labels.at<int>(yy, xx);
            if (label > 0 && label < closest)
            {
                closest = label;
            }
        }
        result[i] = closest == INT_MAX ? 0 : closest;
    }
    return result;
}

} // namespace ai3d::waterpost
