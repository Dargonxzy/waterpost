#pragma once

// 语义掩码：每张图只解码一次，得到 1 位水图（类别 == water_class），按字节预算缓存 zstd 压缩结果。
// 由水图派生：32 像素占据格网（ROI 预检与 apply 反票）、8 连通分量标签（面积 < 256 的分量去掉，编号与
// Python 版 cv2.connectedComponentsWithStats 两遍规则相同）。

#include "camera.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai3d::waterpost
{

struct Labels
{
    cv::Mat labels;          // CV_32S
    std::vector<int> areas;  // 下标 = 标签，0 号为背景
};

class MaskCache
{
public:
    MaskCache(int water_class, std::size_t budget_bytes);

    // CV_8U 0/1，尺寸 (view.height, view.width)。
    cv::Mat water(const View& view);
    Occupancy occupancy(const View& view);
    Labels labels(const View& view, int min_area = 256);

    std::size_t decodes() const;

private:
    struct Entry
    {
        std::vector<std::uint8_t> payload;
        int rows = 0;
        int cols = 0;
        std::vector<int> extra;
    };
    bool get(const std::string& key, Entry& out);
    void put(const std::string& key, Entry entry);

    int m_water_class;
    std::size_t m_budget;
    std::size_t m_bytes = 0;
    std::size_t m_decodes = 0;
    mutable std::mutex m_mutex;
    std::list<std::string> m_order;
    std::unordered_map<std::string, std::pair<Entry, std::list<std::string>::iterator>> m_entries;
};

// 进程内共享的掩码缓存（按水体类别区分，2 GiB 预算）：同一进程连续处理多块时，相邻块共用的视图只解码一次。
MaskCache& shared_mask_cache(int water_class);

// 外环宽度：hypot(w, h) * 0.0015 四舍五入，夹在 [2, 16]。
int boundary_pixels(int width, int height);
cv::Mat ellipse_kernel(int radius);

// boundary_at_pixels：陆地像素 (u, v) 周围椭圆核内最小的正标签；水像素与无标签处为 0。
std::vector<int> boundary_at_pixels(const Labels& labels, int radius, const std::vector<int>& u,
                                    const std::vector<int>& v);

Occupancy occupancy_from_water(const cv::Mat& water);

} // namespace ai3d::waterpost
