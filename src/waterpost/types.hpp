#pragma once

// 公共小类型：块内高斯（从 LOD 最细层 bin 解码）、包围盒、配置。

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

using Vec3 = std::array<double, 3>;

struct Box3
{
    Vec3 minimum{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()};
    Vec3 maximum{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity()};

    bool valid() const
    {
        return minimum[0] <= maximum[0] && minimum[1] <= maximum[1] && minimum[2] <= maximum[2];
    }
    void expand(const Vec3& p)
    {
        for (int k = 0; k < 3; ++k)
        {
            minimum[k] = std::min(minimum[k], p[k]);
            maximum[k] = std::max(maximum[k], p[k]);
        }
    }
    void expand(const Box3& other)
    {
        if (!other.valid())
        {
            return;
        }
        expand(other.minimum);
        expand(other.maximum);
    }
};

// 一个块最细层的全部高斯（按叶节点顺序拼接，不含 bin 的补齐记录）。
struct Gaussians
{
    std::size_t count = 0;
    std::vector<double> xyz;          // 3n
    std::vector<float> cov;           // 6n：xx xy xz yy yz zz
    std::vector<float> opacity;       // n：不透明度（已过 sigmoid）
    std::vector<std::uint8_t> rgba;   // 4n：bin 里的 RGBA
    std::vector<float> sigma_max;     // n：协方差最大特征值的平方根
    std::vector<float> normal_z;      // n：最短轴方向的 |z| 分量
    std::vector<std::uint8_t> semantic; // 2n：bin 第 26/27 字节

    const double* point(std::size_t i) const
    {
        return xyz.data() + 3 * i;
    }
};

constexpr int OCCUPANCY_CELL = 32;

// 32 像素占据格网：格子里有任何一个水像素即为 1（np.packbits 行主序）。
struct Occupancy
{
    int rows = 0;
    int cols = 0;
    std::int64_t water_pixels = 0;
    std::vector<std::uint8_t> packed; // np.packbits(grid) 的字节

    bool at(int row, int col) const
    {
        const std::size_t bit = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
        return (packed[bit >> 3] >> (7 - (bit & 7))) & 1u;
    }
};

// 运行参数（与 Python DEFAULTS 同名同值）。
struct Config
{
    std::filesystem::path project;
    std::filesystem::path lod;
    std::filesystem::path mask_dir;
    std::filesystem::path work;
    std::filesystem::path output;
    std::string mode = "blocks";        // blocks（有分块成果）| poses（LOD + 位姿 JSON + 掩码）
    std::filesystem::path poses;        // poses：位姿 JSON
    std::filesystem::path bbox;         // poses：可选 bbox.json（工程范围与 ROI 的 Normalizer）
    std::filesystem::path roi;          // poses：可选 roi.json（需要 bbox）
    double view_margin = 0.5;           // poses：相机中心 XY 可在块范围外 view_margin * 块边长之内
    int min_view_leaves = 1;            // poses：视图至少看到块的这么多个最细层叶节点才算该块的视图
    int water_class = 9;
    double cell_size = 1.0;
    double spacing = 1.6;
    double footprint_spacing = 0.4;
    bool debug = false;
    int threads = 0;
    int gpu = 0;
    bool link_output = false;

    bool water_carve = true;
    int carve_erosion_cells = 2;
    double carve_surface_tolerance = 0.2;
    double carve_above_band = 1.0;
    double carve_max_depth = 50.0;
    double carve_below_margin = 0.5;
    double carve_below_reach = 10.0;
    double carve_float_band = 3.0;
    double carve_float_visible_water = 0.5;
    double carve_bank_visible_water = 0.3;
    double carve_float_max = 15.0;
    double carve_float_geometry_water = 0.2;

    // 通用漂浮物去噪：中心距离 <= denoise_radius 的高斯连成簇，少于 denoise_min_points 的小簇删除。
    // 方法：cluster（欧氏聚类小簇，默认）、sor（统计离群点）、radius（半径离群点）。
    bool denoise = true;
    std::string denoise_method = "cluster";
    int denoise_k = 20;                 // sor：近邻数
    double denoise_std_ratio = 2.0;     // sor：均值 + std_ratio * 标准差
    double denoise_max_distance = 2.0;  // sor：近邻搜索的最大距离
    double denoise_radius = 0.5;        // radius / cluster：半径
    int denoise_min_neighbors = 5;      // radius：半径内最少邻居
    int denoise_min_points = 30;        // cluster：最小簇点数
};

} // namespace ai3d::waterpost
