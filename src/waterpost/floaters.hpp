#pragma once

// 去噪的第二步：点数达到 denoise_min_points、但仍与主体分离的中等连通簇（例如树冠上方 36～72 点的密集漂浮团）。
// 只按几何与水位关系给出三种决策，不把"点数够多"当作保留理由，也不把"孤立"本身当作删除理由：
//   keep_*    与可信主体（大簇）距离近、或跨块边界，保留；
//   remove_*  与主体隔开且下方悬空（下方最近的主体表面 / 水面离簇底足够远），删除；
//   review_*  隔开但贴近下方支撑、呈细长线状等，保留并记录，交给人工或其他证据。

#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

struct FloaterParams
{
    std::size_t min_points = 30;     // 小于它的簇由原小簇规则处理，不进入本判定
    std::size_t max_points = 2000;   // 大于它的簇视为主体（锚点），不做判定
    double gap = 1.5;                // 到主体的最近中心距离 >= gap 才算"分离"
    double clearance = 1.5;          // 簇底到下方主体表面 / 水面的高差 >= clearance 才算"悬空"
    double column_pad = 0.5;         // 找下方支撑时簇 XY 包围盒外扩
    double elongation = 8.0;         // 主轴 / 次轴标准差之比 >= elongation 且长度 >= elongated_length：线状结构，待核查
    double elongated_length = 2.0;
};

struct FloaterCandidate
{
    std::int32_t component = -1;
    std::size_t points = 0;
    bool border = false;
    Vec3 centroid{};
    Vec3 low{};
    Vec3 high{};
    double spacing = 0;           // 簇内最近邻距离的中位数
    double gap = 0;               // 到主体最近中心距离（搜索上限内没有主体时为 +inf）
    double support_top = 0;       // 簇下方柱体内主体点的最高 z（没有时为 NaN）
    double water_level = 0;      // 簇下方水体水位（没有时为 NaN）
    double clearance = 0;         // 簇底到 max(support_top, water_level) 的高差（都没有时为 +inf）
    bool covered = false;         // 簇上方柱体内是否有主体点（树冠下、桥下等）
    double elongation = 0;
    double length = 0;            // 主轴方向的 4 sigma 长度
    double mean_opacity = 0;
    int semantic = -1;            // 多数语义类别（bin 无语义字节时为 -1）
    // 照片核验（verify_floaters_with_photos）：在看得到该簇的训练视图里，分别渲染含 / 不含候选簇的块，
    // 只在簇包围盒的投影区域内与照片比较。
    std::string geometry;         // 几何判定（下面 decision 的初值）
    std::string photo = "photo_not_checked"; // photo_favors_drop / photo_favors_keep / photo_inconclusive / photo_unavailable
    std::size_t photo_views = 0;  // 区域落在画面内的视图数
    std::size_t photo_differ = 0; // 其中含 / 不含渲染有差别（没被完全遮挡）的视图数
    std::size_t photo_wins = 0;   // 其中"不含"更接近照片的视图数
    double photo_keep_psnr = 0;   // 区域像素上的 PSNR（dB）
    double photo_drop_psnr = 0;
    std::string decision;         // keep_near_structure / keep_block_border / remove_suspended_floater /
                                  // review_detached_supported / review_detached_covered / review_unobserved_below / review_elongated
    bool remove() const
    {
        return decision.rfind("remove_", 0) == 0;
    }
};

// component：每个高斯所属连通簇编号（已删除的点为 -1），sizes[c] 为簇 c 的点数，border[c] 表示碰到块边界。
// water_level(xlo, ylo, xhi, yhi)：该 XY 范围下方已接受水体的最高水位，没有时返回 NaN。
std::vector<FloaterCandidate> classify_floaters(const Gaussians& g, const std::vector<std::int32_t>& component,
                                                const std::vector<std::size_t>& sizes, const std::vector<std::uint8_t>& border,
                                                const FloaterParams& params,
                                                const std::function<double(double, double, double, double)>& water_level);

struct View;

struct FloaterPhotoParams
{
    std::filesystem::path photos;   // 训练照片目录（与 evaluate --photos 相同的找法）
    std::size_t max_views = 24;     // 每个簇最多用多少个视图
    std::size_t min_views = 5;      // 至少这么多个有差别的视图才下结论
    double min_gain = 0.1;          // 去掉后区域 PSNR 至少提高 / 降低这么多 dB
    double win_ratio = 0.6;         // 且至少这个比例的视图里去掉更接近照片（反之 <= 1 - win_ratio 为支持保留）
    int pad = 6;                    // 投影矩形外扩像素
    int gpu = 0;
};

// 对 check[k] 为真的候选簇做照片核验，结果写回 candidates[k] 的 photo_* 字段；返回统计。
// context：邻块的最细层高斯，两次渲染都带上（否则斜视图里簇背后是黑色，任何有颜色的团块都显得更接近照片）。
// 照片目录不存在或没有视图时，全部标记为 photo_unavailable。
void verify_floaters_with_photos(const Gaussians& g, const std::vector<std::int32_t>& component,
                                 std::vector<FloaterCandidate>& candidates, const std::vector<std::uint8_t>& check,
                                 const std::vector<View>& views, const FloaterPhotoParams& params, const Gaussians* context,
                                 std::size_t& rendered_views, double& seconds);

// 照片核验之后的最终决策（floater_check = 1）：只有照片支持删除、且不是块边界 / 线状结构的簇才删除；
// 几何上悬空但照片不支持或无法核验的簇降为待核查。
void combine_floater_evidence(FloaterCandidate& f);

} // namespace ai3d::waterpost
