#pragma once

// 岸线定水位（water_postprocessing.rebuild.shore_level）：在不高于中位数的岸线高度里找窗口 [c - tol/2, c + tol/2]
// 内点数最多的 c（并列取最低），内点不足 3 个时退回 20% 分位数 ± tol；水位取内点中位数。

#include <cstdint>
#include <vector>

namespace ai3d::waterpost
{

struct ShoreLevel
{
    double height = 0;
    std::vector<std::uint8_t> inliers;
};

// O(n log n)：排序后对每个候选用二分查找计窗口内点数（比较式与逐点 |z - c| <= tol/2 完全相同）。
ShoreLevel shore_level(const std::vector<double>& z, double tolerance = .6);

} // namespace ai3d::waterpost
