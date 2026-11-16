#pragma once

// 工程输入与阶段之间的公共工具：块清单、块范围、相机视图、工程范围（Domain）、视图记录、配置读写。

#include "camera.hpp"
#include "gpu.hpp"
#include "json_util.hpp"
#include "types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

// artifacts/blocks/valid_chunk_list.txt（或 valid_chunks.txt）里的块；没有清单时取有 sparse/0 的子目录。
std::vector<std::string> project_blocks(const std::filesystem::path& project);

std::filesystem::path block_dir(const std::filesystem::path& project, const std::string& name);

// block.json 的 core_bounds，或 center.txt ± extent.txt / 2。
Box3 block_core_bounds(const std::filesystem::path& project, const std::string& name);

// 块的相机视图（sparse/0），按掩码路径去重、按 key 排序；缺掩码报错。
std::vector<View> block_views(const std::filesystem::path& project, const std::string& name,
                              const std::filesystem::path& mask_dir);

PinholeCamera pinhole(const View& view);
void require_pinhole(const std::vector<View>& views);

// 工程范围（bbox.json / BoundingBox.json 的 CurrentBoundingBox）。
// 补水范围：工程 bbox ∩ ROI 多边形（roi.json，有则用）∩ LOD 各块瓦片范围（XY）。
struct Domain
{
    Vec3 lower{};
    Vec3 upper{};
    std::vector<std::array<double, 2>> polygon;   // 本地坐标；空 = 不限
    std::vector<std::array<double, 4>> tiles;     // xmin, ymin, xmax, ymax；空 = 不限
    double horizontal_clearance(double x, double y, int axis) const;
    std::array<double, 2> sampling_tolerance(double spacing) const;
    bool inside(double x, double y) const;
    // 到 ROI 多边形边界的距离（内正外负）；没有多边形时为 +inf。
    double polygon_distance(double x, double y) const;
    Json json() const;
    static Domain from_json(const Json& value);
};
Domain load_domain(const std::filesystem::path& project);

Json view_record(const View& view);
View view_from_record(const Json& record);

Json config_to_json(const Config& config);
Config config_from_json(const Json& value);
void save_config(const Config& config);
Config load_config(const std::filesystem::path& work);

// 视图可能看到盒内的点（views_seeing_box，margin 为图像尺寸的比例）。
bool view_sees_box(const View& view, const Vec3& low, const Vec3& high, double margin);

std::string canonical_key(const std::filesystem::path& path);

// numpy 风格线性插值分位数（输入会被排序）。
double quantile(std::vector<double> values, double q);
double median(std::vector<double> values);

void log(const char* format, ...);

} // namespace ai3d::waterpost
