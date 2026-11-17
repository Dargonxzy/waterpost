#pragma once

// 大地坐标换算（WGS84）：地心坐标（ECEF）-> 经纬度高，经纬度 -> UTM（Krüger 级数，带内误差远小于 1 mm）。
// 用来把投影坐标（UTM）下的位姿换到 LOD 本地坐标：LOD 根节点 transform 的平移是本地原点的 ECEF 坐标，
// 本地坐标 = 投影坐标 - 原点的投影坐标（本地轴与投影格网同向，已在滑梯数据上与 bbox.json 的 Normalizer 对上）。

#include <array>
#include <string>

namespace ai3d::waterpost
{

struct Geodetic
{
    double latitude = 0;  // 弧度
    double longitude = 0; // 弧度
    double height = 0;    // 椭球高（m）
};

Geodetic ecef_to_geodetic(double x, double y, double z);

// UTM 正算：zone 1..60，south 为南半球（北向加 10000 km）。返回 {东, 北}。
std::array<double, 2> utm_forward(double latitude, double longitude, int zone, bool south);

// 经度所在的标准 UTM 带。
int utm_zone_of(double longitude);

// 解析 --pose-crs："auto" / "none" / "EPSG:326NN"（北）/ "EPSG:327NN"（南）。zone = 0 表示 auto，-1 表示 none。
void parse_pose_crs(const std::string& text, int& zone, bool& south);

} // namespace ai3d::waterpost
