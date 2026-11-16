#pragma once

// apply 各阶段共用：读 reconcile 报告、全局格网、块记录、视图目录与 detect 交接数组。

#include "camera.hpp"
#include "json_util.hpp"
#include "project.hpp"
#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai3d::waterpost
{

struct GridSpec
{
    double origin[2] = {0, 0};
    std::int64_t shape[2] = {0, 0};
    double spacing = 0.4;
};

struct BlockRecord
{
    std::string name;
    Box3 bounds;
    std::int64_t water = 0;
};

struct Surface
{
    int water_id = 0;
    double height = 0;
    double sigma = 0;
    std::map<std::string, std::vector<int>> signature; // mask_key -> component labels
    std::array<double, 3> rgb{};
};

struct Report
{
    Json json;
    GridSpec grid;
    Domain domain;
    std::vector<BlockRecord> blocks;
    std::vector<Surface> surfaces;

    int block_index(const std::string& name) const;
    const BlockRecord& block(const std::string& name) const;
};

Report load_report(const Config& config);
std::vector<View> load_catalog(const Config& config);

std::filesystem::path detect_dir(const Config& config, const std::string& block);

// 自由函数：detect 交接。
struct LandGrid
{
    std::vector<std::int64_t> cells; // 2n 绝对格号
    std::vector<float> zmin, zmax;
    std::vector<std::int32_t> count;
};
bool load_land_grid(const Config& config, const std::string& block, LandGrid& grid);
std::vector<double> load_seed_xyz(const Config& config, const std::string& block);
// 占据格网：mask_key -> Occupancy（第一个出现的块优先）。
void load_occupancy(const Config& config, const std::string& block, std::unordered_map<std::string, Occupancy>& out);

// 候选/样本交接。
struct Candidates
{
    std::vector<std::int64_t> cells; // 2n 全局格号
    std::vector<std::int32_t> ids;
    std::vector<std::uint8_t> support;
    Json metadata;
};
Candidates load_candidates(const Config& config, const std::string& block);

inline std::uint64_t cell_key(std::int64_t x, std::int64_t y)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) | static_cast<std::uint32_t>(y);
}

} // namespace ai3d::waterpost
