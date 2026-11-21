#pragma once

#include "types.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace ai3d::waterpost
{

// 分片 / 每块独立（可分机）：assign（--shard）/ detect / candidates / samples / write。
// 全局（单机）：scene / reconcile / active / model / finalize。assign 与 scene 在 scene.hpp。
void run_detect(const Config& config, const std::string& block);
void run_reconcile(const Config& config);
void run_candidates(const Config& config, const std::string& block);
void run_active(const Config& config);
void run_samples(const Config& config, const std::string& block);
void run_model(const Config& config);
void run_write(const Config& config, const std::string& block);
void run_finalize(const Config& config);

// evaluate 的局部区域（例如 floaters.json 里的候选簇）：每个视图把包围盒投到图像上，只在这块像素里与照片比较，
// 用来判断"去掉这团高斯后渲染是否更接近照片"（独立于重建本身的观测证据）。
struct EvalRegion
{
    std::string block;
    std::string id;
    std::string decision;
    std::size_t points = 0;
    Vec3 low{};
    Vec3 high{};
};

void run_evaluate(const std::filesystem::path& project, const std::vector<std::pair<std::string, std::filesystem::path>>& variants,
                  const std::filesystem::path& mask_dir, const std::filesystem::path& images, int step,
                  const std::filesystem::path& csv, int gpu, int water_class, const std::vector<std::string>& only,
                  const Config* poses, const std::vector<EvalRegion>& regions = {},
                  const std::filesystem::path& region_csv = {}, int region_pad = 6);

// 参与处理的块：work/scene 里的块（blocks 模式为工程块清单 ∩ LOD 块，poses 模式为 LOD 块）。
std::vector<std::string> pipeline_blocks(const Config& config);

} // namespace ai3d::waterpost
