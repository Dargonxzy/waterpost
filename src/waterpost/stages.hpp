#pragma once

#include "types.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace ai3d::waterpost
{

// 每块独立（可分机）：detect / candidates / samples / write。全局（单机）：reconcile / active / model / finalize。
void run_detect(const Config& config, const std::string& block);
void run_reconcile(const Config& config);
void run_candidates(const Config& config, const std::string& block);
void run_active(const Config& config);
void run_samples(const Config& config, const std::string& block);
void run_model(const Config& config);
void run_write(const Config& config, const std::string& block);
void run_finalize(const Config& config);

void run_evaluate(const std::filesystem::path& project, const std::vector<std::pair<std::string, std::filesystem::path>>& variants,
                  const std::filesystem::path& mask_dir, const std::filesystem::path& images, int step,
                  const std::filesystem::path& csv, int gpu, int water_class, const std::vector<std::string>& only);

// 参与处理的块：工程块清单与 LOD 块的交集。
std::vector<std::string> pipeline_blocks(const Config& config);

} // namespace ai3d::waterpost
