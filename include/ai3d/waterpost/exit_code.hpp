#pragma once

// 进程退出码：与 blockgs_error_codes.ExitCode 中水面预处理用到的编号一致，集群调度按这些编号归类失败。

namespace ai3d::waterpost
{

enum class ExitCode : int
{
    success = 0,
    unknown = 9,
    file_not_found = 10,
    file_permission = 17,
    file_parse_failed = 18,
    param_invalid = 165,
    mem_insufficient = 211,
    water_detect_source_invalid = 3501,
    water_detect_chunk_invalid = 3502,
    water_detect_input_invalid = 3503,
    water_detect_vote_failed = 3504,
    water_detect_save_failed = 3505,
    water_detect_failed = 3506,
    water_recon_input_not_found = 3511,
    water_recon_input_invalid = 3512,
    water_recon_component_failed = 3513,
    water_recon_surface_failed = 3514,
    water_recon_save_failed = 3515,
    water_recon_failed = 3516,
    water_fill_manifest_invalid = 3521,
    water_fill_bbox_invalid = 3522,
    water_fill_generate_failed = 3523,
    water_fill_validate_failed = 3524,
    water_merge_failed = 3525,
    water_fill_save_failed = 3526,
    water_fill_merge_failed = 3527,
};

// 退出码的名字（与 Python 端 BLOCKGS_ERROR 行里的 name 相同，例如 ERR_GS_WATER_DETECT_FAILED）。
const char* exit_code_name(ExitCode code);

} // namespace ai3d::waterpost
