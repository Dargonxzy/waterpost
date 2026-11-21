#pragma once

// 文件系统小工具：UTF-8 路径、整文件读写、原子发布、Python Path.resolve() 等价的规范路径。

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

// 路径的 UTF-8 文本（Windows 上保持反斜杠，与 Python str(Path) 相同）。
std::string path_utf8(const std::filesystem::path& path);

// UTF-8 文本转路径。
std::filesystem::path utf8_path(const std::string& text);

// 整文件读入；打不开抛 WaterError(file_not_found)。
std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path);

std::string read_file_text(const std::filesystem::path& path);

// Copy bytes without chmod/sendfile, which object-storage FUSE mounts may reject.
void copy_file_contents(const std::filesystem::path& source, const std::filesystem::path& target);

// 先写 temporary 再 rename 到 path；失败时删除临时文件并抛错。
void write_file_atomic(const std::filesystem::path& path, const void* data, std::size_t size,
                       const std::filesystem::path& temporary);

// Python Path(path).resolve()：绝对化并解析已存在部分的符号链接与 . / ..。
std::string canonical_path_text(const std::filesystem::path& path);

// 删除文件（不存在时忽略）。
void remove_file_if_exists(const std::filesystem::path& path);

// 进程号（临时文件名用）。
long process_id();

} // namespace ai3d::waterpost
