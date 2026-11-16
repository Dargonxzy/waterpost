#pragma once

// gs-exporter LOD（tileset.json + gssp .bin）的读取与改写。
//
// 支持两种布局：
//   - 旧布局：tileset.json 的根下挂各块 L*/<块>/tile_N.json，每个节点各有一个 JSON；
//   - fix 布局：tileset.json 指向 fix/json/g*.json 与粗层代理，完整源树在 tileset.source.json。
// 两种布局里内容文件都是 L<层>/<块>/tile__*.bin，块名取路径第二段。只读写各块最细层（叶节点）的 bin。

#include "json_util.hpp"
#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

constexpr std::size_t GSSP_RECORD = 32;
constexpr std::size_t GSSP_HEADER = 32;
constexpr std::uint32_t GSSP_PAD = 4096;

struct LodLeaf
{
    std::string uri;   // 相对 LOD 根目录的 POSIX 路径
    int level = 0;
    Box3 box;          // 节点包围盒（JSON boundingVolume，轴对齐）
};

struct LodBlock
{
    std::string name;
    int level = 0;                 // 最细层号
    std::vector<LodLeaf> leaves;   // 最细层叶节点，按 uri 排序
};

struct LodIndex
{
    std::filesystem::path root;
    bool has_fix = false;
    bool semantic = false;          // asset.extras.ai3dSemanticSplat
    std::string source_entry;       // 用来列节点的清单（tileset.source.json 或 tileset.json）
    std::vector<LodBlock> blocks;

    const LodBlock* block(const std::string& name) const;
};

LodIndex read_lod_index(const std::filesystem::path& root);

// 一个叶节点 bin 的原始记录（去掉末尾补齐记录）。
struct LeafContent
{
    std::uint32_t padded = 0;
    std::uint32_t count = 0;
    bool semantic = false;               // 补齐记录或真实记录用了语义字节
    std::vector<std::uint8_t> records;   // count * 32
};

LeafContent read_leaf(const std::filesystem::path& path);

// 写一个 bin：记录数补齐到 4096，文件头写 box。
void write_leaf(const std::filesystem::path& path, const std::vector<std::uint8_t>& records, std::uint32_t count,
                const Box3& box, bool semantic);

// 读一个块最细层的全部高斯；leaf_offsets[i] 为第 i 个叶节点的第一个点下标（长度 = 叶数 + 1）。
Gaussians read_block_gaussians(const LodIndex& index, const LodBlock& block, std::vector<std::size_t>* leaf_offsets);

// 半精度转换。
float half_to_float(std::uint16_t bits);
std::uint16_t float_to_half(float value);

// 生成一条水面记录：轴对齐协方差 diag(sx², sy², sz²)，RGBA，语义字节。
void encode_water_record(std::uint8_t* out, const Vec3& position, const Vec3& scale, const std::array<double, 3>& rgb,
                         double opacity, bool semantic, int water_class);

// 叶节点改写结果（apply-write 记录，finalize 用来改 JSON）。
struct LeafChange
{
    std::string uri;
    Box3 box;
    std::uint32_t padded = 0;
    std::uint32_t count = 0;
    // 非空时这是一个新增的兄弟叶节点（叶节点装不下的水面点），挂在 sibling_of 所在节点的同一父节点下。
    std::string sibling_of;
};

// bin 文件头里的（补齐后）记录数。
std::uint32_t read_leaf_padded(const std::filesystem::path& path);
// 从目录 from 到文件 to 的 POSIX 相对路径。
std::string relative_posix(const std::string& from_dir, const std::string& to);

// 输出 LOD：把输入 LOD 中未被改写的文件复制（或硬链接）到 output，再按 changes 改写全部清单 JSON 的
// 包围盒与 gsplatPointCount（叶节点改值，祖先节点取并集）。
void finalize_lod(const std::filesystem::path& input, const std::filesystem::path& output,
                  const std::vector<LeafChange>& changes, bool hard_link);

// "a/b/../c" -> "a/c"（POSIX）。
std::string normalize_posix(const std::string& path);
std::string dirname_posix(const std::string& path);

} // namespace ai3d::waterpost
