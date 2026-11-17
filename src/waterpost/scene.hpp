#pragma once

// 场景准备：split（写分工清单）+ assign（按清单处理视图，可分机）+ scene（全局合并）。
//
// 两种输入模式产出同一份 work/scene：
//   blocks：块清单、块范围、每块相机取自 <project>/artifacts/blocks，补水范围取自 bbox.json / roi.json；
//   poses ：只有 LOD + 位姿 JSON + 掩码。块取 LOD 中的块，块范围取该块最细层叶节点包围盒的并集，
//           每块的视图由"视图看到该块的叶节点、相机在块附近"确定，补水范围取 --bbox（缺省为 LOD 范围）∩ --roi。
// assign 对每张视图只解码一次掩码，得到 32 像素占据格网；scene 把各分片合成：
//   work/scene/scene.json      块（名字、范围、视图键）与补水范围
//   work/scene/views.json      全部视图记录（按掩码键排序）
//   work/scene/occupancy.npz   全部视图的占据格网
// 之后 detect / reconcile / candidates 只读 work/scene，不再碰分块成果与掩码 PNG 的占据统计。
// blocks 模式可以不跑 split / assign / scene，直接从 detect 开始（见 has_scene）。

#include "camera.hpp"
#include "project.hpp"
#include "types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace ai3d::waterpost
{

struct SceneBlock
{
    std::string name;
    Box3 bounds;
    std::vector<std::string> view_keys; // 掩码键，按视图 key() 排序（与 block_views 的顺序相同）
};

struct Scene
{
    std::string mode;
    std::vector<SceneBlock> blocks;
    Domain domain; // 不含 LOD 瓦片（reconcile 再补）

    const SceneBlock* block(const std::string& name) const;
};

// split：把全部视图分成 parts 份，写 work/assign/lists/part_<i>.list（每行一个图像名），作为各机器的分工。
//   interleave = 第 k 张（按图像名排序）归第 k mod parts 份（默认，有水视图分得更匀）；否则按顺序切成连续段。
void run_split(const Config& config, int parts, bool interleave);
// assign：处理清单里的图像（list 为空 = 全部视图），结果写 work/assign/<清单名或 all>.npz/.json。
void run_assign(const Config& config, const std::filesystem::path& list);
// assign 之前清掉 work/assign 里旧的分片结果（不动 lists/）。
void clear_assign_results(const Config& config);
// scene：合并 work/assign 里的全部分片结果；要求每张视图恰好被一个分片处理过。
void run_scene(const Config& config);

// work/scene 是否存在。blocks 模式下 scene 可选：没有时 load_scene / scene_views 直接从 artifacts/blocks 构造
// （与改动前的流程相同），detect 自己解码占据格网并写 detect/<块>/occupancy.npz。poses 模式必须先有 scene。
bool has_scene(const Config& config);
// 读取 work/scene（进程内缓存）。
const Scene& load_scene(const Config& config);
// 掩码键 -> 视图记录（work/scene/views.json）。
std::vector<View> scene_views(const Config& config, const std::vector<std::string>& keys);
// 掩码键 -> 占据格网（work/scene/occupancy.npz；没有 scene 时报错，调用前先看 has_scene）。
const std::unordered_map<std::string, Occupancy>& scene_occupancy(const Config& config);

// 输入里参与处理的块（scene 之前用）：blocks 模式为工程块清单 ∩ LOD 块，poses 模式为 LOD 块。
std::vector<std::string> input_blocks(const Config& config);

// 位姿（poses 模式）读成视图：--poses（单个 JSON）或 --cameras + --images（两个 JSON）；畸变相机换成针孔并记下
// 原始相机；投影坐标按 --pose-crs 换到 LOD 本地坐标；掩码按 --mask-dir / --image-root 找，没有掩码的视图跳过。
std::vector<View> read_pose_views(const Config& config);

} // namespace ai3d::waterpost
