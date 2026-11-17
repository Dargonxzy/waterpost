// reconcile：只读各块 detect 的交接文件，按共享的掩码分量把岸线锚点合成水体，用岸线定水位
// （water_postprocessing.distributed_reconcile.reconcile_prepared + rebuild.shore_level + domain）。

#include "error.hpp"
#include "fs_util.hpp"
#include "lod.hpp"
#include "npz_util.hpp"
#include "project.hpp"
#include "scene.hpp"
#include "shore.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace ai3d::waterpost
{
namespace
{

} // namespace

ShoreLevel shore_level(const std::vector<double>& z, double tolerance)
{
    const double med = median(z);
    const double half = tolerance / 2;
    std::vector<double> sorted = z;
    std::sort(sorted.begin(), sorted.end());
    // 候选 = 不高于中位数的高度（升序）；分数 = |value - c| <= half 的个数。浮点下 |value - c| 随 value 远离 c 单调
    // 不减，所以满足条件的是排序数组里的一段，两端用与逐点比较完全相同的表达式二分查找。
    std::size_t best_score = 0;
    double center = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < sorted.size() && sorted[k] <= med; ++k)
    {
        const double c = sorted[k];
        if (k > 0 && sorted[k - 1] == c)
        {
            continue;
        }
        const auto first = std::partition_point(sorted.begin(), sorted.end(),
                                                [&](double value) { return value < c && std::abs(value - c) > half; });
        const auto last = std::partition_point(first, sorted.end(),
                                               [&](double value) { return !(value > c && std::abs(value - c) > half); });
        const std::size_t score = static_cast<std::size_t>(last - first);
        if (score > best_score)
        {
            best_score = score;
            center = c; // 候选升序：只有严格更大才替换，并列时保留最低的
        }
    }
    ShoreLevel result;
    result.inliers.assign(z.size(), 0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        result.inliers[i] = std::abs(z[i] - center) <= half;
        count += result.inliers[i];
    }
    if (count < 3)
    {
        center = quantile(z, .2);
        count = 0;
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            result.inliers[i] = std::abs(z[i] - center) <= tolerance;
            count += result.inliers[i];
        }
    }
    if (count)
    {
        std::vector<double> chosen;
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if (result.inliers[i])
            {
                chosen.push_back(z[i]);
            }
        }
        result.height = median(chosen);
    }
    else
    {
        result.height = center;
    }
    return result;
}

namespace
{

struct UnionFind
{
    std::vector<std::size_t> parent;
    explicit UnionFind(std::size_t n) : parent(n)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }
    std::size_t find(std::size_t a)
    {
        while (parent[a] != a)
        {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    }
    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a != b)
        {
            parent[std::max(a, b)] = std::min(a, b);
        }
    }
};

// extend_water_block_bounds：块的 XY 范围向工程边界延伸，直到碰到其他块。
std::vector<Box3> extend_bounds(const std::vector<Box3>& boxes, const Domain& domain)
{
    const std::size_t count = boxes.size();
    std::vector<std::array<double, 2>> low(count), high(count);
    std::vector<std::uint8_t> active(count, 0);
    for (std::size_t j = 0; j < count; ++j)
    {
        for (int a = 0; a < 2; ++a)
        {
            low[j][a] = std::max(boxes[j].minimum[a], domain.lower[a]);
            high[j][a] = std::min(boxes[j].maximum[a], domain.upper[a]);
        }
        active[j] = high[j][0] > low[j][0] && high[j][1] > low[j][1];
    }
    for (std::size_t round = 0; round <= count; ++round)
    {
        bool changed = false;
        for (std::size_t j = 0; j < count; ++j)
        {
            if (!active[j])
            {
                continue;
            }
            for (int axis : {1, 0})
            {
                for (int side : {0, 1})
                {
                    std::array<double, 2> strip_low = low[j], strip_high = high[j];
                    if (side == 0)
                    {
                        strip_high[axis] = low[j][axis];
                        strip_low[axis] = domain.lower[axis];
                    }
                    else
                    {
                        strip_low[axis] = high[j][axis];
                        strip_high[axis] = domain.upper[axis];
                    }
                    if (strip_high[0] - strip_low[0] <= 1e-7 || strip_high[1] - strip_low[1] <= 1e-7)
                    {
                        continue;
                    }
                    bool overlap = false;
                    for (std::size_t k = 0; k < count && !overlap; ++k)
                    {
                        if (k == j || !active[k])
                        {
                            continue;
                        }
                        overlap = std::min(high[k][0], strip_high[0]) - std::max(low[k][0], strip_low[0]) > 1e-7 &&
                                  std::min(high[k][1], strip_high[1]) - std::max(low[k][1], strip_low[1]) > 1e-7;
                    }
                    if (overlap)
                    {
                        continue;
                    }
                    if (side == 0)
                    {
                        low[j][axis] = strip_low[axis];
                    }
                    else
                    {
                        high[j][axis] = strip_high[axis];
                    }
                    changed = true;
                }
            }
        }
        if (!changed)
        {
            break;
        }
    }
    std::vector<Box3> result(boxes);
    for (std::size_t j = 0; j < count; ++j)
    {
        if (active[j])
        {
            for (int a = 0; a < 2; ++a)
            {
                result[j].minimum[a] = std::min(result[j].minimum[a], low[j][a]);
                result[j].maximum[a] = std::max(result[j].maximum[a], high[j][a]);
            }
        }
    }
    return result;
}

Box3 box_from(const Json& value)
{
    Box3 box;
    for (int k = 0; k < 3; ++k)
    {
        box.minimum[k] = jnum(value.at("minimum")[static_cast<std::size_t>(k)]);
        box.maximum[k] = jnum(value.at("maximum")[static_cast<std::size_t>(k)]);
    }
    return box;
}

Json box_json(const Box3& box)
{
    Json v;
    v["minimum"] = box.minimum;
    v["maximum"] = box.maximum;
    return v;
}

} // namespace

void run_reconcile(const Config& config)
{
    const Timer total;
    const std::vector<std::string> names = pipeline_blocks(config);
    const std::filesystem::path out = config.work / "reconcile";
    std::filesystem::create_directories(out);
    const double spacing = config.footprint_spacing;

    struct Prepared
    {
        std::string name;
        Json metadata;
        std::vector<double> shore_xyz;
        std::vector<std::int64_t> shore_indices;
        std::vector<std::int32_t> obs_view, obs_label;
        std::vector<std::int64_t> obs_anchor;
    };
    std::vector<Prepared> prepared;
    std::map<std::string, Json> catalog;
    for (const std::string& name : names)
    {
        const std::filesystem::path dir = config.work / "detect" / utf8_path(name);
        Prepared p;
        p.name = name;
        p.metadata = read_json_file(dir / "components.json");
        const NpzReader reader(dir / "components.npz");
        p.shore_xyz = npz_get<double>(reader, "shore_xyz");
        p.shore_indices = npz_get<std::int64_t>(reader, "shore_indices");
        p.obs_view = npz_get<std::int32_t>(reader, "obs_view");
        p.obs_label = npz_get<std::int32_t>(reader, "obs_label");
        p.obs_anchor = npz_get<std::int64_t>(reader, "obs_anchor");
        if (!p.metadata.at("view_keys").empty())
        {
            // 视图目录：有水块在 detect 里用的视图（work/scene）。
            const SceneBlock* scene_block = load_scene(config).block(name);
            if (!scene_block)
            {
                fail_value("Block is not present in work/scene: " + name);
            }
            for (const View& view : scene_views(config, scene_block->view_keys))
            {
                catalog.emplace(view.mask_key, view_record(view));
            }
        }
        prepared.push_back(std::move(p));
    }

    // _identities：节点 = (掩码, 分量标签)；共享 >= 3 个锚点的节点相连；锚点归给票数唯一最多（>= 3）的连通组。
    std::map<std::pair<std::string, int>, std::size_t> node_index;
    std::vector<std::size_t> offsets(prepared.size() + 1, 0);
    for (std::size_t b = 0; b < prepared.size(); ++b)
    {
        offsets[b + 1] = offsets[b] + prepared[b].shore_indices.size();
        const Json& views = prepared[b].metadata.at("view_keys");
        for (std::size_t k = 0; k < prepared[b].obs_view.size(); ++k)
        {
            node_index.emplace(std::make_pair(views.at(static_cast<std::size_t>(prepared[b].obs_view[k])).get<std::string>(),
                                              static_cast<int>(prepared[b].obs_label[k])),
                               0);
        }
    }
    std::vector<std::pair<std::string, int>> nodes;
    for (auto& [key, index] : node_index)
    {
        index = nodes.size();
        nodes.push_back(key);
    }
    const std::size_t anchors = offsets.back();
    std::vector<std::vector<std::size_t>> anchor_nodes(anchors);
    for (std::size_t b = 0; b < prepared.size(); ++b)
    {
        const Json& views = prepared[b].metadata.at("view_keys");
        for (std::size_t k = 0; k < prepared[b].obs_view.size(); ++k)
        {
            const std::size_t node = node_index.at({views.at(static_cast<std::size_t>(prepared[b].obs_view[k])).get<std::string>(),
                                                    static_cast<int>(prepared[b].obs_label[k])});
            anchor_nodes[offsets[b] + static_cast<std::size_t>(prepared[b].obs_anchor[k])].push_back(node);
        }
    }
    std::unordered_map<std::uint64_t, std::uint32_t> shared;
    for (std::vector<std::size_t>& list : anchor_nodes)
    {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        for (std::size_t a = 0; a < list.size(); ++a)
        {
            for (std::size_t c = a + 1; c < list.size(); ++c)
            {
                ++shared[(static_cast<std::uint64_t>(list[a]) << 32) | list[c]];
            }
        }
    }
    UnionFind sets(nodes.size());
    for (const auto& [key, count] : shared)
    {
        if (count >= 3)
        {
            sets.unite(static_cast<std::size_t>(key >> 32), static_cast<std::size_t>(key & 0xffffffffu));
        }
    }
    std::vector<int> group(nodes.size(), -1);
    std::unordered_map<std::size_t, int> root_label;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const std::size_t root = sets.find(i);
        const auto found = root_label.find(root);
        if (found == root_label.end())
        {
            const int label = static_cast<int>(root_label.size());
            root_label.emplace(root, label);
            group[i] = label;
        }
        else
        {
            group[i] = found->second;
        }
    }
    std::vector<std::int32_t> shore_ids(anchors, 0);
    for (std::size_t a = 0; a < anchors; ++a)
    {
        std::map<int, int> votes;
        for (std::size_t node : anchor_nodes[a])
        {
            ++votes[group[node]];
        }
        int best = 0, best_group = -1, ties = 0;
        for (const auto& [g, count] : votes)
        {
            if (count > best)
            {
                best = count;
                best_group = g;
                ties = 1;
            }
            else if (count == best)
            {
                ++ties;
            }
        }
        if (best >= 3 && ties == 1)
        {
            shore_ids[a] = best_group + 1;
        }
    }
    std::map<int, std::map<std::string, std::vector<int>>> signatures;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        signatures[group[i] + 1][nodes[i].first].push_back(nodes[i].second);
    }

    // 水位：每个水体自己的岸线锚点。
    std::vector<double> shore(3 * anchors);
    for (std::size_t b = 0; b < prepared.size(); ++b)
    {
        std::copy(prepared[b].shore_xyz.begin(), prepared[b].shore_xyz.end(), shore.begin() + 3 * offsets[b]);
    }
    std::vector<std::uint8_t> inliers(anchors, 0); // 水位内点（只用于统计）
    Json surfaces = Json::array();
    Json skipped = Json::array();
    std::set<int> ids(shore_ids.begin(), shore_ids.end());
    // 锚点按水体分组一次（O(锚点数)），不再对每个水体扫一遍全部锚点。
    std::map<int, std::vector<std::size_t>> rows_of;
    for (std::size_t a = 0; a < anchors; ++a)
    {
        if (shore_ids[a] > 0)
        {
            rows_of[shore_ids[a]].push_back(a);
        }
    }
    for (int wid : ids)
    {
        if (wid <= 0)
        {
            continue;
        }
        const std::vector<std::size_t>& rows = rows_of[wid];
        std::vector<double> z;
        z.reserve(rows.size());
        for (std::size_t a : rows)
        {
            z.push_back(shore[3 * a + 2]);
        }
        Json item;
        item["water_id"] = wid;
        item["anchors"] = rows.size();
        if (rows.size() < 3)
        {
            item["reason"] = "insufficient_own_shoreline";
            skipped.push_back(item);
            continue;
        }
        const ShoreLevel level = shore_level(z);
        std::vector<double> residual;
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            if (level.inliers[k])
            {
                residual.push_back(std::abs(z[k] - level.height));
            }
        }
        if (residual.size() < 3)
        {
            item["reason"] = "insufficient_consistent_own_shoreline";
            skipped.push_back(item);
            continue;
        }
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            if (level.inliers[k])
            {
                inliers[rows[k]] = 1;
            }
        }
        item["height"] = level.height;
        item["height_sigma"] = 1.4826 * median(residual);
        item["shore_inliers"] = residual.size();
        Json signature = Json::object();
        for (const auto& [key, labels] : signatures[wid])
        {
            signature[key] = labels;
        }
        item["signature"] = signature;
        surfaces.push_back(item);
    }

    // 块范围、全局格网与每块的岸线分片。
    Domain domain = load_scene(config).domain;
    {
        // 瓦片范围：LOD 中每块最细层叶节点的 XY 外包框。
        const LodIndex lod = read_lod_index(config.lod);
        for (const LodBlock& block : lod.blocks)
        {
            Box3 box;
            for (const LodLeaf& leaf : block.leaves)
            {
                box.expand(leaf.box);
            }
            if (box.valid())
            {
                domain.tiles.push_back({box.minimum[0], box.minimum[1], box.maximum[0], box.maximum[1]});
            }
        }
        log("[RECONCILE] fill domain: bbox, %zu LOD tiles, ROI polygon %s", domain.tiles.size(),
            domain.polygon.empty() ? "none" : (std::to_string(domain.polygon.size()) + " vertices").c_str());
    }
    std::vector<Box3> boxes;
    for (const Prepared& p : prepared)
    {
        boxes.push_back(box_from(p.metadata.at("bounds")));
    }
    const std::vector<Box3> extended = extend_bounds(boxes, domain);
    Json blocks = Json::array();
    for (std::size_t b = 0; b < prepared.size(); ++b)
    {
        Json record;
        record["block"] = prepared[b].name;
        record["water"] = prepared[b].metadata.at("water");
        record["bounds"] = box_json(extended[b]);
        blocks.push_back(record);
    }
    std::array<double, 2> origin{}, shape{};
    for (int a = 0; a < 2; ++a)
    {
        origin[a] = std::floor(domain.lower[a] / spacing) * spacing;
        shape[a] = std::ceil((domain.upper[a] - origin[a]) / spacing) + 1;
    }
    Json views = Json::array();
    for (const auto& [key, record] : catalog)
    {
        views.push_back(record);
    }
    write_json_atomic(out / "view_catalog.json", views, true);
    Json report;
    report["schema"] = "ai3d-waterpost.reconcile/1";
    report["blocks"] = blocks;
    report["surfaces"] = surfaces;
    report["skipped_surfaces"] = skipped;
    Json grid;
    grid["origin"] = origin;
    grid["shape"] = {static_cast<long long>(shape[0]), static_cast<long long>(shape[1])};
    grid["spacing"] = spacing;
    report["global_grid"] = grid;
    report["domain"] = domain.json();
    report["identity_nodes"] = nodes.size();
    report["anchors"] = anchors;
    report["seconds"] = total.seconds();
    write_json_atomic(out / "report.json", report, false);
    log("[RECONCILE] %zu blocks, %zu shoreline anchors, %zu surfaces, %zu skipped (%.1fs)", prepared.size(), anchors,
        surfaces.size(), skipped.size(), total.seconds());
}

} // namespace ai3d::waterpost
