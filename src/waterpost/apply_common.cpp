#include "apply_common.hpp"

#include "error.hpp"
#include "fs_util.hpp"
#include "npz_util.hpp"
#include "scene.hpp"

namespace ai3d::waterpost
{

int Report::block_index(const std::string& name) const
{
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        if (blocks[i].name == name)
        {
            return static_cast<int>(i);
        }
    }
    fail_value("Unknown block in the reconciled report: " + name);
}

const BlockRecord& Report::block(const std::string& name) const
{
    return blocks[static_cast<std::size_t>(block_index(name))];
}

Report load_report(const Config& config)
{
    Report report;
    report.json = read_json_file(config.work / "reconcile" / "report.json");
    const Json& grid = report.json.at("global_grid");
    for (int a = 0; a < 2; ++a)
    {
        report.grid.origin[a] = jnum(grid.at("origin")[static_cast<std::size_t>(a)]);
        report.grid.shape[a] = jint(grid.at("shape")[static_cast<std::size_t>(a)]);
    }
    report.grid.spacing = jnum(grid.at("spacing"));
    report.domain = Domain::from_json(report.json.at("domain"));
    for (const Json& item : report.json.at("blocks"))
    {
        BlockRecord record;
        record.name = item.at("block").get<std::string>();
        for (int k = 0; k < 3; ++k)
        {
            record.bounds.minimum[k] = jnum(item.at("bounds").at("minimum")[static_cast<std::size_t>(k)]);
            record.bounds.maximum[k] = jnum(item.at("bounds").at("maximum")[static_cast<std::size_t>(k)]);
        }
        record.water = jint(item.at("water"));
        report.blocks.push_back(std::move(record));
    }
    for (const Json& item : report.json.at("surfaces"))
    {
        Surface surface;
        surface.water_id = static_cast<int>(jint(item.at("water_id")));
        surface.height = jnum(item.at("height"));
        surface.sigma = item.contains("height_sigma") ? jnum(item.at("height_sigma")) : report.grid.spacing;
        for (auto it = item.at("signature").begin(); it != item.at("signature").end(); ++it)
        {
            surface.signature[it.key()] = it.value().get<std::vector<int>>();
        }
        report.surfaces.push_back(std::move(surface));
    }
    return report;
}

std::vector<View> load_catalog(const Config& config)
{
    std::vector<View> views;
    for (const Json& record : read_json_file(config.work / "reconcile" / "view_catalog.json"))
    {
        views.push_back(view_from_record(record));
    }
    return views;
}

std::filesystem::path detect_dir(const Config& config, const std::string& block)
{
    return config.work / "detect" / utf8_path(block);
}

bool load_land_grid(const Config& config, const std::string& block, LandGrid& grid)
{
    const std::filesystem::path path = detect_dir(config, block) / "land_grid.npz";
    if (!std::filesystem::is_regular_file(path))
    {
        return false;
    }
    const NpzReader reader(path);
    grid.cells = npz_get<std::int64_t>(reader, "land_cells");
    grid.zmin = npz_get<float>(reader, "land_zmin");
    grid.zmax = npz_get<float>(reader, "land_zmax");
    grid.count = npz_get<std::int32_t>(reader, "land_count");
    return true;
}

std::vector<double> load_seed_xyz(const Config& config, const std::string& block)
{
    const NpzReader reader(detect_dir(config, block) / "seeds.npz");
    return npz_get<double>(reader, "water_xyz");
}

void load_occupancy(const Config& config, const std::string& block, std::unordered_map<std::string, Occupancy>& out)
{
    // 有水块在 detect 里用过的视图（components.json 的 view_keys）；占据格网来自 work/scene，没有 scene 时来自本块 detect。
    const std::filesystem::path path = detect_dir(config, block) / "components.json";
    if (!std::filesystem::is_regular_file(path))
    {
        return;
    }
    const Json metadata = read_json_file(path);
    if (metadata.value("no_water", false))
    {
        return;
    }
    if (!has_scene(config))
    {
        // blocks 模式直接从 detect 开始：占据格网在 detect/<块>/occupancy.npz（顺序同 view_keys）。
        const NpzReader reader(detect_dir(config, block) / "occupancy.npz");
        const std::vector<std::uint8_t> blob = npz_get<std::uint8_t>(reader, "occupancy_blob");
        const std::vector<std::int64_t> offsets = npz_get<std::int64_t>(reader, "occupancy_offsets");
        const std::vector<std::int64_t> bytes = npz_get<std::int64_t>(reader, "occupancy_bytes");
        const std::vector<std::int32_t> rows = npz_get<std::int32_t>(reader, "occupancy_rows");
        const std::vector<std::int32_t> cols = npz_get<std::int32_t>(reader, "occupancy_cols");
        const std::vector<std::int64_t> water = npz_get<std::int64_t>(reader, "occupancy_water_pixels");
        const Json& keys = metadata.at("view_keys");
        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            const std::string key = keys[i].get<std::string>();
            if (out.count(key))
            {
                continue;
            }
            Occupancy o;
            o.rows = rows[i];
            o.cols = cols[i];
            o.water_pixels = water[i];
            o.packed.assign(blob.begin() + offsets[i], blob.begin() + offsets[i] + bytes[i]);
            out.emplace(key, std::move(o));
        }
        return;
    }
    const std::unordered_map<std::string, Occupancy>& all = scene_occupancy(config);
    for (const Json& item : metadata.at("view_keys"))
    {
        const std::string key = item.get<std::string>();
        if (out.count(key))
        {
            continue;
        }
        const auto found = all.find(key);
        if (found == all.end())
        {
            fail_value("View of block " + block + " is not in work/scene: " + key);
        }
        out.emplace(key, found->second);
    }
}

Candidates load_candidates(const Config& config, const std::string& block)
{
    const std::filesystem::path dir = config.work / "apply" / "candidates" / utf8_path(block);
    Candidates c;
    const NpzReader reader(dir / "candidates.npz");
    c.cells = npz_get<std::int64_t>(reader, "cells");
    c.ids = npz_get<std::int32_t>(reader, "water_ids");
    c.support = npz_get<std::uint8_t>(reader, "support");
    c.metadata = read_json_file(dir / "metadata.json");
    return c;
}

} // namespace ai3d::waterpost
