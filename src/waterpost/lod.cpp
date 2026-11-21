#include "lod.hpp"

#include "error.hpp"
#include "fs_util.hpp"
#include "parallel.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <regex>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>

namespace ai3d::waterpost
{
namespace
{

struct LodIndexCache
{
    std::mutex mutex;
    bool enabled = false;
    std::map<std::string, LodIndex> indices;
};

LodIndexCache g_index_cache;

bool ends_with(const std::string& text, const std::string& suffix)
{
    if (text.size() < suffix.size())
    {
        return false;
    }
    std::string tail = text.substr(text.size() - suffix.size());
    for (char& c : tail)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return tail == suffix;
}

std::string percent_decode(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '%' && i + 2 < value.size())
        {
            const int hi = std::isxdigit(static_cast<unsigned char>(value[i + 1])) ? std::stoi(value.substr(i + 1, 2), nullptr, 16) : -1;
            if (hi >= 0)
            {
                out.push_back(static_cast<char>(hi));
                i += 2;
                continue;
            }
        }
        out.push_back(value[i]);
    }
    return out;
}

std::vector<std::string> split_posix(const std::string& path)
{
    std::vector<std::string> parts;
    std::string current;
    for (char c : path)
    {
        if (c == '/' || c == '\\')
        {
            if (!current.empty())
            {
                parts.push_back(current);
            }
            current.clear();
        }
        else
        {
            current.push_back(c);
        }
    }
    if (!current.empty())
    {
        parts.push_back(current);
    }
    return parts;
}

std::string resolve_uri(const std::string& base, const std::string& uri)
{
    const std::string decoded = percent_decode(uri);
    return normalize_posix(base.empty() ? decoded : base + "/" + decoded);
}

Box3 box_from_json(const Json& node)
{
    Box3 box;
    if (!node.contains("boundingVolume") || !node["boundingVolume"].contains("box"))
    {
        return box;
    }
    const Json& b = node["boundingVolume"]["box"];
    if (!b.is_array() || b.size() != 12)
    {
        return box;
    }
    double v[12];
    for (int i = 0; i < 12; ++i)
    {
        v[i] = jnum(b[static_cast<std::size_t>(i)]);
    }
    for (int j = 0; j < 3; ++j)
    {
        const double half = std::abs(v[3 + j]) + std::abs(v[6 + j]) + std::abs(v[9 + j]);
        box.minimum[j] = v[j] - half;
        box.maximum[j] = v[j] + half;
    }
    return box;
}

Json box_to_json(const Box3& box)
{
    Json out = Json::array();
    const double c[3] = {(box.minimum[0] + box.maximum[0]) * 0.5, (box.minimum[1] + box.maximum[1]) * 0.5,
                         (box.minimum[2] + box.maximum[2]) * 0.5};
    const double h[3] = {(box.maximum[0] - box.minimum[0]) * 0.5, (box.maximum[1] - box.minimum[1]) * 0.5,
                         (box.maximum[2] - box.minimum[2]) * 0.5};
    out.push_back(c[0]);
    out.push_back(c[1]);
    out.push_back(c[2]);
    out.push_back(h[0]);
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(h[1]);
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(h[2]);
    return out;
}

bool box_contains(const Box3& outer, const Box3& inner)
{
    if (!inner.valid())
    {
        return true;
    }
    if (!outer.valid())
    {
        return false;
    }
    for (int k = 0; k < 3; ++k)
    {
        // 容许浮点往返误差。
        const double tolerance = 1e-6 * std::max(1.0, std::abs(outer.maximum[k]) + std::abs(outer.minimum[k]));
        if (inner.minimum[k] < outer.minimum[k] - tolerance || inner.maximum[k] > outer.maximum[k] + tolerance)
        {
            return false;
        }
    }
    return true;
}

int level_of(const Json& node, const std::vector<std::string>& parts)
{
    if (node.contains("extras") && node["extras"].is_object() && node["extras"].contains("lodLevel"))
    {
        return static_cast<int>(jint(node["extras"]["lodLevel"]));
    }
    if (!parts.empty() && parts[0].size() > 1 && (parts[0][0] == 'L' || parts[0][0] == 'l'))
    {
        try
        {
            return std::stoi(parts[0].substr(1));
        }
        catch (...)
        {
        }
    }
    return 0;
}

bool is_padding(const std::uint8_t* r)
{
    for (int i = 0; i < 26; ++i)
    {
        if (r[i])
        {
            return false;
        }
    }
    return r[28] == 0 && r[29] == 0 && r[30] == 0 && r[31] == 0;
}

std::uint32_t read_u32(const std::uint8_t* p)
{
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

float read_f32(const std::uint8_t* p)
{
    float v;
    std::memcpy(&v, p, 4);
    return v;
}

std::uint16_t read_u16(const std::uint8_t* p)
{
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

void put_f32(std::uint8_t* p, float v)
{
    std::memcpy(p, &v, 4);
}

void put_u16(std::uint8_t* p, std::uint16_t v)
{
    std::memcpy(p, &v, 2);
}

std::uint8_t to_u8(double v)
{
    const double scaled = std::min(255.0, std::max(0.0, v * 255.0));
    return static_cast<std::uint8_t>(scaled);
}

} // namespace

std::string normalize_posix(const std::string& path)
{
    std::vector<std::string> out;
    for (const std::string& part : split_posix(path))
    {
        if (part == ".")
        {
            continue;
        }
        if (part == ".." && !out.empty() && out.back() != "..")
        {
            out.pop_back();
            continue;
        }
        out.push_back(part);
    }
    std::string result;
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        if (i)
        {
            result.push_back('/');
        }
        result += out[i];
    }
    return result;
}

std::string dirname_posix(const std::string& path)
{
    const std::size_t at = path.find_last_of('/');
    return at == std::string::npos ? std::string() : path.substr(0, at);
}

const LodBlock* LodIndex::block(const std::string& name) const
{
    for (const LodBlock& item : blocks)
    {
        if (item.name == name)
        {
            return &item;
        }
    }
    return nullptr;
}

bool read_lod_root_transform(const std::filesystem::path& root, std::array<double, 16>& transform)
{
    for (const char* name : {"tileset.source.json", "tileset.json"})
    {
        const std::filesystem::path path = root / name;
        if (!std::filesystem::is_regular_file(path))
        {
            continue;
        }
        const Json tileset = read_json_file(path);
        const Json& node = tileset.at("root");
        if (node.contains("transform") && node["transform"].is_array() && node["transform"].size() == 16)
        {
            for (std::size_t k = 0; k < 16; ++k)
            {
                transform[k] = jnum(node["transform"][k]);
            }
            return true;
        }
    }
    return false;
}

static LodIndex read_lod_index_uncached(const std::filesystem::path& root)
{
    LodIndex index;
    index.root = root;
    const std::filesystem::path top = root / "tileset.json";
    if (!std::filesystem::is_regular_file(top))
    {
        fail_missing("LOD tileset.json not found: " + path_utf8(top));
    }
    const Json tileset = read_json_file(top);
    if (tileset.contains("asset") && tileset["asset"].contains("extras"))
    {
        const Json& extras = tileset["asset"]["extras"];
        index.has_fix = extras.contains("ai3dLodFix");
        index.semantic = extras.contains("ai3dSemanticSplat");
    }
    index.source_entry = std::filesystem::is_regular_file(root / "tileset.source.json") ? "tileset.source.json" : "tileset.json";
    const Json entry = index.source_entry == "tileset.json" ? tileset : read_json_file(root / "tileset.source.json");

    std::map<std::string, LodBlock> blocks;
    std::set<std::string> visited;
    std::mutex index_mutex;
    std::atomic<std::size_t> manifests_read{0};
    std::function<void(const Json&, const std::string&)> walk = [&](const Json& node, const std::string& base) {
        const bool has_children = node.contains("children") && node["children"].is_array() && !node["children"].empty();
        if (node.contains("content") && node["content"].contains("uri"))
        {
            const std::string resolved = resolve_uri(base, node["content"]["uri"].get<std::string>());
            if (ends_with(resolved, ".json"))
            {
                bool first;
                {
                    std::lock_guard<std::mutex> lock(index_mutex);
                    first = visited.insert(resolved).second;
                }
                if (first)
                {
                    const Json sub = read_json_file(root / utf8_path(resolved));
                    const auto count = ++manifests_read;
                    if (count % 5000 == 0)
                    {
                        std::printf("[LOD] read %zu external manifests\n", count);
                        std::fflush(stdout);
                    }
                    walk(sub.at("root"), dirname_posix(resolved));
                }
            }
            else if (ends_with(resolved, ".bin") && !has_children)
            {
                const std::vector<std::string> parts = split_posix(resolved);
                if (parts.size() >= 3 && parts[0] != "fix")
                {
                    LodLeaf leaf;
                    leaf.uri = resolved;
                    leaf.level = level_of(node, parts);
                    leaf.box = box_from_json(node);
                    std::lock_guard<std::mutex> lock(index_mutex);
                    LodBlock& block = blocks[parts[parts.size() - 2]];
                    block.name = parts[parts.size() - 2];
                    block.leaves.push_back(std::move(leaf));
                }
            }
            else if (!ends_with(resolved, ".bin"))
            {
                fail_value("Only BIN LOD content is supported: " + resolved);
            }
        }
        if (has_children)
        {
            const Json& children = node["children"];
            parallel_for_each(children.size(), [&](std::size_t i) { walk(children[i], base); });
        }
    };
    walk(entry.at("root"), "");
    for (auto& [name, block] : blocks)
    {
        int level = 0;
        for (const LodLeaf& leaf : block.leaves)
        {
            level = std::max(level, leaf.level);
        }
        std::vector<LodLeaf> finest;
        std::size_t dropped = 0;
        for (LodLeaf& leaf : block.leaves)
        {
            if (leaf.level == level)
            {
                finest.push_back(std::move(leaf));
            }
            else
            {
                ++dropped;
            }
        }
        if (dropped)
        {
            std::printf("[LOD] block %s: %zu leaves above the finest level L%d are left unchanged\n", name.c_str(),
                        dropped, level);
        }
        std::sort(finest.begin(), finest.end(), [](const LodLeaf& a, const LodLeaf& b) { return a.uri < b.uri; });
        block.level = level;
        block.leaves = std::move(finest);
        index.blocks.push_back(std::move(block));
    }
    if (index.blocks.empty())
    {
        fail_value("LOD has no block contents: " + path_utf8(root));
    }
    return index;
}

void enable_lod_index_snapshot_cache()
{
    std::lock_guard<std::mutex> lock(g_index_cache.mutex);
    g_index_cache.indices.clear();
    g_index_cache.enabled = true;
}

LodIndex read_lod_index(const std::filesystem::path& root)
{
    std::unique_lock<std::mutex> lock(g_index_cache.mutex);
    if (!g_index_cache.enabled)
    {
        lock.unlock();
        return read_lod_index_uncached(root);
    }
    const std::string key = path_utf8(std::filesystem::absolute(root).lexically_normal());
    const auto cached = g_index_cache.indices.find(key);
    if (cached != g_index_cache.indices.end())
    {
        return cached->second;
    }
    LodIndex index = read_lod_index_uncached(root);
    g_index_cache.indices.emplace(key, index);
    return index;
}

LeafContent read_leaf(const std::filesystem::path& path)
{
    const std::vector<std::uint8_t> bytes = read_file_bytes(path);
    if (bytes.size() < GSSP_HEADER || std::memcmp(bytes.data(), "gssp", 4) != 0)
    {
        fail_value("Not a gssp BIN: " + path_utf8(path));
    }
    LeafContent leaf;
    leaf.padded = read_u32(bytes.data() + 4);
    if (bytes.size() != GSSP_HEADER + std::size_t{leaf.padded} * GSSP_RECORD)
    {
        fail_value("Truncated gssp BIN: " + path_utf8(path));
    }
    std::uint32_t count = leaf.padded;
    const std::uint8_t* records = bytes.data() + GSSP_HEADER;
    while (count > 0 && is_padding(records + std::size_t{count - 1} * GSSP_RECORD))
    {
        --count;
    }
    leaf.count = count;
    for (std::uint32_t i = 0; i < leaf.padded && !leaf.semantic; ++i)
    {
        const std::uint8_t* r = records + std::size_t{i} * GSSP_RECORD;
        leaf.semantic = r[26] != 0 || r[27] != 0;
    }
    leaf.records.assign(records, records + std::size_t{count} * GSSP_RECORD);
    return leaf;
}

std::uint32_t padded_leaf_count(std::uint32_t count)
{
    // A zero-page GSplat content has no _gsplatData in ai3d-studio. Its
    // REPLACE parent then remains the cover and suppresses ready siblings,
    // including the new water surface. Transparent padding is already part
    // of the GSSP format and keeps the real record count at zero on readback.
    if (count == 0)
    {
        return GSSP_PAD;
    }
    return count % GSSP_PAD ? count + GSSP_PAD - count % GSSP_PAD : count;
}

void write_leaf(const std::filesystem::path& path, const std::vector<std::uint8_t>& records, std::uint32_t count,
                const Box3& box, bool semantic)
{
    const std::uint32_t padded = padded_leaf_count(count);
    std::vector<std::uint8_t> out(GSSP_HEADER + std::size_t{padded} * GSSP_RECORD, 0);
    std::memcpy(out.data(), "gssp", 4);
    std::memcpy(out.data() + 4, &padded, 4);
    for (int k = 0; k < 3; ++k)
    {
        put_f32(out.data() + 8 + 4 * k, static_cast<float>(box.minimum[k]));
        put_f32(out.data() + 20 + 4 * k, static_cast<float>(box.maximum[k]));
    }
    if (count != 0)
    {
        std::memcpy(out.data() + GSSP_HEADER, records.data(), std::size_t{count} * GSSP_RECORD);
    }
    if (semantic)
    {
        for (std::uint32_t i = count; i < padded; ++i)
        {
            out[GSSP_HEADER + std::size_t{i} * GSSP_RECORD + 26] = 255;
        }
    }
    std::filesystem::create_directories(path.parent_path());
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    write_file_atomic(path, out.data(), out.size(), temporary);
}

std::uint32_t read_leaf_padded(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    char header[8] = {};
    stream.read(header, 8);
    if (!stream || std::memcmp(header, "gssp", 4) != 0)
    {
        fail_value("Not a gssp BIN: " + path_utf8(path));
    }
    std::uint32_t padded;
    std::memcpy(&padded, header + 4, 4);
    return padded;
}

std::string relative_posix(const std::string& from_dir, const std::string& to)
{
    const std::vector<std::string> a = split_posix(normalize_posix(from_dir)), b = split_posix(normalize_posix(to));
    std::size_t common = 0;
    while (common < a.size() && common + 1 < b.size() && a[common] == b[common])
    {
        ++common;
    }
    std::string out;
    for (std::size_t i = common; i < a.size(); ++i)
    {
        out += "../";
    }
    for (std::size_t i = common; i < b.size(); ++i)
    {
        out += b[i];
        if (i + 1 < b.size())
        {
            out += "/";
        }
    }
    return out;
}

float half_to_float(std::uint16_t bits)
{
    const std::uint32_t sign = (bits >> 15) & 1u;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    float value;
    if (exponent == 0)
    {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    }
    else if (exponent == 31)
    {
        value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    }
    else
    {
        value = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

std::uint16_t float_to_half(float value)
{
    std::uint32_t x;
    std::memcpy(&x, &value, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t magnitude = x & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u));
    }
    if (magnitude >= 0x477ff000u) // >= 65520 rounds to infinity: clamp to the largest finite half
    {
        return static_cast<std::uint16_t>(sign | 0x7bffu);
    }
    if (magnitude < 0x38800000u) // subnormal half
    {
        if (magnitude < 0x33000000u)
        {
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t shift = 126u - (magnitude >> 23);
        const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        // 次正规半精度：值 = half * 2^-24，float 值 = mantissa * 2^(e - 150)，所以 half = mantissa >> (126 - e)。
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (half & 1u)))
        {
            ++half;
        }
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = ((magnitude - 0x38000000u) >> 13);
    const std::uint32_t remainder = magnitude & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u)))
    {
        ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
}

void encode_water_record(std::uint8_t* out, const Vec3& position, const Vec3& scale, const std::array<double, 3>& rgb,
                         double opacity, bool semantic, int water_class)
{
    std::memset(out, 0, GSSP_RECORD);
    for (int k = 0; k < 3; ++k)
    {
        put_f32(out + 4 * k, static_cast<float>(position[k]));
    }
    const float xx = static_cast<float>(scale[0] * scale[0]);
    const float yy = static_cast<float>(scale[1] * scale[1]);
    const float zz = static_cast<float>(scale[2] * scale[2]);
    put_f32(out + 12, yy);
    put_u16(out + 16, float_to_half(xx));
    put_u16(out + 18, float_to_half(0.0f));
    put_u16(out + 20, float_to_half(0.0f));
    put_u16(out + 22, float_to_half(0.0f));
    put_u16(out + 24, float_to_half(zz));
    if (semantic)
    {
        out[26] = static_cast<std::uint8_t>(water_class);
        out[27] = 255;
    }
    out[28] = to_u8(rgb[0]);
    out[29] = to_u8(rgb[1]);
    out[30] = to_u8(rgb[2]);
    out[31] = to_u8(opacity);
}

Gaussians read_block_gaussians(const LodIndex& index, const LodBlock& block, std::vector<std::size_t>* leaf_offsets)
{
    std::vector<LeafContent> leaves(block.leaves.size());
    parallel_for_each(leaves.size(), [&](std::size_t i) {
        leaves[i] = read_leaf(index.root / utf8_path(block.leaves[i].uri));
    });
    std::vector<std::size_t> offsets(leaves.size() + 1, 0);
    for (std::size_t i = 0; i < leaves.size(); ++i)
    {
        offsets[i + 1] = offsets[i] + leaves[i].count;
    }
    Gaussians g;
    g.count = offsets.back();
    g.xyz.resize(3 * g.count);
    g.cov.resize(6 * g.count);
    g.opacity.resize(g.count);
    g.rgba.resize(4 * g.count);
    g.sigma_max.resize(g.count);
    g.normal_z.resize(g.count);
    g.semantic.resize(2 * g.count);
    parallel_for_each(leaves.size(), [&](std::size_t leaf) {
        const LeafContent& content = leaves[leaf];
        for (std::uint32_t r = 0; r < content.count; ++r)
        {
            const std::size_t i = offsets[leaf] + r;
            const std::uint8_t* p = content.records.data() + std::size_t{r} * GSSP_RECORD;
            for (int k = 0; k < 3; ++k)
            {
                g.xyz[3 * i + k] = static_cast<double>(read_f32(p + 4 * k));
            }
            const float yy = read_f32(p + 12);
            const float xx = half_to_float(read_u16(p + 16));
            const float xy = half_to_float(read_u16(p + 18));
            const float xz = half_to_float(read_u16(p + 20));
            const float yz = half_to_float(read_u16(p + 22));
            const float zz = half_to_float(read_u16(p + 24));
            float* c = g.cov.data() + 6 * i;
            c[0] = xx;
            c[1] = xy;
            c[2] = xz;
            c[3] = yy;
            c[4] = yz;
            c[5] = zz;
            g.semantic[2 * i] = p[26];
            g.semantic[2 * i + 1] = p[27];
            for (int k = 0; k < 4; ++k)
            {
                g.rgba[4 * i + k] = p[28 + k];
            }
            g.opacity[i] = std::min(1.0f, (static_cast<float>(p[31]) + 0.5f) / 255.0f);
            Eigen::Matrix3d m;
            m << xx, xy, xz, xy, yy, yz, xz, yz, zz;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
            solver.computeDirect(m);
            const Eigen::Vector3d values = solver.eigenvalues();
            g.sigma_max[i] = static_cast<float>(std::sqrt(std::max(0.0, values[2])));
            g.normal_z[i] = static_cast<float>(std::abs(solver.eigenvectors()(2, 0)));
        }
    });
    for (std::size_t i = 0; i < 3 * g.count; ++i)
    {
        if (!std::isfinite(g.xyz[i]))
        {
            fail_value("Nonfinite Gaussian position in LOD block " + block.name);
        }
    }
    if (leaf_offsets)
    {
        *leaf_offsets = std::move(offsets);
    }
    return g;
}

void finalize_lod(const std::filesystem::path& input, const std::filesystem::path& output,
                  const std::vector<LeafChange>& changes, bool hard_link)
{
    namespace fs = std::filesystem;
    if (fs::exists(output) && fs::equivalent(input, output))
    {
        fail_value("Output LOD must differ from the input LOD");
    }
    std::unordered_map<std::string, const LeafChange*> by_uri;
    std::unordered_map<std::string, std::vector<const LeafChange*>> siblings;
    for (const LeafChange& change : changes)
    {
        by_uri[change.uri] = &change;
        if (!change.sibling_of.empty())
        {
            siblings[change.sibling_of].push_back(&change);
        }
    }
    // 0) 上一次写到同一 output 的溢出兄弟节点（<叶>_water<k>.bin / .json）不在本次结果里时删掉，
    //    免得留下没人引用的旧文件。只动这种命名的文件，不碰 output 里的其他内容。
    if (fs::is_directory(output))
    {
        static const std::regex water_sibling(R"(.*_water\d+\.(bin|json)$)");
        std::vector<fs::path> stale;
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(output))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const fs::path relative = entry.path().lexically_relative(output);
            const std::string key = relative.generic_u8string();
            if (!std::regex_match(key, water_sibling) || by_uri.count(key) || fs::exists(input / relative))
            {
                continue;
            }
            stale.push_back(entry.path());
        }
        for (const fs::path& path : stale)
        {
            std::error_code error;
            fs::remove(path, error);
        }
        if (!stale.empty())
        {
            std::printf("[LOD] removed %zu stale water sibling files from an earlier run\n", stale.size());
        }
    }
    // 1) 复制未改写的文件。目标已存在且大小与修改时间都与输入相同才跳过（复制时会把修改时间设成输入的），
    //    上一次 write 改写过、这一次没改写的叶节点因此会被换回输入内容。
    fs::create_directories(output);
    std::vector<std::pair<fs::path, fs::path>> copies;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(input))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        // The iterator already supplies a path under input. Avoid resolving
        // every ancestor through remote FUSE metadata for each file.
        const fs::path relative = entry.path().lexically_relative(input);
        const std::string key = relative.generic_u8string();
        if (by_uri.count(key))
        {
            continue;
        }
        const fs::path target = output / relative;
        std::error_code error;
        if (fs::is_regular_file(target, error) && fs::file_size(target, error) == entry.file_size() &&
            fs::last_write_time(target, error) == entry.last_write_time() && !error && !ends_with(key, ".json"))
        {
            continue;
        }
        copies.emplace_back(entry.path(), target);
    }
    parallel_for_each(copies.size(), [&](std::size_t i) {
        const auto& [source, target] = copies[i];
        std::error_code error;
        fs::create_directories(target.parent_path(), error);
        fs::remove(target, error);
        if (hard_link && !ends_with(path_utf8(target), ".json"))
        {
            fs::create_hard_link(source, target, error);
            if (!error)
            {
                return;
            }
        }
        // Only bytes are needed here. std::filesystem::copy_file can fail with
        // EPERM on writable OSS FUSE paths because it also sets file modes.
        copy_file_contents(source, target);
        fs::last_write_time(target, fs::last_write_time(source, error), error);
    });
    for (const LeafChange& change : changes)
    {
        if (!fs::is_regular_file(output / utf8_path(change.uri)))
        {
            fail_missing("Rewritten LOD content is missing from the output: " + change.uri);
        }
    }
    // 2) 改清单：叶节点换包围盒与点数，祖先取并集；外部清单按引用递归。
    std::unordered_map<std::string, Box3> patched;
    std::size_t written = 0;
    std::function<Box3(const std::string&)> patch_file;
    std::function<Box3(Json&, const std::string&, bool&)> patch_node = [&](Json& node, const std::string& base, bool& dirty) {
        Box3 box = box_from_json(node);
        Box3 grown = box;
        if (node.contains("content") && node["content"].contains("uri"))
        {
            const std::string resolved = resolve_uri(base, node["content"]["uri"].get<std::string>());
            if (ends_with(resolved, ".json"))
            {
                grown.expand(patch_file(resolved));
            }
            else
            {
                const auto found = by_uri.find(resolved);
                if (found != by_uri.end())
                {
                    grown.expand(found->second->box);
                    if (node.contains("extras") && node["extras"].is_object() &&
                        node["extras"].contains("gsplatPointCount"))
                    {
                        node["extras"]["gsplatPointCount"] = found->second->padded;
                        dirty = true;
                    }
                }
            }
        }
        if (node.contains("children") && node["children"].is_array())
        {
            // 装不下的水面点：在同一父节点下追加兄弟叶节点（REPLACE 时与原叶一起显示）。
            Json additions = Json::array();
            for (const Json& child : node["children"])
            {
                if (!child.contains("content") || !child["content"].contains("uri"))
                {
                    continue;
                }
                const std::string resolved = resolve_uri(base, child["content"]["uri"].get<std::string>());
                if (ends_with(resolved, ".bin"))
                {
                    const auto found = siblings.find(resolved);
                    if (found == siblings.end())
                    {
                        continue;
                    }
                    for (const LeafChange* extra : found->second)
                    {
                        Json copy = child;
                        copy.erase("children");
                        copy["content"]["uri"] = relative_posix(base, extra->uri);
                        copy["boundingVolume"]["box"] = box_to_json(extra->box);
                        if (copy.contains("extras") && copy["extras"].is_object())
                        {
                            copy["extras"]["gsplatPointCount"] = extra->padded;
                        }
                        additions.push_back(copy);
                    }
                }
                else if (ends_with(resolved, ".json"))
                {
                    const fs::path child_path = output / utf8_path(resolved);
                    const Json sub = read_json_file(child_path);
                    if (!sub.contains("root") || !sub["root"].contains("content"))
                    {
                        continue;
                    }
                    const std::string dir = dirname_posix(resolved);
                    const std::string leaf = resolve_uri(dir, sub["root"]["content"]["uri"].get<std::string>());
                    const auto found = siblings.find(leaf);
                    if (found == siblings.end())
                    {
                        continue;
                    }
                    for (const LeafChange* extra : found->second)
                    {
                        std::string stem = extra->uri.substr(extra->uri.find_last_of('/') + 1);
                        stem = stem.substr(0, stem.size() - 4);
                        const std::string json_rel = (dir.empty() ? std::string() : dir + "/") + stem + ".json";
                        Json doc = sub;
                        doc["root"].erase("children");
                        doc["root"]["content"]["uri"] = relative_posix(dir, extra->uri);
                        doc["root"]["boundingVolume"]["box"] = box_to_json(extra->box);
                        if (doc["root"].contains("extras") && doc["root"]["extras"].is_object() &&
                            doc["root"]["extras"].contains("gsplatPointCount"))
                        {
                            doc["root"]["extras"]["gsplatPointCount"] = extra->padded;
                        }
                        write_json_atomic(output / utf8_path(json_rel), doc, true);
                        Json copy = child;
                        copy["content"]["uri"] = relative_posix(base, json_rel);
                        copy["boundingVolume"]["box"] = box_to_json(extra->box);
                        additions.push_back(copy);
                    }
                }
            }
            for (Json& addition : additions)
            {
                node["children"].push_back(std::move(addition));
                dirty = true;
            }
            for (Json& child : node["children"])
            {
                grown.expand(patch_node(child, base, dirty));
            }
        }
        if (!box_contains(box, grown))
        {
            node["boundingVolume"]["box"] = box_to_json(grown);
            dirty = true;
            return grown;
        }
        return box;
    };
    patch_file = [&](const std::string& relative) -> Box3 {
        const auto found = patched.find(relative);
        if (found != patched.end())
        {
            return found->second;
        }
        const fs::path path = output / utf8_path(relative);
        Json document = read_json_file(path);
        bool dirty = false;
        const Box3 box = patch_node(document.at("root"), dirname_posix(relative), dirty);
        if (dirty)
        {
            write_json_atomic(path, document, true);
            ++written;
        }
        patched[relative] = box;
        return box;
    };
    patch_file("tileset.json");
    if (fs::is_regular_file(output / "tileset.source.json"))
    {
        patch_file("tileset.source.json");
    }
    std::printf("[LOD] output %s: %zu rewritten contents, %zu copied files, %zu manifests updated\n",
                path_utf8(output).c_str(), changes.size(), copies.size(), written);
    std::fflush(stdout);
}

} // namespace ai3d::waterpost
