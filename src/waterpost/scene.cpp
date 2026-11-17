// assign / scene：见 scene.hpp。

#include "scene.hpp"

#include "error.hpp"
#include "fs_util.hpp"
#include "geo.hpp"
#include "json_util.hpp"
#include "lod.hpp"
#include "masks.hpp"
#include "npz_util.hpp"
#include "parallel.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>

namespace ai3d::waterpost
{
namespace
{

constexpr const char* ASSIGN_SCHEMA = "ai3d-waterpost.assign/1";
constexpr const char* SCENE_SCHEMA = "ai3d-waterpost.scene/1";

std::filesystem::path assign_dir(const Config& config)
{
    return config.work / "assign";
}

std::filesystem::path scene_dir(const Config& config)
{
    return config.work / "scene";
}

std::string normalized_name(std::string name)
{
    for (char& c : name)
    {
        c = c == '\\' ? '/' : c;
    }
    return name;
}

// 清单：每行一个图像名（与位姿 JSON / COLMAP 里的 name 相同）；空行与 # 开头的行忽略。
std::vector<std::string> read_list(const std::filesystem::path& path)
{
    std::vector<std::string> names;
    std::string text = read_file_text(path);
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF)
    {
        text = text.substr(3);
    }
    std::size_t start = 0;
    while (start < text.size())
    {
        std::size_t end = text.find('\n', start);
        end = end == std::string::npos ? text.size() : end;
        std::string line = text.substr(start, end - start);
        start = end + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
        {
            line.pop_back();
        }
        const std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }
        names.push_back(normalized_name(line.substr(first)));
    }
    return names;
}

std::string fnv_hex(const std::vector<View>& views)
{
    std::uint64_t h = 1469598103934665603ULL;
    for (const View& view : views)
    {
        for (unsigned char c : view.mask_key)
        {
            h ^= c;
            h *= 1099511628211ULL;
        }
        h ^= 0xffu;
        h *= 1099511628211ULL;
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h));
    return text;
}

Json box_json(const Box3& box)
{
    Json value;
    value["minimum"] = box.minimum;
    value["maximum"] = box.maximum;
    return value;
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

int model_id_of(const std::string& name)
{
    static const std::map<std::string, int> ids = {{"SIMPLE_PINHOLE", 0}, {"PINHOLE", 1},       {"SIMPLE_RADIAL", 2},
                                                   {"RADIAL", 3},         {"OPENCV", 4},        {"OPENCV_FISHEYE", 5},
                                                   {"FULL_OPENCV", 6},    {"FOV", 7},           {"SIMPLE_RADIAL_FISHEYE", 8},
                                                   {"RADIAL_FISHEYE", 9}, {"THIN_PRISM_FISHEYE", 10}};
    const auto found = ids.find(name);
    if (found == ids.end())
    {
        fail_value("Unknown camera model in the pose file: " + name);
    }
    return found->second;
}

// 3x3：嵌套 [[..],[..],[..]] 或扁平 9 个数（行主序）。
std::array<double, 9> matrix3(const Json& value)
{
    std::array<double, 9> m{};
    if (value.size() == 9)
    {
        for (std::size_t k = 0; k < 9; ++k)
        {
            m[k] = jnum(value[k]);
        }
        return m;
    }
    if (value.size() != 3)
    {
        fail_value("Rotation must be 3x3 (nested or 9 numbers)");
    }
    for (std::size_t r = 0; r < 3; ++r)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            m[3 * r + c] = jnum(value[r].at(c));
        }
    }
    return m;
}

std::array<double, 3> vector3(const Json& value)
{
    if (value.size() != 3)
    {
        fail_value("Expected a 3-vector in the pose file");
    }
    return {jnum(value[0]), jnum(value[1]), jnum(value[2])};
}

struct PoseCamera
{
    std::string model;
    int width = 0;
    int height = 0;
    std::vector<double> params;
};

PoseCamera pose_camera(const Json& value)
{
    PoseCamera camera;
    camera.model = value.value("model", std::string("PINHOLE"));
    camera.width = static_cast<int>(jint(value.at("width")));
    camera.height = static_cast<int>(jint(value.at("height")));
    if (value.contains("params"))
    {
        for (const Json& p : value.at("params"))
        {
            camera.params.push_back(jnum(p));
        }
    }
    else
    {
        // 也接受 fx / fy / cx / cy 字段。
        camera.model = "PINHOLE";
        camera.params = {jnum(value.at("fx")), jnum(value.at("fy")), jnum(value.at("cx")), jnum(value.at("cy"))};
    }
    return camera;
}

// 输入视图：blocks 模式为各块 sparse/0 的并集，poses 模式为位姿 JSON；按 key() 排序、去重。
std::vector<View> input_views(const Config& config, const std::vector<std::string>& blocks)
{
    if (config.mode == "poses")
    {
        return read_pose_views(config);
    }
    std::map<std::string, View> unique;
    for (const std::string& name : blocks)
    {
        for (View& view : block_views(config.project, name, config.mask_dir))
        {
            unique.emplace(view.key(), std::move(view));
        }
    }
    std::vector<View> views;
    for (auto& [key, view] : unique)
    {
        views.push_back(std::move(view));
    }
    return views;
}

struct BlockGeometry
{
    std::string name;
    Box3 bounds;
    std::vector<LodLeaf> leaves;
    double reach[2] = {0, 0}; // 相机中心 XY 允许落在 [min - reach, max + reach]
};

std::vector<BlockGeometry> block_geometry(const Config& config, const std::vector<std::string>& names)
{
    const LodIndex lod = read_lod_index(config.lod);
    std::vector<BlockGeometry> out;
    for (const std::string& name : names)
    {
        const LodBlock* block = lod.block(name);
        BlockGeometry g;
        g.name = name;
        if (block)
        {
            g.leaves = block->leaves;
            for (const LodLeaf& leaf : block->leaves)
            {
                g.bounds.expand(leaf.box);
            }
        }
        if (g.bounds.valid())
        {
            const double size = std::max(g.bounds.maximum[0] - g.bounds.minimum[0], g.bounds.maximum[1] - g.bounds.minimum[1]);
            g.reach[0] = g.reach[1] = config.view_margin * size;
        }
        out.push_back(std::move(g));
    }
    return out;
}

// 视图看到的叶节点数（相机中心不在块附近时为 0）。
int visible_leaves(const View& view, const BlockGeometry& block)
{
    if (!block.bounds.valid())
    {
        return 0;
    }
    for (int a = 0; a < 2; ++a)
    {
        if (view.center[static_cast<std::size_t>(a)] < block.bounds.minimum[a] - block.reach[a] ||
            view.center[static_cast<std::size_t>(a)] > block.bounds.maximum[a] + block.reach[a])
        {
            return 0;
        }
    }
    int count = 0;
    for (const LodLeaf& leaf : block.leaves)
    {
        count += view_sees_box(view, leaf.box.minimum, leaf.box.maximum, 0.0) ? 1 : 0;
    }
    return count;
}

struct SceneCache
{
    std::mutex mutex;
    std::filesystem::path work;
    bool scene_loaded = false;
    Scene scene;
    bool views_loaded = false;
    std::unordered_map<std::string, View> views;
    bool occupancy_loaded = false;
    std::unordered_map<std::string, Occupancy> occupancy;
};

SceneCache& cache_for(const Config& config)
{
    static SceneCache cache;
    if (cache.work != config.work)
    {
        cache.work = config.work;
        cache.scene_loaded = cache.views_loaded = cache.occupancy_loaded = false;
        cache.scene = Scene();
        cache.views.clear();
        cache.occupancy.clear();
    }
    return cache;
}

std::filesystem::path require_scene_file(const Config& config, const char* name)
{
    const std::filesystem::path path = scene_dir(config) / name;
    if (!std::filesystem::is_regular_file(path))
    {
        fail_missing("Missing " + path_utf8(path) + "; run the assign and scene stages first");
    }
    return path;
}

} // namespace

const SceneBlock* Scene::block(const std::string& name) const
{
    for (const SceneBlock& b : blocks)
    {
        if (b.name == name)
        {
            return &b;
        }
    }
    return nullptr;
}

namespace
{

std::string lower_ascii(std::string text)
{
    for (char& c : text)
    {
        c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }
    return text;
}

// 图像名拆成路径分量（反斜杠当斜杠，去掉空分量；UNC 的 //server/share 也一样拆开）。
std::vector<std::string> name_components(const std::string& name)
{
    std::vector<std::string> parts;
    std::string current;
    for (char c : name)
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
            current += c;
        }
    }
    if (!current.empty())
    {
        parts.push_back(current);
    }
    return parts;
}

// 去掉前 skip 个分量后的相对路径，扩展名换成 .png，接到 mask_dir 后面。
std::filesystem::path mask_under(const std::filesystem::path& mask_dir, const std::vector<std::string>& parts, std::size_t skip)
{
    std::filesystem::path result = mask_dir;
    for (std::size_t k = skip; k < parts.size(); ++k)
    {
        if (parts[k] == "..")
        {
            fail_value("Image name must not contain '..'");
        }
        result /= utf8_path(parts[k]);
    }
    result.replace_extension(".png");
    return result;
}

bool is_absolute_name(const std::string& name)
{
    return !name.empty() && (name[0] == '/' || name[0] == '\\' || (name.size() > 1 && name[1] == ':'));
}

int distortion_model(const std::string& model)
{
    if (model == "SIMPLE_RADIAL") return 2;
    if (model == "RADIAL") return 3;
    if (model == "OPENCV") return 4;
    if (model == "FULL_OPENCV") return 6;
    return -1;
}

} // namespace

std::vector<View> read_pose_views(const Config& config)
{
    // 1) 位姿文件：--poses（cameras 与 images 在同一个 JSON）或 --cameras + --images（两个文件）。
    Json cameras_json = Json::object(), images_json;
    std::string source;
    if (!config.poses.empty())
    {
        const Json root = read_json_file(config.poses);
        if (root.contains("cameras"))
        {
            cameras_json = root.at("cameras");
        }
        images_json = root.at("images");
        source = path_utf8(config.poses);
    }
    else if (!config.cameras.empty() && !config.images.empty())
    {
        cameras_json = read_json_file(config.cameras);
        images_json = read_json_file(config.images);
        if (cameras_json.is_object() && cameras_json.contains("cameras"))
        {
            cameras_json = Json(cameras_json.at("cameras"));
        }
        if (images_json.is_object() && images_json.contains("images"))
        {
            images_json = Json(images_json.at("images"));
        }
        source = path_utf8(config.images);
    }
    else
    {
        fail_value("poses mode needs --poses FILE, or --cameras FILE with --images FILE");
    }
    std::map<std::string, PoseCamera> cameras;
    if (cameras_json.is_array())
    {
        for (const Json& item : cameras_json)
        {
            cameras[std::to_string(jint(item.at("id")))] = pose_camera(item);
        }
    }
    else
    {
        for (auto it = cameras_json.begin(); it != cameras_json.end(); ++it)
        {
            cameras[it.key()] = pose_camera(it.value());
        }
    }
    // 图像：数组（id 字段或序号）或以 id 为键的对象。
    std::vector<std::pair<long long, const Json*>> entries;
    if (images_json.is_array())
    {
        long long index = 0;
        for (const Json& item : images_json)
        {
            ++index;
            entries.emplace_back(item.contains("id") ? jint(item.at("id")) : index, &item);
        }
    }
    else
    {
        for (auto it = images_json.begin(); it != images_json.end(); ++it)
        {
            entries.emplace_back(std::stoll(it.key()), &it.value());
        }
    }

    // 2) 逐张建视图（掩码稍后定）。
    std::vector<View> parsed;
    parsed.reserve(entries.size());
    std::size_t distorted = 0;
    for (const auto& [id, item] : entries)
    {
        const Json& image = *item;
        View view;
        view.name = image.at("name").get<std::string>();
        view.image_id = static_cast<std::uint32_t>(id);
        PoseCamera camera;
        if (image.contains("camera"))
        {
            camera = pose_camera(image.at("camera"));
        }
        else
        {
            const std::string cid = std::to_string(jint(image.at("camera_id")));
            const auto found = cameras.find(cid);
            if (found == cameras.end())
            {
                fail_value("Pose file: image " + view.name + " references missing camera " + cid);
            }
            camera = found->second;
            view.camera_id = static_cast<std::uint32_t>(std::stoul(cid));
        }
        const int model_id = model_id_of(camera.model);
        {
            static const int counts[] = {3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12};
            const std::size_t expected = static_cast<std::size_t>(counts[model_id]);
            if (camera.params.size() != expected)
            {
                fail_value("Pose file: camera of image " + view.name + " (" + camera.model + ") needs " +
                           std::to_string(expected) + " params, got " + std::to_string(camera.params.size()));
            }
            if (camera.width <= 0 || camera.height <= 0)
            {
                fail_value("Pose file: camera of image " + view.name + " has a non-positive size");
            }
        }
        view.width = camera.width;
        view.height = camera.height;
        const int raw = distortion_model(camera.model);
        if (raw >= 0)
        {
            // 带畸变：投影用内参不变的针孔相机，原始相机留着给读掩码时去畸变；畸变系数全 0 时就是针孔。
            const std::size_t first = raw == 2 || raw == 3 ? 3 : 4;
            bool any = false;
            for (std::size_t k = first; k < camera.params.size(); ++k)
            {
                any |= camera.params[k] != 0;
            }
            double fx, fy, cx, cy;
            raw_intrinsics(raw, camera.params, fx, fy, cx, cy);
            view.model = "PINHOLE";
            view.model_id = 1;
            view.params = {fx, fy, cx, cy};
            if (any)
            {
                view.raw_model_id = raw;
                view.raw_params = camera.params;
                ++distorted;
            }
        }
        else if (model_id == 0 || model_id == 1)
        {
            view.model = camera.model;
            view.model_id = model_id;
            view.params = camera.params;
        }
        else
        {
            fail_value("Pose file: camera model " + camera.model + " of image " + view.name +
                       " is not supported (fisheye models need their own undistortion)");
        }
        // 世界到相机：cam = R * p + t。
        if (image.contains("qvec") || image.contains("q"))
        {
            const Json& q = image.contains("qvec") ? image.at("qvec") : image.at("q");
            view.r_wc = qvec_to_rotmat({jnum(q.at(0)), jnum(q.at(1)), jnum(q.at(2)), jnum(q.at(3))});
            view.t_wc = vector3(image.contains("tvec") ? image.at("tvec") : image.at("t"));
        }
        else if (image.contains("camera_to_world"))
        {
            const Json& m = image.at("camera_to_world");
            std::array<double, 9> c2w{};
            std::array<double, 3> c{};
            for (std::size_t r = 0; r < 3; ++r)
            {
                for (std::size_t k = 0; k < 3; ++k)
                {
                    c2w[3 * r + k] = m.size() == 16 ? jnum(m[4 * r + k]) : jnum(m[r].at(k));
                }
                c[r] = m.size() == 16 ? jnum(m[4 * r + 3]) : jnum(m[r].at(3));
            }
            for (std::size_t r = 0; r < 3; ++r)
            {
                for (std::size_t k = 0; k < 3; ++k)
                {
                    view.r_wc[3 * r + k] = c2w[3 * k + r];
                }
            }
            for (std::size_t r = 0; r < 3; ++r)
            {
                view.t_wc[r] = -(view.r_wc[3 * r] * c[0] + view.r_wc[3 * r + 1] * c[1] + view.r_wc[3 * r + 2] * c[2]);
            }
        }
        else if (image.contains("R"))
        {
            view.r_wc = matrix3(image.at("R"));
            if (image.contains("t"))
            {
                view.t_wc = vector3(image.at("t"));
            }
            else
            {
                const std::array<double, 3> c = vector3(image.at("center"));
                for (std::size_t r = 0; r < 3; ++r)
                {
                    view.t_wc[r] = -(view.r_wc[3 * r] * c[0] + view.r_wc[3 * r + 1] * c[1] + view.r_wc[3 * r + 2] * c[2]);
                }
            }
        }
        else
        {
            fail_value("Pose file: image " + view.name + " needs q(vec)+t(vec), R+t, R+center or camera_to_world");
        }
        for (std::size_t i = 0; i < 3; ++i)
        {
            view.center[i] = -(view.r_wc[0 * 3 + i] * view.t_wc[0] + view.r_wc[1 * 3 + i] * view.t_wc[1] +
                               view.r_wc[2 * 3 + i] * view.t_wc[2]);
        }
        if (image.contains("mask"))
        {
            const std::filesystem::path mask = utf8_path(image.at("mask").get<std::string>());
            view.mask_path = mask.is_absolute() ? mask : config.mask_dir / mask;
        }
        parsed.push_back(std::move(view));
    }
    if (distorted)
    {
        log("[POSES] %zu of %zu images use a distorted camera: masks are undistorted to the pinhole camera when read",
            distorted, parsed.size());
    }

    // 3) 坐标系：位姿是投影坐标（UTM）时换到 LOD 本地坐标。本地原点 = LOD 根节点 transform 的平移（ECEF）。
    {
        int zone = 0;
        bool south = false;
        parse_pose_crs(config.pose_crs, zone, south);
        if (zone == 0)
        {
            std::vector<double> magnitude;
            for (const View& view : parsed)
            {
                magnitude.push_back(std::max(std::abs(view.center[0]), std::abs(view.center[1])));
            }
            zone = !magnitude.empty() && median(magnitude) > 1e5 ? 0 : -1; // 明显不是本地坐标才换算
            if (zone == 0)
            {
                log("[POSES] camera centres look like projected coordinates; converting with the LOD georeference "
                    "(pass --pose-crs none if they are already local)");
            }
        }
        if (zone != -1)
        {
            std::array<double, 16> m{};
            if (!read_lod_root_transform(config.lod, m))
            {
                fail_value("--pose-crs needs the LOD root transform (local -> ECEF), but " + path_utf8(config.lod) +
                           " has none; pass --pose-crs none if the poses are already in LOD coordinates");
            }
            const Geodetic origin = ecef_to_geodetic(m[12], m[13], m[14]);
            if (zone == 0)
            {
                zone = utm_zone_of(origin.longitude);
                south = origin.latitude < 0;
            }
            const std::array<double, 2> en = utm_forward(origin.latitude, origin.longitude, zone, south);
            const double offset[3] = {en[0], en[1], origin.height};
            for (View& view : parsed)
            {
                for (int k = 0; k < 3; ++k)
                {
                    view.center[static_cast<std::size_t>(k)] -= offset[k];
                }
                for (std::size_t r = 0; r < 3; ++r)
                {
                    view.t_wc[r] = -(view.r_wc[3 * r] * view.center[0] + view.r_wc[3 * r + 1] * view.center[1] +
                                     view.r_wc[3 * r + 2] * view.center[2]);
                }
            }
            log("[POSES] UTM zone %d%s: LOD origin (lon %.7f, lat %.7f, h %.3f) = E %.3f N %.3f; poses shifted to LOD coordinates",
                zone, south ? "S" : "N", origin.longitude * 180 / M_PI, origin.latitude * 180 / M_PI, origin.height, en[0], en[1]);
        }
    }

    // 4) 掩码：<mask-dir>/<图像名去掉前 k 级目录，扩展名换 .png>。--image-root 给了就去掉这个前缀；
    //    没给时用前几张图自动找 k（从 0 起，第一个能在掩码目录下找到文件的级数），之后所有图用同一个 k。
    std::size_t skip = 0;
    bool skip_known = false;
    std::vector<std::string> root_parts;
    if (!config.image_root.empty())
    {
        root_parts = name_components(config.image_root);
    }
    else
    {
        std::size_t tried = 0;
        for (const View& view : parsed)
        {
            if (!view.mask_path.empty())
            {
                continue;
            }
            const std::vector<std::string> parts = name_components(view.name);
            for (std::size_t k = 0; k + 1 <= parts.size() && !skip_known; ++k)
            {
                if (std::filesystem::is_regular_file(mask_under(config.mask_dir, parts, k)))
                {
                    skip = k;
                    skip_known = true;
                }
            }
            if (skip_known || ++tried >= 50)
            {
                break;
            }
        }
        if (!skip_known)
        {
            for (const View& view : parsed)
            {
                if (view.mask_path.empty())
                {
                    fail_value("No mask found under " + path_utf8(config.mask_dir) + " for image " + view.name +
                               " (tried every leading-folder cut); check --mask-dir or pass --image-root");
                }
            }
        }
    }
    std::map<std::string, View> unique;
    std::unordered_map<std::string, std::string> owner;
    std::size_t missing = 0;
    std::string example;
    for (View& view : parsed)
    {
        if (view.mask_path.empty())
        {
            std::vector<std::string> parts = name_components(view.name);
            std::size_t cut = skip;
            if (!root_parts.empty())
            {
                bool match = parts.size() > root_parts.size();
                for (std::size_t k = 0; match && k < root_parts.size(); ++k)
                {
                    match = lower_ascii(parts[k]) == lower_ascii(root_parts[k]);
                }
                if (!match)
                {
                    fail_value("Image " + view.name + " does not start with --image-root " + config.image_root);
                }
                cut = root_parts.size();
            }
            else if (!skip_known && is_absolute_name(view.name))
            {
                fail_value("Image name is an absolute path; pass --image-root: " + view.name);
            }
            view.mask_path = mask_under(config.mask_dir, parts, std::min(cut, parts.size() - 1));
        }
        if (!std::filesystem::is_regular_file(view.mask_path))
        {
            example = example.empty() ? path_utf8(view.mask_path) : example;
            ++missing;
            continue;
        }
        view.mask_key = canonical_key(view.mask_path);
        const auto [it, inserted] = owner.emplace(view.mask_key, view.name);
        if (!inserted)
        {
            fail_value("Images " + it->second + " and " + view.name + " map to the same mask " + path_utf8(view.mask_path));
        }
        unique.emplace(view.key(), std::move(view));
    }
    if (!root_parts.empty() || skip_known)
    {
        log("[POSES] masks: <mask-dir>/<image name without its first %zu folder(s)>.png",
            root_parts.empty() ? skip : root_parts.size());
    }
    if (missing)
    {
        log("[VIEW] %zu posed views without a semantic mask are ignored (e.g. %s)", missing, example.c_str());
    }
    std::vector<View> views;
    for (auto& [key, view] : unique)
    {
        views.push_back(std::move(view));
    }
    if (views.size() < 3)
    {
        fail_value("Back-projection requires at least three semantic camera views in " + source);
    }

    // 5) 检查：相机应在 LOD 附近（坐标系弄错时这里就能发现）。
    if (!config.lod.empty() && std::filesystem::is_regular_file(config.lod / "tileset.json"))
    {
        Box3 extent;
        for (const LodBlock& block : read_lod_index(config.lod).blocks)
        {
            for (const LodLeaf& leaf : block.leaves)
            {
                extent.expand(leaf.box);
            }
        }
        if (extent.valid())
        {
            const double size = std::max({extent.maximum[0] - extent.minimum[0], extent.maximum[1] - extent.minimum[1], 1000.0});
            std::size_t near = 0;
            for (const View& view : views)
            {
                near += view.center[0] > extent.minimum[0] - size && view.center[0] < extent.maximum[0] + size &&
                        view.center[1] > extent.minimum[1] - size && view.center[1] < extent.maximum[1] + size;
            }
            if (near * 2 < views.size())
            {
                fail_value("Only " + std::to_string(near) + " of " + std::to_string(views.size()) +
                           " cameras are near the LOD; the poses and the LOD are in different coordinate systems "
                           "(check --pose-crs)");
            }
        }
    }
    return views;
}

std::vector<std::string> input_blocks(const Config& config)
{
    const LodIndex lod = read_lod_index(config.lod);
    std::vector<std::string> result;
    if (config.mode == "poses")
    {
        for (const LodBlock& block : lod.blocks)
        {
            result.push_back(block.name);
        }
        std::sort(result.begin(), result.end());
    }
    else
    {
        for (const std::string& name : project_blocks(config.project))
        {
            if (lod.block(name))
            {
                result.push_back(name);
            }
            else
            {
                log("[BLOCKS] %s has no contents in the LOD; skipped", name.c_str());
            }
        }
    }
    if (result.empty())
    {
        fail_value("No block of the input is present in the LOD");
    }
    return result;
}

void run_split(const Config& config, int parts, bool interleave)
{
    if (parts < 1)
    {
        fail_value("--parts must be >= 1");
    }
    const std::vector<std::string> blocks = input_blocks(config);
    const std::vector<View> views = input_views(config, blocks);
    const std::filesystem::path dir = assign_dir(config) / "lists";
    std::filesystem::create_directories(dir);
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        if (entry.path().extension() == ".list")
        {
            std::filesystem::remove(entry.path());
        }
    }
    const std::size_t n = views.size(), p = static_cast<std::size_t>(parts);
    for (std::size_t i = 0; i < p; ++i)
    {
        std::string text;
        std::size_t count = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            const bool mine = interleave ? k % p == i : (k * p / n) == i;
            if (mine)
            {
                text += views[k].key() + "\n";
                ++count;
            }
        }
        char name[32];
        std::snprintf(name, sizeof(name), "part_%02zu.list", i);
        const std::filesystem::path target = dir / name;
        write_file_atomic(target, text.data(), text.size(), target.string() + ".tmp");
        log("[SPLIT] %s: %zu views", name, count);
    }
    log("[SPLIT] %zu views into %d lists (%s) in %s", n, parts, interleave ? "interleave" : "contiguous", path_utf8(dir).c_str());
}

void clear_assign_results(const Config& config)
{
    const std::filesystem::path dir = assign_dir(config);
    if (!std::filesystem::is_directory(dir))
    {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        const auto ext = entry.path().extension();
        if (entry.is_regular_file() && (ext == ".json" || ext == ".npz"))
        {
            std::filesystem::remove(entry.path());
        }
    }
}

void run_assign(const Config& config, const std::filesystem::path& list)
{
    const Timer total;
    const std::vector<std::string> blocks = input_blocks(config);
    const std::vector<View> views = input_views(config, blocks);
    std::vector<std::size_t> mine;
    std::string shard_name = "all";
    if (list.empty())
    {
        for (std::size_t k = 0; k < views.size(); ++k)
        {
            mine.push_back(k);
        }
    }
    else
    {
        std::unordered_map<std::string, std::size_t> by_name;
        for (std::size_t k = 0; k < views.size(); ++k)
        {
            by_name[views[k].key()] = k;
        }
        std::set<std::size_t> seen;
        std::size_t unknown = 0;
        std::string example;
        for (const std::string& name : read_list(list))
        {
            const auto found = by_name.find(name);
            if (found == by_name.end())
            {
                example = example.empty() ? name : example;
                ++unknown;
                continue;
            }
            if (seen.insert(found->second).second)
            {
                mine.push_back(found->second);
            }
        }
        if (unknown)
        {
            fail_value(std::to_string(unknown) + " images of " + path_utf8(list) +
                       " are not input views with a mask (e.g. " + example + ")");
        }
        std::sort(mine.begin(), mine.end());
        shard_name = list.stem().u8string();
        if (shard_name.empty() || shard_name == "lists")
        {
            fail_value("Invalid list file name: " + path_utf8(list));
        }
    }
    const bool poses = config.mode == "poses";
    std::vector<BlockGeometry> geometry;
    if (poses)
    {
        geometry = block_geometry(config, blocks);
    }
    std::vector<View> subset;
    for (std::size_t k : mine)
    {
        subset.push_back(views[k]);
    }
    require_pinhole(subset);

    MaskCache cache(config.water_class, std::size_t{64} << 20);
    std::vector<Occupancy> occupancy(mine.size());
    std::vector<std::vector<std::int32_t>> leaves(mine.size(), std::vector<std::int32_t>(geometry.size(), 0));
    parallel_for_limited(mine.size(), 8, [&](std::size_t j) {
        const View& view = views[mine[j]];
        occupancy[j] = cache.occupancy(view);
        for (std::size_t b = 0; b < geometry.size(); ++b)
        {
            leaves[j][b] = visible_leaves(view, geometry[b]);
        }
    });

    std::vector<std::uint8_t> blob;
    std::vector<std::int64_t> offsets, bytes, water_pixels;
    std::vector<std::int32_t> rows, cols, vis_view, vis_block, vis_leaves;
    Json keys = Json::array();
    std::size_t wet = 0;
    for (std::size_t j = 0; j < mine.size(); ++j)
    {
        const Occupancy& o = occupancy[j];
        keys.push_back(views[mine[j]].mask_key);
        offsets.push_back(static_cast<std::int64_t>(blob.size()));
        bytes.push_back(static_cast<std::int64_t>(o.packed.size()));
        blob.insert(blob.end(), o.packed.begin(), o.packed.end());
        rows.push_back(o.rows);
        cols.push_back(o.cols);
        water_pixels.push_back(o.water_pixels);
        wet += o.water_pixels ? 1 : 0;
        for (std::size_t b = 0; b < geometry.size(); ++b)
        {
            if (leaves[j][b] > 0)
            {
                vis_view.push_back(static_cast<std::int32_t>(j));
                vis_block.push_back(static_cast<std::int32_t>(b));
                vis_leaves.push_back(leaves[j][b]);
            }
        }
    }
    const std::filesystem::path out = assign_dir(config);
    std::filesystem::create_directories(out);
    const std::string stem = shard_name;
    NpzWriter writer;
    npz_put(writer, "occupancy_blob", blob);
    npz_put(writer, "occupancy_offsets", offsets);
    npz_put(writer, "occupancy_bytes", bytes);
    npz_put(writer, "occupancy_rows", rows);
    npz_put(writer, "occupancy_cols", cols);
    npz_put(writer, "occupancy_water_pixels", water_pixels);
    npz_put(writer, "visible_view", vis_view);
    npz_put(writer, "visible_block", vis_block);
    npz_put(writer, "visible_leaves", vis_leaves);
    writer.write(out / (stem + ".npz"), true);
    // JSON 最后写：它在就表示这一片完整。
    Json record;
    record["schema"] = ASSIGN_SCHEMA;
    record["mode"] = config.mode;
    record["name"] = shard_name;
    record["list"] = list.empty() ? std::string() : path_utf8(list);
    record["view_count"] = views.size();
    record["views_hash"] = fnv_hex(views);
    record["blocks"] = blocks;
    record["keys"] = keys;
    record["wet_views"] = wet;
    record["mask_decodes"] = cache.decodes();
    record["seconds"] = total.seconds();
    write_json_atomic(out / (stem + ".json"), record, false);
    log("[ASSIGN] %s: %zu of %zu views, %zu with water, %zu block visibilities (%.1fs)", shard_name.c_str(), mine.size(),
        views.size(), wet, vis_view.size(), total.seconds());
}

void run_scene(const Config& config)
{
    const Timer total;
    const std::vector<std::string> names = input_blocks(config);
    const std::vector<View> views = input_views(config, names);
    const std::string hash = fnv_hex(views);
    const std::filesystem::path dir = assign_dir(config);
    std::vector<std::string> shard_names;
    if (std::filesystem::is_directory(dir))
    {
        for (const auto& entry : std::filesystem::directory_iterator(dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".json")
            {
                shard_names.push_back(entry.path().stem().u8string());
            }
        }
    }
    std::sort(shard_names.begin(), shard_names.end());
    if (shard_names.empty())
    {
        fail_missing("No assign result in " + path_utf8(dir) + "; run the assign stage first");
    }
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t k = 0; k < views.size(); ++k)
    {
        index[views[k].mask_key] = k;
    }
    std::vector<Occupancy> occupancy(views.size());
    std::vector<int> owner(views.size(), -1);
    // 稀疏：每块只记看到它的视图（视图下标, 叶节点数），内存与"块数 × 视图数"无关。
    std::vector<std::vector<std::pair<std::size_t, std::int32_t>>> visible(names.size());
    for (std::size_t s = 0; s < shard_names.size(); ++s)
    {
        const std::string& stem = shard_names[s];
        const Json record = read_json_file(dir / (stem + ".json"));
        if (record.value("schema", std::string()) != ASSIGN_SCHEMA || record.value("mode", std::string()) != config.mode ||
            record.value("views_hash", std::string()) != hash ||
            record.at("blocks").get<std::vector<std::string>>() != names)
        {
            fail_value("Assign result " + stem + " was made from different inputs (mode, views or LOD blocks); rerun assign");
        }
        const NpzReader reader(dir / (stem + ".npz"));
        const std::vector<std::uint8_t> blob = npz_get<std::uint8_t>(reader, "occupancy_blob");
        const std::vector<std::int64_t> offsets = npz_get<std::int64_t>(reader, "occupancy_offsets");
        const std::vector<std::int64_t> bytes = npz_get<std::int64_t>(reader, "occupancy_bytes");
        const std::vector<std::int32_t> rows = npz_get<std::int32_t>(reader, "occupancy_rows");
        const std::vector<std::int32_t> cols = npz_get<std::int32_t>(reader, "occupancy_cols");
        const std::vector<std::int64_t> water = npz_get<std::int64_t>(reader, "occupancy_water_pixels");
        const std::vector<std::int32_t> vis_view = npz_get<std::int32_t>(reader, "visible_view");
        const std::vector<std::int32_t> vis_block = npz_get<std::int32_t>(reader, "visible_block");
        const std::vector<std::int32_t> vis_leaves = npz_get<std::int32_t>(reader, "visible_leaves");
        const Json& keys = record.at("keys");
        std::vector<std::size_t> global(keys.size());
        for (std::size_t j = 0; j < keys.size(); ++j)
        {
            const auto found = index.find(keys[j].get<std::string>());
            if (found == index.end())
            {
                fail_value("Assign shard " + stem + " has an unknown view");
            }
            const std::size_t k = found->second;
            global[j] = k;
            if (owner[k] >= 0)
            {
                fail_value("Image " + views[k].name + " was processed by both " + shard_names[static_cast<std::size_t>(owner[k])] +
                           " and " + stem + "; remove the stale result from " + path_utf8(dir));
            }
            owner[k] = static_cast<int>(s);
            Occupancy& o = occupancy[k];
            o.rows = rows[j];
            o.cols = cols[j];
            o.water_pixels = water[j];
            o.packed.assign(blob.begin() + offsets[j], blob.begin() + offsets[j] + bytes[j]);
        }
        for (std::size_t r = 0; r < vis_view.size(); ++r)
        {
            visible[static_cast<std::size_t>(vis_block[r])].emplace_back(global[static_cast<std::size_t>(vis_view[r])], vis_leaves[r]);
        }
    }
    std::size_t missing = 0;
    std::string example;
    for (std::size_t k = 0; k < views.size(); ++k)
    {
        if (owner[k] < 0)
        {
            example = example.empty() ? views[k].name : example;
            ++missing;
        }
    }
    if (missing)
    {
        fail_value(std::to_string(missing) + " images are not in any assign result (e.g. " + example +
                   "); some list was not processed");
    }

    Scene scene;
    scene.mode = config.mode;
    const bool poses = config.mode == "poses";
    std::vector<BlockGeometry> geometry;
    if (poses)
    {
        geometry = block_geometry(config, names);
        if (!config.bbox.empty())
        {
            scene.domain = load_domain_files(config.bbox, config.roi);
        }
        else
        {
            Box3 all;
            for (const BlockGeometry& g : geometry)
            {
                all.expand(g.bounds);
            }
            scene.domain.lower = all.minimum;
            scene.domain.upper = all.maximum;
            if (!config.roi.empty())
            {
                log("[SCENE] --roi needs --bbox (its Normalizer maps the ROI to local coordinates); ROI ignored");
            }
        }
    }
    else
    {
        scene.domain = load_domain(config.project);
    }
    for (std::size_t b = 0; b < names.size(); ++b)
    {
        SceneBlock block;
        block.name = names[b];
        if (poses)
        {
            block.bounds = geometry[b].bounds;
            std::sort(visible[b].begin(), visible[b].end());
            for (const auto& [k, leaves] : visible[b])
            {
                if (leaves >= std::max(1, config.min_view_leaves))
                {
                    block.view_keys.push_back(views[k].mask_key);
                }
            }
        }
        else
        {
            block.bounds = block_core_bounds(config.project, names[b]);
            for (const View& view : block_views(config.project, names[b], config.mask_dir))
            {
                block.view_keys.push_back(view.mask_key);
            }
        }
        if (block.view_keys.size() < 3)
        {
            log("[SCENE] %s: only %zu views see the block; it is treated as dry", block.name.c_str(), block.view_keys.size());
        }
        scene.blocks.push_back(std::move(block));
    }

    const std::filesystem::path out = scene_dir(config);
    std::filesystem::create_directories(out);
    Json records = Json::array();
    NpzWriter writer;
    std::vector<std::uint8_t> blob;
    std::vector<std::int64_t> offsets, bytes, water_pixels;
    std::vector<std::int32_t> rows, cols;
    std::size_t wet = 0;
    for (std::size_t k = 0; k < views.size(); ++k)
    {
        records.push_back(view_record(views[k]));
        const Occupancy& o = occupancy[k];
        offsets.push_back(static_cast<std::int64_t>(blob.size()));
        bytes.push_back(static_cast<std::int64_t>(o.packed.size()));
        blob.insert(blob.end(), o.packed.begin(), o.packed.end());
        rows.push_back(o.rows);
        cols.push_back(o.cols);
        water_pixels.push_back(o.water_pixels);
        wet += o.water_pixels ? 1 : 0;
    }
    npz_put(writer, "occupancy_blob", blob);
    npz_put(writer, "occupancy_offsets", offsets);
    npz_put(writer, "occupancy_bytes", bytes);
    npz_put(writer, "occupancy_rows", rows);
    npz_put(writer, "occupancy_cols", cols);
    npz_put(writer, "occupancy_water_pixels", water_pixels);
    writer.write(out / "occupancy.npz", true);
    write_json_atomic(out / "views.json", records, true);
    Json report;
    report["schema"] = SCENE_SCHEMA;
    report["mode"] = scene.mode;
    report["shards"] = shard_names;
    report["view_count"] = views.size();
    report["wet_views"] = wet;
    report["views_hash"] = hash;
    report["domain"] = scene.domain.json();
    Json blocks = Json::array();
    for (const SceneBlock& block : scene.blocks)
    {
        Json item;
        item["name"] = block.name;
        item["bounds"] = box_json(block.bounds);
        item["view_count"] = block.view_keys.size();
        item["views"] = block.view_keys;
        blocks.push_back(item);
        log("[SCENE] %s: %zu views", block.name.c_str(), block.view_keys.size());
    }
    report["blocks"] = blocks;
    report["seconds"] = total.seconds();
    write_json_atomic(out / "scene.json", report, false);
    log("[SCENE] %s mode: %zu blocks, %zu views (%zu with water) from %zu assign result(s) (%.1fs)", scene.mode.c_str(),
        scene.blocks.size(), views.size(), wet, shard_names.size(), total.seconds());
}

bool has_scene(const Config& config)
{
    return std::filesystem::is_regular_file(scene_dir(config) / "scene.json");
}

namespace
{

// 调用方持有 cache.mutex。
void ensure_scene(SceneCache& cache, const Config& config)
{
    if (cache.scene_loaded)
    {
        return;
    }
    Scene scene;
    if (!has_scene(config))
    {
        if (config.mode != "blocks")
        {
            require_scene_file(config, "scene.json");
        }
        // blocks 模式、没有 scene：与改动前相同，块、范围、视图直接取自 artifacts/blocks。
        scene.mode = "blocks";
        scene.domain = load_domain(config.project);
        for (const std::string& name : input_blocks(config))
        {
            SceneBlock block;
            block.name = name;
            block.bounds = block_core_bounds(config.project, name);
            for (View& view : block_views(config.project, name, config.mask_dir))
            {
                block.view_keys.push_back(view.mask_key);
                cache.views.emplace(view.mask_key, std::move(view));
            }
            scene.blocks.push_back(std::move(block));
        }
        cache.views_loaded = true;
    }
    else
    {
        const Json value = read_json_file(scene_dir(config) / "scene.json");
        if (value.value("schema", std::string()) != SCENE_SCHEMA)
        {
            fail_value("Unexpected work/scene/scene.json schema");
        }
        scene.mode = value.at("mode").get<std::string>();
        if (scene.mode != config.mode)
        {
            fail_value("work/scene was made in " + scene.mode + " mode but the work directory is in " + config.mode +
                       " mode; rerun assign and scene");
        }
        scene.domain = Domain::from_json(value.at("domain"));
        for (const Json& item : value.at("blocks"))
        {
            SceneBlock block;
            block.name = item.at("name").get<std::string>();
            block.bounds = box_from(item.at("bounds"));
            block.view_keys = item.at("views").get<std::vector<std::string>>();
            scene.blocks.push_back(std::move(block));
        }
    }
    cache.scene = std::move(scene);
    cache.scene_loaded = true;
}

} // namespace

const Scene& load_scene(const Config& config)
{
    SceneCache& cache = cache_for(config);
    std::lock_guard<std::mutex> lock(cache.mutex);
    ensure_scene(cache, config);
    return cache.scene;
}

std::vector<View> scene_views(const Config& config, const std::vector<std::string>& keys)
{
    SceneCache& cache = cache_for(config);
    std::lock_guard<std::mutex> lock(cache.mutex);
    ensure_scene(cache, config);
    if (!cache.views_loaded)
    {
        for (const Json& record : read_json_file(require_scene_file(config, "views.json")))
        {
            View view = view_from_record(record);
            cache.views.emplace(view.mask_key, std::move(view));
        }
        cache.views_loaded = true;
    }
    std::vector<View> out;
    out.reserve(keys.size());
    for (const std::string& key : keys)
    {
        const auto found = cache.views.find(key);
        if (found == cache.views.end())
        {
            fail_value("View is not in work/scene/views.json: " + key);
        }
        out.push_back(found->second);
    }
    return out;
}

const std::unordered_map<std::string, Occupancy>& scene_occupancy(const Config& config)
{
    SceneCache& cache = cache_for(config);
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.occupancy_loaded)
    {
        const Json records = read_json_file(require_scene_file(config, "views.json"));
        const NpzReader reader(require_scene_file(config, "occupancy.npz"));
        const std::vector<std::uint8_t> blob = npz_get<std::uint8_t>(reader, "occupancy_blob");
        const std::vector<std::int64_t> offsets = npz_get<std::int64_t>(reader, "occupancy_offsets");
        const std::vector<std::int64_t> bytes = npz_get<std::int64_t>(reader, "occupancy_bytes");
        const std::vector<std::int32_t> rows = npz_get<std::int32_t>(reader, "occupancy_rows");
        const std::vector<std::int32_t> cols = npz_get<std::int32_t>(reader, "occupancy_cols");
        const std::vector<std::int64_t> water = npz_get<std::int64_t>(reader, "occupancy_water_pixels");
        for (std::size_t i = 0; i < records.size(); ++i)
        {
            Occupancy o;
            o.rows = rows[i];
            o.cols = cols[i];
            o.water_pixels = water[i];
            o.packed.assign(blob.begin() + offsets[i], blob.begin() + offsets[i] + bytes[i]);
            cache.occupancy.emplace(records[i].at("mask_key").get<std::string>(), std::move(o));
        }
        cache.occupancy_loaded = true;
    }
    return cache.occupancy;
}

} // namespace ai3d::waterpost
