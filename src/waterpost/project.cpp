#include "project.hpp"

#include "error.hpp"
#include "fs_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

namespace ai3d::waterpost
{

void log(const char* format, ...)
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    va_list args;
    va_start(args, format);
    std::vfprintf(stdout, format, args);
    va_end(args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string canonical_key(const std::filesystem::path& path)
{
    std::string text = path_utf8(path);
    std::replace(text.begin(), text.end(), '\\', '/');
    return text;
}

std::vector<std::string> project_blocks(const std::filesystem::path& project)
{
    const std::filesystem::path root = project / "artifacts" / "blocks";
    for (const char* name : {"valid_chunk_list.txt", "valid_chunks.txt"})
    {
        const std::filesystem::path list = root / name;
        if (!std::filesystem::is_regular_file(list))
        {
            continue;
        }
        std::istringstream stream(read_file_text(list));
        std::string line;
        std::vector<std::string> names;
        bool first = true;
        long long expected = -1;
        while (std::getline(stream, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (first)
            {
                first = false;
                if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF)
                {
                    line = line.substr(3);
                }
                expected = std::stoll(line);
                continue;
            }
            std::istringstream fields(line);
            std::string block;
            if (fields >> block)
            {
                names.push_back(block);
            }
        }
        if (expected != static_cast<long long>(names.size()))
        {
            fail_value("Invalid block inventory: " + path_utf8(list));
        }
        std::sort(names.begin(), names.end());
        return names;
    }
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(root))
    {
        if (entry.is_directory() && std::filesystem::is_directory(entry.path() / "sparse" / "0"))
        {
            names.push_back(entry.path().filename().u8string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::filesystem::path block_dir(const std::filesystem::path& project, const std::string& name)
{
    return project / "artifacts" / "blocks" / utf8_path(name);
}

Box3 block_core_bounds(const std::filesystem::path& project, const std::string& name)
{
    const std::filesystem::path dir = block_dir(project, name);
    Box3 box;
    if (std::filesystem::is_regular_file(dir / "block.json"))
    {
        const Json value = read_json_file(dir / "block.json");
        const Json& core = value.at("core_bounds");
        for (int k = 0; k < 3; ++k)
        {
            box.minimum[k] = jnum(core.at("minimum")[static_cast<std::size_t>(k)]);
            box.maximum[k] = jnum(core.at("maximum")[static_cast<std::size_t>(k)]);
        }
        return box;
    }
    auto read3 = [](const std::filesystem::path& path) {
        std::istringstream stream(read_file_text(path));
        Vec3 v{};
        stream >> v[0] >> v[1] >> v[2];
        if (!stream)
        {
            fail_value("Invalid block extent file: " + path_utf8(path));
        }
        return v;
    };
    const Vec3 center = read3(dir / "center.txt");
    const Vec3 extent = read3(dir / "extent.txt");
    for (int k = 0; k < 3; ++k)
    {
        box.minimum[k] = center[k] - extent[k] / 2;
        box.maximum[k] = center[k] + extent[k] / 2;
    }
    return box;
}

std::vector<View> block_views(const std::filesystem::path& project, const std::string& name,
                              const std::filesystem::path& mask_dir)
{
    const ColmapCatalog catalog = read_colmap_catalog(block_dir(project, name) / "sparse" / "0");
    std::map<std::string, View> unique;
    std::size_t missing = 0;
    for (const ColmapCatalog::Image& image : catalog.images)
    {
        View view = make_view(image, catalog.camera(image.camera_id), semantic_mask_path(mask_dir, image.name));
        if (!std::filesystem::is_regular_file(view.mask_path))
        {
            ++missing;
            continue;
        }
        view.mask_key = canonical_key(view.mask_path);
        unique.emplace(view.key(), std::move(view));
    }
    if (missing)
    {
        log("[VIEW] block %s: %zu views without a semantic mask are ignored", name.c_str(), missing);
    }
    std::vector<View> views;
    for (auto& [key, view] : unique)
    {
        views.push_back(std::move(view));
    }
    if (views.size() < 3)
    {
        fail_value("Back-projection requires at least three semantic camera views: " + name);
    }
    return views;
}

PinholeCamera pinhole(const View& view)
{
    PinholeCamera camera;
    camera.width = view.width;
    camera.height = view.height;
    const std::vector<double>& p = view.params;
    if (view.model_id == 0)
    {
        camera.fx = camera.fy = p[0];
        camera.cx = p[1];
        camera.cy = p[2];
    }
    else if (view.model_id == 1)
    {
        camera.fx = p[0];
        camera.fy = p[1];
        camera.cx = p[2];
        camera.cy = p[3];
    }
    else
    {
        fail_value("Only PINHOLE/SIMPLE_PINHOLE cameras are supported by the C++ voter: " + view.model);
    }
    for (int k = 0; k < 9; ++k)
    {
        camera.r[k] = view.r_wc[static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < 3; ++k)
    {
        camera.center[k] = view.center[static_cast<std::size_t>(k)];
    }
    return camera;
}

void require_pinhole(const std::vector<View>& views)
{
    for (const View& view : views)
    {
        if (!view.pinhole())
        {
            fail_value("Distorted camera models are not supported yet (undistort first): " + view.model + " " + view.name);
        }
    }
}

double Domain::horizontal_clearance(double x, double y, int axis) const
{
    const double value = axis == 0 ? x : y;
    return std::min(value - lower[static_cast<std::size_t>(axis)], upper[static_cast<std::size_t>(axis)] - value);
}

std::array<double, 2> Domain::sampling_tolerance(double spacing) const
{
    std::array<double, 2> result{};
    for (int k = 0; k < 2; ++k)
    {
        const float magnitude = static_cast<float>(std::max(std::abs(lower[k]), std::abs(upper[k])));
        const double ulp = static_cast<double>(std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude);
        result[static_cast<std::size_t>(k)] = std::max(spacing / 64, 8 * ulp);
    }
    return result;
}

Domain load_domain(const std::filesystem::path& project)
{
    std::filesystem::path path = project / "bbox.json";
    if (!std::filesystem::is_regular_file(path))
    {
        path = project / "BoundingBox.json";
    }
    return load_domain_files(path, project / "roi.json");
}

Domain load_domain_files(const std::filesystem::path& path, const std::filesystem::path& roi)
{
    const Json value = read_json_file(path);
    const Json& box = value.at("CurrentBoundingBox");
    Domain domain;
    const char* axes[3] = {"x", "y", "z"};
    for (int k = 0; k < 3; ++k)
    {
        domain.lower[k] = jnum(box.at("min").at(axes[k]));
        domain.upper[k] = jnum(box.at("max").at(axes[k]));
    }
    // ROI：投影坐标经 Normalizer（只支持无旋转）换到本地坐标。
    if (!roi.empty() && std::filesystem::is_regular_file(roi) && value.contains("Normalizer"))
    {
        const Json r = read_json_file(roi);
        if (r.contains("projected") && r["projected"].contains("points") && r["projected"]["points"].size() >= 3)
        {
            const Json& transform = value.at("Normalizer").at("transform");
            const Json& q = transform.at("rotation");
            if (std::abs(jnum(q.at("w")) - 1) > 1e-9 || std::abs(jnum(q.at("x"))) > 1e-9 ||
                std::abs(jnum(q.at("y"))) > 1e-9 || std::abs(jnum(q.at("z"))) > 1e-9)
            {
                log("[DOMAIN] Normalizer has a rotation; roi.json is ignored");
            }
            else
            {
                const double scale = jnum(transform.at("scale"));
                const double tx = jnum(transform.at("translation").at("x")), ty = jnum(transform.at("translation").at("y"));
                for (const Json& point : r["projected"]["points"])
                {
                    domain.polygon.push_back({jnum(point[0]) * scale + tx, jnum(point[1]) * scale + ty});
                }
            }
        }
    }
    return domain;
}

bool Domain::inside(double x, double y) const
{
    if (x < lower[0] || x > upper[0] || y < lower[1] || y > upper[1])
    {
        return false;
    }
    if (!tiles.empty())
    {
        bool hit = false;
        for (const auto& t : tiles)
        {
            hit |= x >= t[0] && x <= t[2] && y >= t[1] && y <= t[3];
        }
        if (!hit)
        {
            return false;
        }
    }
    return polygon.empty() || polygon_distance(x, y) > 0;
}

double Domain::polygon_distance(double x, double y) const
{
    if (polygon.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    bool in = false;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
    {
        const auto& a = polygon[j];
        const auto& b = polygon[i];
        const double abx = b[0] - a[0], aby = b[1] - a[1];
        const double length = abx * abx + aby * aby;
        if (length > 0)
        {
            const double t = std::min(1.0, std::max(0.0, ((x - a[0]) * abx + (y - a[1]) * aby) / length));
            best = std::min(best, std::hypot(x - (a[0] + t * abx), y - (a[1] + t * aby)));
        }
        if ((a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
        {
            in = !in;
        }
    }
    return in ? best : -best;
}

Json Domain::json() const
{
    Json v;
    v["lower"] = lower;
    v["upper"] = upper;
    v["polygon"] = polygon;
    v["tiles"] = tiles;
    return v;
}

Domain Domain::from_json(const Json& v)
{
    Domain d;
    for (int k = 0; k < 3; ++k)
    {
        d.lower[k] = jnum(v.at("lower")[static_cast<std::size_t>(k)]);
        d.upper[k] = jnum(v.at("upper")[static_cast<std::size_t>(k)]);
    }
    if (v.contains("polygon"))
    {
        for (const Json& p : v["polygon"])
        {
            d.polygon.push_back({jnum(p[0]), jnum(p[1])});
        }
    }
    if (v.contains("tiles"))
    {
        for (const Json& t : v["tiles"])
        {
            d.tiles.push_back({jnum(t[0]), jnum(t[1]), jnum(t[2]), jnum(t[3])});
        }
    }
    return d;
}

Json view_record(const View& view)
{
    Json record;
    record["image_id"] = view.image_id;
    record["name"] = view.name;
    record["camera_id"] = view.camera_id;
    record["model"] = view.model;
    record["model_id"] = view.model_id;
    record["width"] = view.width;
    record["height"] = view.height;
    record["params"] = view.params;
    record["r_wc"] = view.r_wc;
    record["t_wc"] = view.t_wc;
    record["center"] = view.center;
    record["mask_path"] = path_utf8(view.mask_path);
    record["mask_key"] = view.mask_key;
    if (view.distorted())
    {
        record["raw_model_id"] = view.raw_model_id;
        record["raw_params"] = view.raw_params;
    }
    return record;
}

View view_from_record(const Json& record)
{
    View view;
    view.image_id = record.at("image_id").get<std::uint32_t>();
    view.name = record.at("name").get<std::string>();
    view.camera_id = record.at("camera_id").get<std::uint32_t>();
    view.model = record.at("model").get<std::string>();
    view.model_id = record.at("model_id").get<int>();
    view.width = record.at("width").get<int>();
    view.height = record.at("height").get<int>();
    for (const Json& p : record.at("params"))
    {
        view.params.push_back(jnum(p));
    }
    for (int k = 0; k < 9; ++k)
    {
        view.r_wc[static_cast<std::size_t>(k)] = jnum(record.at("r_wc")[static_cast<std::size_t>(k)]);
    }
    for (int k = 0; k < 3; ++k)
    {
        view.t_wc[static_cast<std::size_t>(k)] = jnum(record.at("t_wc")[static_cast<std::size_t>(k)]);
        view.center[static_cast<std::size_t>(k)] = jnum(record.at("center")[static_cast<std::size_t>(k)]);
    }
    view.mask_path = utf8_path(record.at("mask_path").get<std::string>());
    view.mask_key = record.at("mask_key").get<std::string>();
    if (record.contains("raw_model_id"))
    {
        view.raw_model_id = record.at("raw_model_id").get<int>();
        for (const Json& p : record.at("raw_params"))
        {
            view.raw_params.push_back(jnum(p));
        }
    }
    return view;
}

Json config_to_json(const Config& c)
{
    Json v;
    v["project"] = path_utf8(c.project);
    v["lod"] = path_utf8(c.lod);
    v["mask_dir"] = path_utf8(c.mask_dir);
    v["work"] = path_utf8(c.work);
    v["output"] = path_utf8(c.output);
    v["mode"] = c.mode;
    v["poses"] = path_utf8(c.poses);
    v["cameras"] = path_utf8(c.cameras);
    v["images"] = path_utf8(c.images);
    v["image_root"] = c.image_root;
    v["pose_crs"] = c.pose_crs;
    v["bbox"] = path_utf8(c.bbox);
    v["roi"] = path_utf8(c.roi);
    v["view_margin"] = c.view_margin;
    v["min_view_leaves"] = c.min_view_leaves;
    v["water_class"] = c.water_class;
    v["cell_size"] = c.cell_size;
    v["spacing"] = c.spacing;
    v["footprint_spacing"] = c.footprint_spacing;
    v["debug"] = c.debug;
    v["water_carve"] = c.water_carve;
    v["carve_erosion_cells"] = c.carve_erosion_cells;
    v["carve_surface_tolerance"] = c.carve_surface_tolerance;
    v["carve_above_band"] = c.carve_above_band;
    v["carve_max_depth"] = c.carve_max_depth;
    v["carve_below_margin"] = c.carve_below_margin;
    v["carve_below_reach"] = c.carve_below_reach;
    v["carve_float_band"] = c.carve_float_band;
    v["carve_float_visible_water"] = c.carve_float_visible_water;
    v["carve_bank_visible_water"] = c.carve_bank_visible_water;
    v["carve_float_max"] = c.carve_float_max;
    v["carve_float_geometry_water"] = c.carve_float_geometry_water;
    v["denoise"] = c.denoise;
    v["denoise_method"] = c.denoise_method;
    v["denoise_k"] = c.denoise_k;
    v["denoise_std_ratio"] = c.denoise_std_ratio;
    v["denoise_max_distance"] = c.denoise_max_distance;
    v["denoise_min_neighbors"] = c.denoise_min_neighbors;
    v["denoise_radius"] = c.denoise_radius;
    v["denoise_min_points"] = c.denoise_min_points;
    return v;
}

Config config_from_json(const Json& v)
{
    Config c;
    c.project = utf8_path(v.at("project").get<std::string>());
    c.lod = utf8_path(v.at("lod").get<std::string>());
    c.mask_dir = utf8_path(v.at("mask_dir").get<std::string>());
    c.work = utf8_path(v.at("work").get<std::string>());
    c.output = utf8_path(v.value("output", std::string()));
    c.mode = v.value("mode", std::string("blocks"));
    c.poses = utf8_path(v.value("poses", std::string()));
    c.cameras = utf8_path(v.value("cameras", std::string()));
    c.images = utf8_path(v.value("images", std::string()));
    c.image_root = v.value("image_root", std::string());
    c.pose_crs = v.value("pose_crs", std::string("auto"));
    c.bbox = utf8_path(v.value("bbox", std::string()));
    c.roi = utf8_path(v.value("roi", std::string()));
    c.view_margin = v.contains("view_margin") ? jnum(v.at("view_margin")) : 0.5;
    c.min_view_leaves = v.value("min_view_leaves", 1);
    c.water_class = v.at("water_class").get<int>();
    c.cell_size = jnum(v.at("cell_size"));
    c.spacing = jnum(v.at("spacing"));
    c.footprint_spacing = jnum(v.at("footprint_spacing"));
    c.debug = v.value("debug", false);
    c.water_carve = v.value("water_carve", true);
    c.carve_erosion_cells = v.value("carve_erosion_cells", 2);
    c.carve_surface_tolerance = jnum(v.at("carve_surface_tolerance"));
    c.carve_above_band = jnum(v.at("carve_above_band"));
    c.carve_max_depth = jnum(v.at("carve_max_depth"));
    c.carve_below_margin = jnum(v.at("carve_below_margin"));
    c.carve_below_reach = jnum(v.at("carve_below_reach"));
    c.carve_float_band = jnum(v.at("carve_float_band"));
    c.carve_float_visible_water = jnum(v.at("carve_float_visible_water"));
    c.carve_bank_visible_water = jnum(v.at("carve_bank_visible_water"));
    c.carve_float_max = jnum(v.at("carve_float_max"));
    c.carve_float_geometry_water = jnum(v.at("carve_float_geometry_water"));
    c.denoise = v.value("denoise", true);
    c.denoise_radius = v.contains("denoise_radius") ? jnum(v.at("denoise_radius")) : 0.5;
    c.denoise_min_points = v.value("denoise_min_points", 30);
    c.denoise_method = v.value("denoise_method", std::string("cluster"));
    c.denoise_k = v.value("denoise_k", 20);
    c.denoise_std_ratio = v.contains("denoise_std_ratio") ? jnum(v.at("denoise_std_ratio")) : 2.0;
    c.denoise_max_distance = v.contains("denoise_max_distance") ? jnum(v.at("denoise_max_distance")) : 2.0;
    c.denoise_min_neighbors = v.value("denoise_min_neighbors", 5);
    return c;
}

void save_config(const Config& config)
{
    std::filesystem::create_directories(config.work);
    write_json_atomic(config.work / "config.json", config_to_json(config), false);
}

Config load_config(const std::filesystem::path& work)
{
    const std::filesystem::path path = work / "config.json";
    if (!std::filesystem::is_regular_file(path))
    {
        fail_missing("Missing work/config.json; run assign (or all) first: " + path_utf8(path));
    }
    Config config = config_from_json(read_json_file(path));
    config.work = work;
    return config;
}

bool view_sees_box(const View& view, const Vec3& low, const Vec3& high, double margin)
{
    double umin = std::numeric_limits<double>::infinity(), umax = -umin, vmin = umin, vmax = -umin;
    for (int c = 0; c < 8; ++c)
    {
        const double p[3] = {(c & 1) ? high[0] : low[0], (c & 2) ? high[1] : low[1], (c & 4) ? high[2] : low[2]};
        double u = 0, v = 0;
        if (!project_point(view, p, u, v))
        {
            return true;
        }
        umin = std::min(umin, u);
        umax = std::max(umax, u);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    const double mx = margin * view.width, my = margin * view.height;
    return !(umax < -mx || vmax < -my || umin > view.width + mx || vmin > view.height + my);
}

double quantile(std::vector<double> values, double q)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(values.size() - 1, lo + 1);
    const double fraction = position - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * fraction;
}

double median(std::vector<double> values)
{
    return quantile(std::move(values), 0.5);
}

} // namespace ai3d::waterpost
