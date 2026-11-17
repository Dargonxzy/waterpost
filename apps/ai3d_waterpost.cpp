// 命令行入口：ai3d_waterpost <阶段> [选项]
//
// 阶段：all | split | assign | scene | detect | reconcile | candidates | active | samples | model | write | finalize
//   按图像清单（可分机，split 写清单，assign --list 选清单）：assign
//   每块（可分机，--tile-name 选块）：detect / candidates / samples / write
//   全局（单机）：split / scene / reconcile / active / model / finalize
// mode：blocks（有分块成果：块的相机与范围来自 artifacts/blocks，高斯来自 LOD 最细层）；
//       poses（没有分块成果：LOD + 位姿 JSON + 掩码，块取 LOD 中的块，每块的视图由 assign 按可见性确定）。

#include "error.hpp"
#include "fs_util.hpp"
#include "parallel.hpp"
#include "project.hpp"
#include "scene.hpp"
#include "stages.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <shellapi.h>
#endif

namespace
{

using namespace ai3d::waterpost;

std::vector<std::string> arguments(int argc, char** argv)
{
    std::vector<std::string> result;
#ifdef _WIN32
    (void)argv;
    (void)argc;
    int count = 0;
    LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &count);
    for (int index = 1; index < count; ++index)
    {
        const int size = WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, nullptr, 0, nullptr, nullptr);
        std::string text(static_cast<std::size_t>(size > 0 ? size - 1 : 0), '\0');
        if (size > 1)
        {
            WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, text.data(), size, nullptr, nullptr);
        }
        result.push_back(std::move(text));
    }
    LocalFree(wide);
#else
    for (int index = 1; index < argc; ++index)
    {
        result.emplace_back(argv[index]);
    }
#endif
    return result;
}

const char* HELP =
    "usage: ai3d_waterpost <stage> --work DIR [options]\n"
    "       ai3d_waterpost evaluate (--project DIR | (--poses FILE | --cameras FILE --images FILE) --mask-dir DIR)\n"
    "                               [--photos DIR] [--image-root PREFIX] [--pose-crs auto|none|EPSG:326NN] --out CSV\n"
    "                               --variant name=LOD [--variant ...] [--step N] [--views a.jpg,b.jpg] [--gpu-id N]\n"
    "stages: all split assign scene detect reconcile candidates active samples model write finalize\n"
    "  --mode blocks|poses  blocks: cameras/blocks from <project>/artifacts/blocks; poses: LOD + --poses JSON\n"
    "  --project DIR        task directory (artifacts/blocks, bbox.json); required in blocks mode\n"
    "  --poses FILE         camera pose JSON with cameras and images (poses mode)\n"
    "  --cameras FILE --images FILE  or cameras / images in two JSON files (objects keyed by id, or arrays)\n"
    "  --image-root PREFIX  mask = <mask-dir>/<image name without PREFIX>.png (default: found automatically)\n"
    "  --pose-crs auto|none|EPSG:326NN  poses in UTM are shifted into LOD coordinates via the LOD root transform\n"
    "  --bbox FILE --roi FILE  fill domain in poses mode (default: LOD extent, no ROI)\n"
    "  --view-margin X      poses: camera XY may lie X * block size outside the block (default 0.5)\n"
    "  --min-view-leaves N  poses: a view belongs to a block if it sees >= N finest leaves (default 1)\n"
    "  --parts N            split: write N image lists to work/assign/lists/part_XX.list\n"
    "  --split interleave|contiguous  split: image k goes to list k mod N (default) or contiguous runs\n"
    "  --list FILE          assign: process only the images in FILE (one name per line; default: all)\n"
    "  --lod DIR            input BIN LOD (default <project>/results/3dgs/lod/bin)\n"
    "  --mask-dir DIR       semantic masks (default <project>/.temp/semantic/masks)\n"
    "  --output DIR         output LOD directory (required before write/finalize)\n"
    "  --work DIR           intermediate files shared by all stages\n"
    "  --tile-name NAME     restrict a per-block stage to this block (repeatable)\n"
    "  --threads N --gpu-id N --debug --link-output --water-class N\n"
    "  --cell-size X --spacing X --footprint-spacing X\n"
    "  --no-water-carve --carve-erosion-cells N --carve-surface-tolerance X --carve-above-band X\n"
    "  --carve-max-depth X --carve-below-margin X --carve-below-reach X --carve-float-band X\n"
    "  --carve-float-visible-water X --carve-bank-visible-water X --carve-float-max X --carve-float-geometry-water X\n"
    "  --denoise 0|1 --denoise-method sor|radius|cluster --denoise-k N --denoise-std-ratio X\n"
    "  --denoise-max-distance X --denoise-radius X --denoise-min-neighbors N --denoise-min-points N\n";

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const std::vector<std::string> args = arguments(argc, argv);
    if (args.empty() || args[0] == "-h" || args[0] == "--help")
    {
        std::printf("%s", HELP);
        return args.empty() ? 2 : 0;
    }
    const std::string stage = args[0];
    if (stage == "evaluate")
    {
        // evaluate (--project P | --poses F | --cameras C --images I) [--photos DIR] [--mask-dir M] --out CSV --variant name=LOD ...
        try
        {
            std::filesystem::path project, images, masks, csv, poses, cameras, image_list;
            std::string image_root, pose_crs = "auto";
            int step = 10, gpu = 0, water_class = 9;
            std::vector<std::pair<std::string, std::filesystem::path>> variants;
            std::vector<std::string> only;
            for (std::size_t i = 1; i + 1 < args.size(); i += 2)
            {
                const std::string& key = args[i];
                const std::string& value = args[i + 1];
                if (key == "--project") project = utf8_path(value);
                else if (key == "--photos") images = utf8_path(value);
                else if (key == "--images")
                {
                    // 与主流程一致：--images 是 images.json；给的是目录时按旧用法当照片目录。
                    if (std::filesystem::is_directory(utf8_path(value))) images = utf8_path(value);
                    else image_list = utf8_path(value);
                }
                else if (key == "--cameras") cameras = utf8_path(value);
                else if (key == "--image-root") image_root = value;
                else if (key == "--pose-crs") pose_crs = value;
                else if (key == "--mask-dir") masks = utf8_path(value);
                else if (key == "--poses") poses = utf8_path(value);
                else if (key == "--out") csv = utf8_path(value);
                else if (key == "--step") step = std::stoi(value);
                else if (key == "--gpu-id") gpu = std::stoi(value);
                else if (key == "--water-class") water_class = std::stoi(value);
                else if (key == "--views")
                {
                    std::size_t start = 0;
                    while (start <= value.size())
                    {
                        const std::size_t comma = std::min(value.find(',', start), value.size());
                        if (comma > start) only.push_back(value.substr(start, comma - start));
                        start = comma + 1;
                    }
                }
                else if (key == "--variant")
                {
                    const std::size_t eq = value.find('=');
                    if (eq == std::string::npos || eq == 0)
                    {
                        fail_value("--variant expects name=LOD_DIR: " + value);
                    }
                    variants.emplace_back(value.substr(0, eq), utf8_path(value.substr(eq + 1)));
                }
                else fail_value("Unknown evaluate option " + key);
            }
            if (masks.empty()) masks = project / ".temp" / "semantic" / "masks";
            if (images.empty()) images = project / ".temp" / "undistortion" / "images";
            Config pose_config;
            const bool posed = !poses.empty() || !cameras.empty() || !image_list.empty();
            if (posed)
            {
                if (variants.empty()) fail_value("evaluate needs at least one --variant");
                pose_config.poses = poses;
                pose_config.cameras = cameras;
                pose_config.images = image_list;
                pose_config.image_root = image_root;
                pose_config.pose_crs = pose_crs;
                pose_config.mask_dir = masks;
                pose_config.lod = variants.front().second; // 坐标换算与检查用第一个 LOD 的地理参考
            }
            run_evaluate(project, variants, masks, images, step, csv, gpu, water_class, only, posed ? &pose_config : nullptr);
            return 0;
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "[FAILED] %s\n", error.what());
            return 1;
        }
    }
    const std::set<std::string> stages = {"all",    "split",   "assign", "scene", "detect",  "reconcile",
                                          "candidates", "active", "samples", "model", "write", "finalize"};
    if (!stages.count(stage))
    {
        std::fprintf(stderr, "%sunknown stage: %s\n", HELP, stage.c_str());
        return 2;
    }
    try
    {
        Config config;
        std::vector<std::string> tiles;
        std::string project, lod, masks, work, output, mode, poses, bbox, roi, list, split = "interleave";
        std::string cameras, images, image_root, pose_crs;
        int threads = -1, gpu = -1, parts = 0;
        bool debug = false, link = false;
        std::vector<std::pair<std::string, std::string>> overrides;
        for (std::size_t i = 1; i < args.size(); ++i)
        {
            const std::string& key = args[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= args.size())
                {
                    fail_value("Missing value for " + key);
                }
                return args[++i];
            };
            if (key == "--project") project = value();
            else if (key == "--lod") lod = value();
            else if (key == "--mask-dir") masks = value();
            else if (key == "--work") work = value();
            else if (key == "--output") output = value();
            else if (key == "--mode") mode = value();
            else if (key == "--poses") poses = value();
            else if (key == "--cameras") cameras = value();
            else if (key == "--images") images = value();
            else if (key == "--image-root") image_root = value();
            else if (key == "--pose-crs") pose_crs = value();
            else if (key == "--bbox") bbox = value();
            else if (key == "--roi") roi = value();
            else if (key == "--list") list = value();
            else if (key == "--parts") parts = std::stoi(value());
            else if (key == "--split") split = value();
            else if (key == "--tile-name") tiles.push_back(value());
            else if (key == "--threads") threads = std::stoi(value());
            else if (key == "--gpu-id") gpu = std::stoi(value());
            else if (key == "--debug") debug = true;
            else if (key == "--link-output") link = true;
            else if (key == "--denoise-method") overrides.emplace_back("denoise_method:" + value(), "0");
            else if (key == "--no-water-carve") overrides.emplace_back("water_carve", "0");
            else if (key.rfind("--", 0) == 0)
            {
                std::string name = key.substr(2);
                for (char& c : name)
                {
                    c = c == '-' ? '_' : c;
                }
                overrides.emplace_back(name, value());
            }
            else
            {
                fail_value("Unexpected argument: " + key);
            }
        }
        if (work.empty())
        {
            fail_value("--work is required");
        }
        const std::filesystem::path work_dir = std::filesystem::absolute(utf8_path(work));
        // 第一次运行时建 config.json：all / split / assign，以及 blocks 模式直接从 detect 开始（不跑 split / assign / scene）。
        const bool fresh = stage == "all" || stage == "split" || stage == "assign" ||
                           (stage == "detect" && (mode.empty() || mode == "blocks"));
        if (fresh && !std::filesystem::is_regular_file(work_dir / "config.json"))
        {
            config.mode = mode.empty() ? "blocks" : mode;
            if (!project.empty())
            {
                config.project = std::filesystem::absolute(utf8_path(project));
            }
            if (config.mode == "blocks" && project.empty())
            {
                fail_value("--project is required for the first run in blocks mode");
            }
            if (config.mode == "poses" && poses.empty() && (cameras.empty() || images.empty()))
            {
                fail_value("poses mode needs --poses FILE, or --cameras FILE with --images FILE, on the first run");
            }
            if (project.empty() && (lod.empty() || masks.empty()))
            {
                fail_value("--lod and --mask-dir are required without --project");
            }
            config.lod = lod.empty() ? config.project / "results" / "3dgs" / "lod" / "bin" : std::filesystem::absolute(utf8_path(lod));
            config.mask_dir = masks.empty() ? config.project / ".temp" / "semantic" / "masks" : std::filesystem::absolute(utf8_path(masks));
            config.work = work_dir;
        }
        else
        {
            config = load_config(work_dir);
            // work 目录已经有 config.json：输入只在第一次运行时确定。再传不同的输入说明想处理另一份数据，
            // 静默沿用旧路径会算错，直接报错。
            auto same = [](const std::filesystem::path& a, const std::string& b) {
                return canonical_path_text(a) == canonical_path_text(std::filesystem::absolute(utf8_path(b)));
            };
            auto check = [&](const std::string& given, const std::filesystem::path& stored, const char* option) {
                if (!given.empty() && !same(stored, given))
                {
                    fail_value(std::string(option) + " differs from " + path_utf8(work_dir / "config.json") + " (" +
                               path_utf8(stored) + "); use a new --work directory for different inputs");
                }
            };
            if (!config.project.empty()) check(project, config.project, "--project");
            check(lod, config.lod, "--lod");
            check(masks, config.mask_dir, "--mask-dir");
            if (!mode.empty() && mode != config.mode)
            {
                fail_value("--mode " + mode + " differs from config.json (" + config.mode + "); use a new --work directory");
            }
            if (!config.poses.empty()) check(poses, config.poses, "--poses");
            if (!config.cameras.empty()) check(cameras, config.cameras, "--cameras");
            if (!config.images.empty()) check(images, config.images, "--images");
            if (!image_root.empty() && !config.image_root.empty() && image_root != config.image_root)
            {
                fail_value("--image-root differs from config.json; use a new --work directory");
            }
            if (!pose_crs.empty() && pose_crs != config.pose_crs)
            {
                fail_value("--pose-crs differs from config.json (" + config.pose_crs + "); use a new --work directory");
            }
            if (!config.bbox.empty()) check(bbox, config.bbox, "--bbox");
            if (!config.roi.empty()) check(roi, config.roi, "--roi");
        }
        if (!poses.empty()) config.poses = std::filesystem::absolute(utf8_path(poses));
        if (!cameras.empty()) config.cameras = std::filesystem::absolute(utf8_path(cameras));
        if (!images.empty()) config.images = std::filesystem::absolute(utf8_path(images));
        if (!image_root.empty()) config.image_root = image_root;
        if (!pose_crs.empty()) config.pose_crs = pose_crs;
        if (!bbox.empty()) config.bbox = std::filesystem::absolute(utf8_path(bbox));
        if (!roi.empty()) config.roi = std::filesystem::absolute(utf8_path(roi));
        if (!output.empty())
        {
            config.output = std::filesystem::absolute(utf8_path(output));
        }
        if (debug)
        {
            config.debug = true;
        }
        config.link_output = link;
        config.threads = threads >= 0 ? threads : 0;
        config.gpu = gpu >= 0 ? gpu : 0;
        for (const auto& [name, text] : overrides)
        {
            if (name.rfind("denoise_method:", 0) == 0)
            {
                config.denoise_method = name.substr(15);
                continue;
            }
            const double v = std::stod(text);
            if (name == "water_class") config.water_class = static_cast<int>(v);
            else if (name == "cell_size") config.cell_size = v;
            else if (name == "spacing") config.spacing = v;
            else if (name == "footprint_spacing") config.footprint_spacing = v;
            else if (name == "water_carve") config.water_carve = v != 0;
            else if (name == "carve_erosion_cells") config.carve_erosion_cells = static_cast<int>(v);
            else if (name == "carve_surface_tolerance") config.carve_surface_tolerance = v;
            else if (name == "carve_above_band") config.carve_above_band = v;
            else if (name == "carve_max_depth") config.carve_max_depth = v;
            else if (name == "carve_below_margin") config.carve_below_margin = v;
            else if (name == "carve_below_reach") config.carve_below_reach = v;
            else if (name == "carve_float_band") config.carve_float_band = v;
            else if (name == "carve_float_visible_water") config.carve_float_visible_water = v;
            else if (name == "carve_bank_visible_water") config.carve_bank_visible_water = v;
            else if (name == "carve_float_max") config.carve_float_max = v;
            else if (name == "carve_float_geometry_water") config.carve_float_geometry_water = v;
            else if (name == "denoise") config.denoise = v != 0;
            else if (name == "denoise_radius") config.denoise_radius = v;
            else if (name == "denoise_min_points") config.denoise_min_points = static_cast<int>(v);
            else if (name == "denoise_k") config.denoise_k = static_cast<int>(v);
            else if (name == "denoise_std_ratio") config.denoise_std_ratio = v;
            else if (name == "denoise_max_distance") config.denoise_max_distance = v;
            else if (name == "denoise_min_neighbors") config.denoise_min_neighbors = static_cast<int>(v);
            else if (name == "view_margin") config.view_margin = v;
            else if (name == "min_view_leaves") config.min_view_leaves = static_cast<int>(v);
            else fail_value("Unknown option --" + name);
        }
        if (config.mode != "blocks" && config.mode != "poses")
        {
            fail_value("Unknown mode '" + config.mode + "'; use --mode blocks or --mode poses");
        }
        set_thread_count(config.threads);
        save_config(config);
        const Timer total;
        auto needs_output = [&]() {
            if (config.output.empty())
            {
                fail_value("--output is required for write/finalize");
            }
        };
        auto per_block = [&](void (*function)(const Config&, const std::string&), const char* label) {
            // 块清单来自 work/scene（scene 阶段之后才有）。
            const std::vector<std::string> known = pipeline_blocks(config);
            for (const std::string& tile : tiles)
            {
                if (std::find(known.begin(), known.end(), tile) == known.end())
                {
                    std::string list;
                    for (const std::string& name : known)
                    {
                        list += (list.empty() ? "" : ", ") + name;
                    }
                    fail_value("Unknown --tile-name " + tile + "; blocks: " + list);
                }
            }
            const std::vector<std::string> blocks = tiles.empty() ? known : tiles;
            for (const std::string& block : blocks)
            {
                const Timer timer;
                function(config, block);
                log("[TIME] %s %s %.1fs", label, block.c_str(), timer.seconds());
            }
        };
        if (stage == "all")
        {
            needs_output();
            clear_assign_results(config);
            run_assign(config, {});
            run_scene(config);
            per_block(run_detect, "detect");
            run_reconcile(config);
            per_block(run_candidates, "candidates");
            run_active(config);
            per_block(run_samples, "samples");
            run_model(config);
            per_block(run_write, "write");
            run_finalize(config);
        }
        else if (stage == "split")
        {
            if (parts < 1) fail_value("split needs --parts N");
            if (split != "interleave" && split != "contiguous") fail_value("--split must be interleave or contiguous");
            run_split(config, parts, split == "interleave");
        }
        else if (stage == "assign") run_assign(config, list.empty() ? std::filesystem::path() : std::filesystem::absolute(utf8_path(list)));
        else if (stage == "scene") run_scene(config);
        else if (stage == "detect") per_block(run_detect, "detect");
        else if (stage == "reconcile") run_reconcile(config);
        else if (stage == "candidates") per_block(run_candidates, "candidates");
        else if (stage == "active") run_active(config);
        else if (stage == "samples") per_block(run_samples, "samples");
        else if (stage == "model") run_model(config);
        else if (stage == "write")
        {
            needs_output();
            per_block(run_write, "write");
        }
        else if (stage == "finalize")
        {
            needs_output();
            run_finalize(config);
        }
        log("[DONE] %s %.1fs", stage.c_str(), total.seconds());
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fflush(stdout);
        std::fprintf(stderr, "[FAILED] %s\n", error.what());
        return 1;
    }
}
