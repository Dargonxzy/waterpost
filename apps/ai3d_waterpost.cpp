// 命令行入口：ai3d_waterpost <阶段> [选项]
//
// 阶段：all | detect | reconcile | candidates | active | samples | model | write | finalize
//   每块（可分机，--tile-name 选块）：detect / candidates / samples / write
//   全局（单机）：reconcile / active / model / finalize
// mode：blocks（有分块成果：块的相机与范围来自 artifacts/blocks，高斯来自 LOD 最细层）。

#include "error.hpp"
#include "fs_util.hpp"
#include "parallel.hpp"
#include "project.hpp"
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
    "stages: all detect reconcile candidates active samples model write finalize\n"
    "  --project DIR        task directory (artifacts/blocks, bbox.json); required for detect/all\n"
    "  --lod DIR            input BIN LOD (default <project>/results/3dgs/lod/bin)\n"
    "  --mask-dir DIR       semantic masks (default <project>/.temp/semantic/masks)\n"
    "  --output DIR         output LOD directory (required before write/finalize)\n"
    "  --work DIR           intermediate files shared by all stages\n"
    "  --mode blocks        processing mode (only 'blocks' is implemented)\n"
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
        // evaluate --project P --images DIR [--mask-dir M] --out CSV [--step N] --variant name=LOD ...
        try
        {
            std::filesystem::path project, images, masks, csv;
            int step = 10, gpu = 0, water_class = 9;
            std::vector<std::pair<std::string, std::filesystem::path>> variants;
            std::vector<std::string> only;
            for (std::size_t i = 1; i + 1 < args.size(); i += 2)
            {
                const std::string& key = args[i];
                const std::string& value = args[i + 1];
                if (key == "--project") project = utf8_path(value);
                else if (key == "--images") images = utf8_path(value);
                else if (key == "--mask-dir") masks = utf8_path(value);
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
                    variants.emplace_back(value.substr(0, eq), utf8_path(value.substr(eq + 1)));
                }
                else fail_value("Unknown evaluate option " + key);
            }
            if (masks.empty()) masks = project / ".temp" / "semantic" / "masks";
            if (images.empty()) images = project / ".temp" / "undistortion" / "images";
            run_evaluate(project, variants, masks, images, step, csv, gpu, water_class, only);
            return 0;
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "[FAILED] %s\n", error.what());
            return 1;
        }
    }
    const std::set<std::string> stages = {"all", "detect", "reconcile", "candidates", "active", "samples", "model", "write", "finalize"};
    if (!stages.count(stage))
    {
        std::fprintf(stderr, "%sunknown stage: %s\n", HELP, stage.c_str());
        return 2;
    }
    try
    {
        Config config;
        std::vector<std::string> tiles;
        std::string project, lod, masks, work, output, mode;
        int threads = -1, gpu = -1;
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
        const bool fresh = stage == "all" || stage == "detect";
        if (fresh && !std::filesystem::is_regular_file(work_dir / "config.json"))
        {
            if (project.empty())
            {
                fail_value("--project is required for the first detect/all run");
            }
            config.project = std::filesystem::absolute(utf8_path(project));
            config.lod = lod.empty() ? config.project / "results" / "3dgs" / "lod" / "bin" : std::filesystem::absolute(utf8_path(lod));
            config.mask_dir = masks.empty() ? config.project / ".temp" / "semantic" / "masks" : std::filesystem::absolute(utf8_path(masks));
            config.work = work_dir;
        }
        else
        {
            config = load_config(work_dir);
        }
        if (!output.empty())
        {
            config.output = std::filesystem::absolute(utf8_path(output));
        }
        if (!mode.empty())
        {
            config.mode = mode;
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
            else fail_value("Unknown option --" + name);
        }
        if (config.mode != "blocks")
        {
            fail_value("Mode '" + config.mode + "' is not implemented yet; use --mode blocks");
        }
        set_thread_count(config.threads);
        save_config(config);
        const Timer total;
        const std::vector<std::string> all_blocks = pipeline_blocks(config);
        const std::vector<std::string> blocks = tiles.empty() ? all_blocks : tiles;
        auto needs_output = [&]() {
            if (config.output.empty())
            {
                fail_value("--output is required for write/finalize");
            }
        };
        auto per_block = [&](void (*function)(const Config&, const std::string&), const char* label) {
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
            per_block(run_detect, "detect");
            run_reconcile(config);
            per_block(run_candidates, "candidates");
            run_active(config);
            per_block(run_samples, "samples");
            run_model(config);
            per_block(run_write, "write");
            run_finalize(config);
        }
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
