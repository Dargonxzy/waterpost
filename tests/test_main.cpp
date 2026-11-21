// 纯函数单元测试：半精度转换、POSIX 路径工具、npz 读写往返。失败时返回非 0。

#include "floaters.hpp"
#include "lod.hpp"
#include "npz.hpp"
#include "npz_util.hpp"
#include "project.hpp"
#include "shore.hpp"
#include "geo.hpp"
#include "masks.hpp"
#include "fs_util.hpp"
#include "json_util.hpp"
#include "parallel.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <limits>
#include <random>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{

using namespace ai3d::waterpost;

int g_failures = 0;

#define EXPECT(condition, ...)                                                                                          \
    do                                                                                                                  \
    {                                                                                                                   \
        if (!(condition))                                                                                               \
        {                                                                                                               \
            ++g_failures;                                                                                               \
            if (g_failures <= 20)                                                                                       \
            {                                                                                                           \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                        \
                std::printf(__VA_ARGS__);                                                                               \
                std::printf("\n");                                                                                      \
            }                                                                                                           \
        }                                                                                                               \
    } while (0)

float bits_to_float(std::uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

// 参照实现：用 double 精确算出最近的半精度（偶数舍入），>= 65520 夹到最大有限值（与实现约定一致）。
std::uint16_t reference_half(float value)
{
    const double v = std::fabs(static_cast<double>(value));
    const std::uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
    if (v >= 65520.0)
    {
        return static_cast<std::uint16_t>(sign | 0x7bffu);
    }
    if (v < std::ldexp(1.0, -14))
    {
        const double q = std::nearbyint(std::ldexp(v, 24)); // 次正规：值 = q * 2^-24（q = 1024 时正好是最小正规数）
        return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(q));
    }
    int e = 0;
    std::frexp(v, &e); // v = m * 2^e, m ∈ [0.5, 1)
    int exponent = e - 1;
    double q = std::nearbyint(std::ldexp(v, 10 - exponent)); // ∈ [1024, 2048]
    if (q >= 2048)
    {
        q = 1024;
        ++exponent;
    }
    return static_cast<std::uint16_t>(sign | ((exponent + 15) << 10) | (static_cast<int>(q) - 1024));
}

void test_float_to_half()
{
    std::size_t checked = 0, mismatched = 0;
    auto check = [&](std::uint32_t bits) {
        const float value = bits_to_float(bits);
        const std::uint16_t got = float_to_half(value), want = reference_half(value);
        ++checked;
        if (got != want)
        {
            ++mismatched;
            EXPECT(false, "float_to_half(%.9g) = 0x%04x, want 0x%04x", static_cast<double>(value), got, want);
        }
    };
    for (std::uint32_t bits = 0x33000000u; bits < 0x38800000u; ++bits) // 次正规半精度范围：逐个检查
    {
        check(bits);
        check(bits | 0x80000000u);
    }
    for (std::uint32_t bits = 0x38800000u; bits < 0x47800000u; bits += 7) // 正规范围与溢出：抽样
    {
        check(bits);
    }
    for (std::uint32_t bits = 0; bits < 0x33000000u; bits += 0x10001u) // 下溢到 0
    {
        check(bits);
    }
    std::printf("float_to_half: %zu values checked, %zu mismatches\n", checked, mismatched);

    // 半精度 -> float -> 半精度 必须原样返回（全部有限值）。
    std::size_t round_trip = 0;
    for (std::uint32_t h = 0; h < 0x10000u; ++h)
    {
        if ((h & 0x7c00u) == 0x7c00u)
        {
            continue;
        }
        const std::uint16_t back = float_to_half(half_to_float(static_cast<std::uint16_t>(h)));
        if (back != h)
        {
            ++round_trip;
            EXPECT(false, "round trip 0x%04x -> 0x%04x", h, back);
        }
    }
    EXPECT(float_to_half(INFINITY) == 0x7c00u, "inf");
    EXPECT((float_to_half(NAN) & 0x7c00u) == 0x7c00u && (float_to_half(NAN) & 0x3ffu) != 0, "nan");
    EXPECT(float_to_half(1e6f) == 0x7bffu, "overflow clamps to the largest finite half");
    std::printf("half round trip: %zu mismatches\n", round_trip);
}

void test_posix_paths()
{
    EXPECT(normalize_posix("a/b/../c") == "a/c", "normalize a/b/../c -> %s", normalize_posix("a/b/../c").c_str());
    EXPECT(normalize_posix("./a/./b") == "a/b", "normalize ./a/./b -> %s", normalize_posix("./a/./b").c_str());
    EXPECT(dirname_posix("L22/T/a.bin") == "L22/T", "dirname -> %s", dirname_posix("L22/T/a.bin").c_str());
    EXPECT(relative_posix("L22/T", "L22/T/a.bin") == "a.bin", "relative same dir -> %s",
           relative_posix("L22/T", "L22/T/a.bin").c_str());
    EXPECT(relative_posix("fix/json", "L22/T/a.bin") == "../../L22/T/a.bin", "relative up -> %s",
           relative_posix("fix/json", "L22/T/a.bin").c_str());
    EXPECT(relative_posix("", "L22/T/a.bin") == "L22/T/a.bin", "relative from root -> %s",
           relative_posix("", "L22/T/a.bin").c_str());
}

void test_npz_round_trip()
{
    const std::vector<std::int64_t> ints = {0, -1, 1LL << 40, 42};
    const std::vector<float> floats = {0.5f, -2.25f, 1e-7f, 3.0f, 4.0f, 5.0f};
    const std::vector<std::uint8_t> bytes = {0, 1, 2, 254, 255};
    for (bool compressed : {false, true})
    {
        const std::filesystem::path path =
            std::filesystem::temp_directory_path() / (compressed ? "ai3d_waterpost_test_c.npz" : "ai3d_waterpost_test.npz");
        {
            NpzWriter writer;
            npz_put(writer, "ints", ints);
            npz_put(writer, "floats", floats, {2, 3});
            npz_put(writer, "bytes", bytes);
            npz_put(writer, "empty", std::vector<double>{});
            writer.write(path, compressed);
        }
        const NpzReader reader(path);
        EXPECT(npz_get<std::int64_t>(reader, "ints") == ints, "npz ints (compressed=%d)", compressed);
        EXPECT(npz_get<float>(reader, "floats") == floats, "npz floats (compressed=%d)", compressed);
        EXPECT(reader.get("floats").shape == (std::vector<std::size_t>{2, 3}), "npz shape (compressed=%d)", compressed);
        EXPECT(npz_get<std::uint8_t>(reader, "bytes") == bytes, "npz bytes (compressed=%d)", compressed);
        EXPECT(npz_get<double>(reader, "empty").empty(), "npz empty (compressed=%d)", compressed);
        EXPECT(!reader.contains("missing"), "npz contains");
        std::error_code error;
        std::filesystem::remove(path, error);
    }
}

// 旧的 O(n²) 实现，原样保留作对照。
ShoreLevel shore_level_reference(const std::vector<double>& z, double tolerance = .6)
{
    const double med = median(z);
    std::vector<double> candidates;
    for (double value : z)
    {
        if (value <= med)
        {
            candidates.push_back(value);
        }
    }
    std::vector<std::size_t> scores(candidates.size(), 0);
    std::size_t best_score = 0;
    for (std::size_t c = 0; c < candidates.size(); ++c)
    {
        for (double value : z)
        {
            scores[c] += std::abs(value - candidates[c]) <= tolerance / 2 ? 1 : 0;
        }
        best_score = std::max(best_score, scores[c]);
    }
    double center = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < candidates.size(); ++c)
    {
        if (scores[c] == best_score)
        {
            center = std::min(center, candidates[c]);
        }
    }
    ShoreLevel result;
    result.inliers.assign(z.size(), 0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        result.inliers[i] = std::abs(z[i] - center) <= tolerance / 2;
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

std::vector<double> random_shore(std::mt19937_64& rng, std::size_t n)
{
    // 水位附近的一簇 + 岸坡上的散点 + 少量离群；一部分取整到毫米，制造并列与重复值。
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> noise(0, 0.08);
    const double level = uniform(rng) * 20 - 10;
    std::vector<double> z(n);
    for (double& v : z)
    {
        const double u = uniform(rng);
        v = u < .6 ? level + noise(rng) : u < .95 ? level + uniform(rng) * 3 : level + (uniform(rng) - .5) * 40;
        if (uniform(rng) < .3)
        {
            v = std::round(v * 1000) / 1000;
        }
    }
    return z;
}

void test_shore_level()
{
    std::mt19937_64 rng(20261117);
    std::size_t cases = 0;
    for (int trial = 0; trial < 3000; ++trial)
    {
        const std::size_t n = 3 + static_cast<std::size_t>(rng() % (trial < 2500 ? 60 : 3000));
        const std::vector<double> z = random_shore(rng, n);
        for (double tolerance : {.6, .2, 1.5})
        {
            const ShoreLevel a = shore_level(z, tolerance), b = shore_level_reference(z, tolerance);
            ++cases;
            EXPECT(a.height == b.height && a.inliers == b.inliers, "shore_level differs (n=%zu, tol=%.2f): %.9f vs %.9f", n,
                   tolerance, a.height, b.height);
        }
    }
    std::printf("shore_level: %zu random cases identical to the O(n^2) reference\n", cases);
    for (std::size_t n : {20000, 100000, 1000000})
    {
        const std::vector<double> z = random_shore(rng, n);
        auto t0 = std::chrono::steady_clock::now();
        const ShoreLevel a = shore_level(z);
        const double fast = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double slow = -1;
        if (n <= 20000)
        {
            t0 = std::chrono::steady_clock::now();
            const ShoreLevel b = shore_level_reference(z);
            slow = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            EXPECT(a.height == b.height && a.inliers == b.inliers, "shore_level differs at n=%zu", n);
        }
        std::printf("shore_level n=%zu: new %.3fs%s\n", n, fast,
                    slow >= 0 ? (", old " + std::to_string(slow) + "s").c_str() : "");
    }
}

void test_geo()
{
    // 滑梯：LOD 根变换的平移（ECEF）换到 UTM 50N，应与 bbox.json 的 Normalizer 原点一致（< 1 mm）。
    const Geodetic g = ecef_to_geodetic(-2401151.5303558623, 5379932.625303118, 2435680.754753059);
    const std::array<double, 2> en = utm_forward(g.latitude, g.longitude, utm_zone_of(g.longitude), false);
    EXPECT(utm_zone_of(g.longitude) == 50, "zone %d", utm_zone_of(g.longitude));
    EXPECT(std::abs(en[0] - 196900.08032304232) < 1e-3 && std::abs(en[1] - 2501983.224301045) < 1e-3 &&
               std::abs(g.height - 85.81369735362543) < 1e-3,
           "UTM origin %.4f %.4f %.4f", en[0], en[1], g.height);
    int zone = 0;
    bool south = false;
    parse_pose_crs("EPSG:32749", zone, south);
    EXPECT(zone == 49 && south, "parse EPSG:32749");
    parse_pose_crs("none", zone, south);
    EXPECT(zone == -1, "parse none");
    std::printf("geo: UTM origin %.4f %.4f h %.4f\n", en[0], en[1], g.height);
}

void test_undistort()
{
    // 去畸变后的针孔像素 = 该像素中心对应的视线经原始相机（camera.hpp 的 OPENCV 投影公式）落入的原始像素。
    View pin;
    pin.width = 240;
    pin.height = 160;
    pin.model = "PINHOLE";
    pin.model_id = 1;
    pin.params = {200, 205, 121.3, 79.6};
    pin.r_wc = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    pin.raw_model_id = 4;
    pin.raw_params = {200, 205, 121.3, 79.6, -0.12, 0.03, 0.002, -0.001};
    View raw = pin;
    raw.model = "OPENCV";
    raw.model_id = 4;
    raw.params = pin.raw_params;
    raw.raw_model_id = -1;
    cv::Mat classes(pin.height, pin.width, CV_8U);
    for (int y = 0; y < pin.height; ++y)
    {
        for (int x = 0; x < pin.width; ++x)
        {
            classes.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>((x * 7 + y * 13) % 251);
        }
    }
    const cv::Mat out = undistort_nearest(pin, classes);
    std::size_t checked = 0, wrong = 0;
    for (int y = 0; y < pin.height; ++y)
    {
        for (int x = 0; x < pin.width; ++x)
        {
            // 针孔像素中心 -> 归一化视线（深度 1）-> 原始相机投影。
            const double point[3] = {(x + .5 - pin.params[2]) / pin.params[0], (y + .5 - pin.params[3]) / pin.params[1], 1.0};
            double u = 0, v = 0;
            project_point(raw, point, u, v);
            const double sx = std::floor(u), sy = std::floor(v);
            const bool inside = sx >= 0 && sy >= 0 && sx < pin.width && sy < pin.height;
            const int want = inside ? classes.at<std::uint8_t>(static_cast<int>(sy), static_cast<int>(sx)) : 0;
            ++checked;
            if (out.at<std::uint8_t>(y, x) != want)
            {
                ++wrong;
            }
        }
    }
    EXPECT(wrong == 0, "undistort_nearest: %zu of %zu pixels differ from the OPENCV projection", wrong, checked);
    std::printf("undistort: %zu pixels match the OPENCV projection\n", checked - wrong);
}

} // namespace

void test_concurrent_json()
{
#ifndef _WIN32
    using namespace ai3d::waterpost;
    const auto root = std::filesystem::temp_directory_path() /
                      ("waterpost-json-" + std::to_string(process_id()));
    const auto path = root / "config.json";
    std::vector<pid_t> children;
    for (int worker = 0; worker < 8; ++worker)
    {
        const pid_t child = fork();
        if (child == 0)
        {
            try
            {
                for (int iteration = 0; iteration < 100; ++iteration)
                {
                    Json value = {{"worker", worker}, {"iteration", iteration},
                                  {"payload", std::string(16384, 'a' + worker)}};
                    write_json_atomic(path, value, false);
                    const Json read = read_json_file(path);
                    const int writer = read.at("worker").get<int>();
                    if (writer < 0 || writer >= 8 ||
                        read.at("payload").get<std::string>() != std::string(16384, 'a' + writer))
                        _exit(2);
                }
                _exit(0);
            }
            catch (...) { _exit(3); }
        }
        EXPECT(child > 0, "fork failed");
        if (child > 0) children.push_back(child);
    }
    for (const pid_t child : children)
    {
        int status = 0;
        const auto waited = waitpid(child, &status, 0);
        EXPECT(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "concurrent JSON writer failed: status %d", status);
    }
    std::size_t remaining = 0;
    for (auto it = std::filesystem::directory_iterator(root); it != std::filesystem::directory_iterator(); ++it)
        ++remaining;
    EXPECT(remaining == 1, "temporary JSON files remain: %zu", remaining);
    std::filesystem::remove_all(root);
    std::printf("concurrent JSON: 8 processes, 800 atomic publications\n");
#endif
}

void test_copy_contents()
{
    using namespace ai3d::waterpost;
    const auto root = std::filesystem::temp_directory_path() /
                      ("waterpost-copy-" + std::to_string(process_id()));
    std::filesystem::create_directories(root);
    const auto source = root / "source.bin", target = root / "target.bin";
    std::vector<std::uint8_t> payload((2 << 20) + 137);
    for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<std::uint8_t>((i * 37 + i / 251) & 255);
    write_file_atomic(source, payload.data(), payload.size(), root / "source.tmp");
    copy_file_contents(source, target);
    EXPECT(read_file_bytes(target) == payload, "multi-buffer binary copy changed bytes");
    write_file_atomic(source, payload.data(), 0, root / "source.tmp");
    copy_file_contents(source, target);
    EXPECT(std::filesystem::file_size(target) == 0, "empty copy did not truncate target");
    bool rejected = false;
    try { copy_file_contents(root / "missing", target); }
    catch (const std::exception&) { rejected = true; }
    EXPECT(rejected, "copy accepted a missing source");
    std::filesystem::remove_all(root);
    std::printf("copy contents: binary, partial buffer, empty and missing source passed\n");
}

void test_empty_lod_leaf()
{
    const auto root = std::filesystem::temp_directory_path() /
                      ("waterpost-empty-leaf-" + std::to_string(process_id()));
    Box3 box;
    box.minimum = {133.0, -10.0, -35.0};
    box.maximum = {157.0, 13.0, -18.0};
    for (bool semantic : {false, true})
    {
        const auto path = root / (semantic ? "semantic.bin" : "plain.bin");
        write_leaf(path, {}, 0, box, semantic);
        const auto bytes = read_file_bytes(path);
        const auto padded = read_leaf_padded(path);
        EXPECT(padded == GSSP_PAD, "empty leaf must have one loadable page");
        EXPECT(padded == padded_leaf_count(0), "empty leaf write metadata differs from BIN header");
        EXPECT(bytes.size() == GSSP_HEADER + std::size_t{GSSP_PAD} * GSSP_RECORD,
               "empty leaf payload does not match its page count");
        for (std::size_t i = GSSP_HEADER; i < bytes.size(); ++i)
        {
            const auto expected = semantic && (i - GSSP_HEADER) % GSSP_RECORD == 26 ? 255 : 0;
            EXPECT(bytes[i] == expected, "empty leaf padding contains visible or misclassified records");
        }
        const auto leaf = read_leaf(path);
        EXPECT(leaf.count == 0 && leaf.records.empty(), "padding was read as real Gaussian points");
        EXPECT(leaf.semantic == semantic, "empty leaf semantic padding was not preserved");
    }
    // Preserve the normal nonempty writer/reader contract at a page boundary.
    std::vector<std::uint8_t> records((GSSP_PAD + 1) * GSSP_RECORD);
    for (std::uint32_t i = 0; i < GSSP_PAD + 1; ++i)
        encode_water_record(records.data() + std::size_t{i} * GSSP_RECORD,
                            Vec3{140.0, 0.0, -19.0}, Vec3{0.2, 0.2, 0.0005},
                            {0.2, 0.3, 0.4}, 0.9, true, 1);
    const auto path = root / "water.bin";
    write_leaf(path, records, GSSP_PAD + 1, box, true);
    const auto leaf = read_leaf(path);
    EXPECT(leaf.padded == 2 * GSSP_PAD && leaf.count == GSSP_PAD + 1 && leaf.records == records,
           "nonempty water records changed across page padding");
    std::filesystem::remove_all(root);
    std::printf("LOD empty leaves: loadable transparent page, semantic padding and real-count round trip passed\n");
}

void test_parallel_lod_index()
{
    using namespace ai3d::waterpost;
    const auto root = std::filesystem::temp_directory_path() /
                      ("waterpost-index-" + std::to_string(process_id()));
    Json children = Json::array();
    for (int b = 0; b < 64; ++b)
    {
        Json leaves = Json::array();
        for (int i = 0; i < 2; ++i)
        {
            leaves.push_back({{"content", {{"uri", "../L22/Node_" + std::to_string(b) +
                                              "/leaf_" + std::to_string(i) + ".bin"}}},
                              {"boundingVolume", {{"box", Json::array({b, i, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1})}}}});
        }
        if (b == 0)
            leaves.push_back({{"content", {{"uri", "../L20/Node_0/coarse.bin"}}},
                              {"boundingVolume", {{"box", Json::array({0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2})}}}});
        const std::string uri = "manifests/block_" + std::to_string(b) + ".json";
        write_json_atomic(root / uri, {{"root", {{"children", leaves}}}}, false);
        children.push_back({{"content", {{"uri", uri}}}});
    }
    // Two branches reference the same external manifest. Parallel traversal
    // must still visit it once and retain deterministic finest-leaf ordering.
    children.push_back(children[0]);
    write_json_atomic(root / "tileset.json", {{"root", {{"children", children}}}}, false);
    auto describe = [](const LodIndex& index) {
        Json result = Json::array();
        for (const LodBlock& block : index.blocks)
        {
            Json leaves = Json::array();
            for (const LodLeaf& leaf : block.leaves)
                leaves.push_back({{"uri", leaf.uri}, {"level", leaf.level},
                                  {"minimum", leaf.box.minimum}, {"maximum", leaf.box.maximum}});
            result.push_back({{"name", block.name}, {"level", block.level}, {"leaves", leaves}});
        }
        return result;
    };
    const int previous_threads = thread_count();
    set_thread_count(1);
    const auto serial = read_lod_index(root);
    set_thread_count(8);
    const auto parallel = read_lod_index(root);
    EXPECT(describe(serial) == describe(parallel), "serial and parallel LOD indices differ");
    EXPECT(parallel.blocks.size() == 64, "wrong block count after parallel traversal");
    for (const auto& block : parallel.blocks)
        EXPECT(block.level == 22 && block.leaves.size() == 2, "duplicate or coarse leaves in %s", block.name.c_str());
    enable_lod_index_snapshot_cache();
    const auto snapshot = describe(read_lod_index(root));
    const auto first = root / "manifests/block_0.json", hidden = root / "saved.json";
    std::filesystem::rename(first, hidden);
    EXPECT(describe(read_lod_index(root)) == snapshot, "cached snapshot unexpectedly reread input");
    enable_lod_index_snapshot_cache();
    bool rejected = false;
    try { read_lod_index(root); }
    catch (const std::exception&) { rejected = true; }
    EXPECT(rejected, "cleared snapshot did not reread its missing manifest");
    std::filesystem::rename(hidden, first);
    enable_lod_index_snapshot_cache();
    EXPECT(describe(read_lod_index(root)) == snapshot, "refreshed index differs");
    set_thread_count(previous_threads);
    std::filesystem::remove_all(root);
    std::printf("LOD index: serial/parallel equivalence, shared references, finest levels and snapshot reset passed\n");
}

// 中等连通簇的漂浮物判定：地面上方悬空的密集簇删除；贴近主体、贴着水面、下方没有观测、上方有遮挡、
// 细长线状、碰到块边界的都不删。
void test_floater_classification()
{
    Gaussians g;
    std::vector<std::int32_t> component;
    auto add = [&](double x, double y, double z, std::int32_t c) {
        g.xyz.insert(g.xyz.end(), {x, y, z});
        g.opacity.push_back(0.5f);
        g.semantic.insert(g.semantic.end(), {3, 255});
        component.push_back(c);
        ++g.count;
    };
    // 0：地面 x∈[-10, 20]、y∈[-10, 10]；水面区域 x∈[20, 30] 没有地面点（水位 0）；x∈[-30, -20] 什么都没有。
    for (double x = -10; x <= 20; x += 0.3)
        for (double y = -10; y <= 10; y += 0.3)
            add(x, y, 0.0, 0);
    // 1：屋顶 x∈[10, 18]、y∈[-4, 4]、z = 10。
    for (double x = 10; x <= 18; x += 0.15)
        for (double y = -4; y <= 4; y += 0.15)
            add(x, y, 10.0, 1);
    std::mt19937 random(7);
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);
    auto blob = [&](double x, double y, double z, std::int32_t c) {
        for (int k = 0; k < 50; ++k)
            add(x + jitter(random), y + jitter(random), z + jitter(random), c);
    };
    blob(0, 0, 6, 2);          // 悬空：删除
    blob(-5, 5, 1.0, 3);       // 离地 0.7 m：贴近主体
    blob(25, 0, 0.4, 4);       // 水面上的小船：下方是水面
    blob(-25, 0, 6, 5);        // 下方没有任何观测
    blob(14, 0, 6, 6);         // 屋顶下方
    for (int k = 0; k < 50; ++k) add(-4 + 0.12 * k, -6, 6 + 0.01 * jitter(random), 7); // 6 m 长的线
    blob(5, -5, 6, 8);         // 同样悬空，但标记为碰到块边界
    std::vector<std::size_t> sizes(9, 0);
    for (std::int32_t c : component) ++sizes[static_cast<std::size_t>(c)];
    std::vector<std::uint8_t> border(9, 0);
    border[8] = 1;
    FloaterParams params;
    const auto water = [](double x0, double, double x1, double) {
        return (x1 >= 20 && x0 <= 30) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    };
    const std::vector<FloaterCandidate> result = classify_floaters(g, component, sizes, border, params, water);
    EXPECT(result.size() == 7, "candidates %zu", result.size());
    const char* expected[9] = {"", "", "remove_suspended_floater", "keep_near_structure", "review_detached_supported",
                               "review_unobserved_below", "review_detached_covered", "review_elongated", "keep_block_border"};
    for (const FloaterCandidate& f : result)
    {
        EXPECT(f.decision == expected[f.component], "component %d: %s (gap %.2f clearance %.2f)", f.component, f.decision.c_str(),
               f.gap, f.clearance);
        EXPECT(f.points == 50, "component %d has %zu points", f.component, f.points);
    }
    // 点数阈值不再是免检条件：把 max_points 降到 40，50 点的簇都成了主体，不再判定。
    params.max_points = 40;
    EXPECT(classify_floaters(g, component, sizes, border, params, water).empty(), "max_points");
}

int main()
{
    test_copy_contents();
    test_concurrent_json();
    test_floater_classification();
    test_float_to_half();
    test_posix_paths();
    test_npz_round_trip();
    test_shore_level();
    test_geo();
    test_undistort();
    test_empty_lod_leaf();
    test_parallel_lod_index();
    if (g_failures)
    {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
