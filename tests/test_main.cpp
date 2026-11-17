// 纯函数单元测试：半精度转换、POSIX 路径工具、npz 读写往返。失败时返回非 0。

#include "lod.hpp"
#include "npz.hpp"
#include "npz_util.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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

} // namespace

int main()
{
    test_float_to_half();
    test_posix_paths();
    test_npz_round_trip();
    if (g_failures)
    {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
