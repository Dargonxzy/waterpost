#pragma once

// NumPy .npz 读写（zip 容器 + .npy v1/v2 成员），与 np.savez / np.savez_compressed / np.load 互通。
// 水面管线的阶段间交接文件（检测下标、局部分量缓存、删除下标）都是 npz，C++ 版写出的文件 Python 版能直接读，
// 反之亦然，两种实现可以按阶段混用。

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

struct NpyArray
{
    // NumPy 类型串，例如 "<i8"、"<f4"、"|u1"、"<U37"。
    std::string descr;
    std::vector<std::size_t> shape;
    bool fortran_order = false;
    std::vector<std::uint8_t> data;

    std::size_t element_count() const;

    // 一维整数数组（任意整数宽度）转 int64。
    std::vector<std::int64_t> as_int64() const;
    // 浮点 / 整数数组转 double（按 C 序展平）。
    std::vector<double> as_double() const;
    // 0 维 Unicode 标量（np.asarray(str)）转 UTF-8。
    std::string as_unicode() const;
    bool is_integer() const;
};

class NpzReader
{
public:
    explicit NpzReader(const std::filesystem::path& path);

    bool contains(const std::string& name) const;
    // 读出并解压一个成员（name 不含 .npy）；缺失时抛 WaterError(value)。
    NpyArray get(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    struct Entry
    {
        std::uint16_t method = 0;
        std::uint64_t compressed = 0;
        std::uint64_t uncompressed = 0;
        std::uint64_t local_offset = 0;
    };
    std::filesystem::path m_path;
    std::vector<std::uint8_t> m_bytes;
    std::map<std::string, Entry> m_entries;
};

class NpzWriter
{
public:
    void add_int64(const std::string& name, const std::vector<std::int64_t>& values);
    void add_uint16(const std::string& name, const std::vector<std::uint16_t>& values);
    void add_double(const std::string& name, const std::vector<double>& values, std::vector<std::size_t> shape);
    void add_float(const std::string& name, const std::vector<float>& values, std::vector<std::size_t> shape);
    void add_int32(const std::string& name, const std::vector<std::int32_t>& values);
    void add_uint32(const std::string& name, const std::vector<std::uint32_t>& values);
    void add_uint8(const std::string& name, const std::vector<std::uint8_t>& values);
    void add_float32_1d(const std::string& name, const std::vector<float>& values);
    // np.asarray(<str>)：0 维 '<U{n}'（UTF-32LE）。
    void add_unicode(const std::string& name, const std::string& utf8);
    void add_raw(const std::string& name, NpyArray array);
    // 把另一个写出器的全部成员按原次序接在后面。
    void append(NpzWriter&& other);

    // 在内存中组装整个 zip，写 <path>.<pid>.tmp 后原子替换（savez_sequential）。
    void write(const std::filesystem::path& path, bool compressed) const;

private:
    std::vector<std::pair<std::string, NpyArray>> m_members;
};

} // namespace ai3d::waterpost
