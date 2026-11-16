#pragma once

// npz 读写的小工具：任意形状的定长数组、按名字读成 C++ 向量。

#include "error.hpp"
#include "npz.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

template <typename T>
inline const char* npy_descr();
template <>
inline const char* npy_descr<std::int64_t>()
{
    return "<i8";
}
template <>
inline const char* npy_descr<std::int32_t>()
{
    return "<i4";
}
template <>
inline const char* npy_descr<std::uint8_t>()
{
    return "|u1";
}
template <>
inline const char* npy_descr<std::uint16_t>()
{
    return "<u2";
}
template <>
inline const char* npy_descr<std::uint32_t>()
{
    return "<u4";
}
template <>
inline const char* npy_descr<float>()
{
    return "<f4";
}
template <>
inline const char* npy_descr<double>()
{
    return "<f8";
}

template <typename T>
inline void npz_put(NpzWriter& writer, const std::string& name, const std::vector<T>& values,
                    std::vector<std::size_t> shape = {})
{
    NpyArray array;
    array.descr = npy_descr<T>();
    array.shape = shape.empty() ? std::vector<std::size_t>{values.size()} : std::move(shape);
    array.data.resize(values.size() * sizeof(T));
    if (!values.empty())
    {
        std::memcpy(array.data.data(), values.data(), array.data.size());
    }
    writer.add_raw(name, std::move(array));
}

template <typename T>
inline std::vector<T> npz_get(const NpzReader& reader, const std::string& name)
{
    const NpyArray array = reader.get(name);
    std::vector<T> out(array.element_count());
    if (array.descr == npy_descr<T>())
    {
        if (!out.empty())
        {
            std::memcpy(out.data(), array.data.data(), out.size() * sizeof(T));
        }
        return out;
    }
    if (array.is_integer())
    {
        const std::vector<std::int64_t> values = array.as_int64();
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = static_cast<T>(values[i]);
        }
        return out;
    }
    const std::vector<double> values = array.as_double();
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out[i] = static_cast<T>(values[i]);
    }
    return out;
}

} // namespace ai3d::waterpost
