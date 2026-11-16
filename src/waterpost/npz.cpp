#include "npz.hpp"

#include "error.hpp"
#include "fs_util.hpp"

#include <zlib.h>

#include <cstring>
#include <sstream>

namespace ai3d::waterpost
{
namespace
{

std::uint16_t read_u16(const std::uint8_t* data)
{
    return static_cast<std::uint16_t>(data[0] | (data[1] << 8));
}

std::uint32_t read_u32(const std::uint8_t* data)
{
    return static_cast<std::uint32_t>(data[0]) | (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) | (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint64_t read_u64(const std::uint8_t* data)
{
    return static_cast<std::uint64_t>(read_u32(data)) | (static_cast<std::uint64_t>(read_u32(data + 4)) << 32);
}

void put_u16(std::vector<std::uint8_t>& out, std::uint32_t value)
{
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value)
{
    for (int shift = 0; shift < 32; shift += 8)
    {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

std::size_t element_size(const std::string& descr)
{
    if (descr.size() < 3)
    {
        fail_value("Unsupported npy dtype: " + descr);
    }
    const char kind = descr[1];
    const std::size_t width = static_cast<std::size_t>(std::stoul(descr.substr(2)));
    if (kind == 'U')
    {
        return width * 4;
    }
    return width;
}

[[noreturn]] void bad_npy(const std::string& why)
{
    throw WaterError(ErrorType::value, "Invalid npy member: " + why);
}

NpyArray parse_npy(const std::vector<std::uint8_t>& bytes)
{
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
    {
        bad_npy("bad magic");
    }
    const int major = bytes[6];
    std::size_t header_length = 0;
    std::size_t offset = 0;
    if (major == 1)
    {
        header_length = read_u16(bytes.data() + 8);
        offset = 10;
    }
    else
    {
        if (bytes.size() < 12)
        {
            bad_npy("truncated");
        }
        header_length = read_u32(bytes.data() + 8);
        offset = 12;
    }
    if (offset + header_length > bytes.size())
    {
        bad_npy("truncated header");
    }
    const std::string header(reinterpret_cast<const char*>(bytes.data() + offset), header_length);
    NpyArray array;
    const auto value_after = [&](const std::string& key) -> std::size_t {
        const std::size_t position = header.find("'" + key + "'");
        if (position == std::string::npos)
        {
            bad_npy("missing " + key);
        }
        const std::size_t colon = header.find(':', position);
        std::size_t start = colon + 1;
        while (start < header.size() && header[start] == ' ')
        {
            ++start;
        }
        return start;
    };
    {
        const std::size_t start = value_after("descr");
        const char quote = header[start];
        const std::size_t end = header.find(quote, start + 1);
        array.descr = header.substr(start + 1, end - start - 1);
        if (array.descr.size() >= 2 && array.descr[0] == '=')
        {
            array.descr[0] = '<';
        }
    }
    {
        const std::size_t start = value_after("fortran_order");
        array.fortran_order = header.compare(start, 4, "True") == 0;
    }
    {
        const std::size_t start = value_after("shape");
        const std::size_t end = header.find(')', start);
        const std::string inside = header.substr(start + 1, end - start - 1);
        std::stringstream stream(inside);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            std::size_t first = token.find_first_not_of(' ');
            if (first == std::string::npos)
            {
                continue;
            }
            std::string number = token.substr(first);
            while (!number.empty() && (number.back() == ' ' || number.back() == 'L'))
            {
                number.pop_back();
            }
            if (!number.empty())
            {
                array.shape.push_back(static_cast<std::size_t>(std::stoull(number)));
            }
        }
    }
    if (array.descr[0] == '>')
    {
        bad_npy("big-endian arrays are not supported");
    }
    const std::size_t data_offset = offset + header_length;
    const std::size_t expected = array.element_count() * element_size(array.descr);
    if (bytes.size() - data_offset < expected)
    {
        bad_npy("truncated data");
    }
    array.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                      bytes.begin() + static_cast<std::ptrdiff_t>(data_offset + expected));
    return array;
}

std::vector<std::uint8_t> serialize_npy(const NpyArray& array)
{
    std::string shape = "(";
    for (std::size_t index = 0; index < array.shape.size(); ++index)
    {
        shape += std::to_string(array.shape[index]);
        if (array.shape.size() == 1 || index + 1 < array.shape.size())
        {
            shape += array.shape.size() == 1 ? "," : ", ";
        }
    }
    shape += ")";
    std::string header = "{'descr': '" + array.descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    // 与 numpy 一样把魔数 + 头长 + 头文本补齐到 64 字节的倍数，末尾是换行。
    std::size_t total = 10 + header.size() + 1;
    const std::size_t padding = (64 - (total % 64)) % 64;
    header.append(padding, ' ');
    header.push_back('\n');
    std::vector<std::uint8_t> out;
    out.reserve(10 + header.size() + array.data.size());
    const char magic[] = "\x93NUMPY";
    out.insert(out.end(), magic, magic + 6);
    out.push_back(1);
    out.push_back(0);
    put_u16(out, static_cast<std::uint32_t>(header.size()));
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), array.data.begin(), array.data.end());
    return out;
}

std::vector<std::uint8_t> deflate_raw(const std::vector<std::uint8_t>& input)
{
    z_stream stream{};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    {
        fail_runtime("deflateInit2 failed");
    }
    std::vector<std::uint8_t> output(deflateBound(&stream, static_cast<uLong>(input.size())) + 64);
    stream.next_in = const_cast<Bytef*>(input.data());
    stream.avail_in = static_cast<uInt>(input.size());
    stream.next_out = output.data();
    stream.avail_out = static_cast<uInt>(output.size());
    const int status = deflate(&stream, Z_FINISH);
    if (status != Z_STREAM_END)
    {
        deflateEnd(&stream);
        fail_runtime("deflate failed");
    }
    output.resize(stream.total_out);
    deflateEnd(&stream);
    return output;
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* input, std::size_t size, std::size_t expected)
{
    z_stream stream{};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
    {
        fail_runtime("inflateInit2 failed");
    }
    std::vector<std::uint8_t> output(expected);
    stream.next_in = const_cast<Bytef*>(input);
    stream.avail_in = static_cast<uInt>(size);
    stream.next_out = output.data();
    stream.avail_out = static_cast<uInt>(output.size());
    const int status = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (status != Z_STREAM_END || stream.total_out != expected)
    {
        throw WaterError(ErrorType::value, "Corrupt deflate stream in npz archive");
    }
    return output;
}

std::vector<std::uint32_t> utf8_to_code_points(const std::string& text)
{
    std::vector<std::uint32_t> points;
    for (std::size_t index = 0; index < text.size();)
    {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        std::uint32_t value = 0;
        int extra = 0;
        if (lead < 0x80)
        {
            value = lead;
        }
        else if ((lead >> 5) == 0x6)
        {
            value = lead & 0x1f;
            extra = 1;
        }
        else if ((lead >> 4) == 0xe)
        {
            value = lead & 0x0f;
            extra = 2;
        }
        else
        {
            value = lead & 0x07;
            extra = 3;
        }
        ++index;
        for (int k = 0; k < extra && index < text.size(); ++k, ++index)
        {
            value = (value << 6) | (static_cast<unsigned char>(text[index]) & 0x3f);
        }
        points.push_back(value);
    }
    return points;
}

void append_utf8(std::string& out, std::uint32_t value)
{
    if (value < 0x80)
    {
        out.push_back(static_cast<char>(value));
    }
    else if (value < 0x800)
    {
        out.push_back(static_cast<char>(0xc0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    else if (value < 0x10000)
    {
        out.push_back(static_cast<char>(0xe0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    else
    {
        out.push_back(static_cast<char>(0xf0 | (value >> 18)));
        out.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
}

template <typename T>
NpyArray make_array(const std::string& descr, const std::vector<T>& values, std::vector<std::size_t> shape)
{
    NpyArray array;
    array.descr = descr;
    array.shape = std::move(shape);
    array.data.resize(values.size() * sizeof(T));
    if (!values.empty())
    {
        std::memcpy(array.data.data(), values.data(), array.data.size());
    }
    return array;
}

} // namespace

std::size_t NpyArray::element_count() const
{
    std::size_t count = 1;
    for (std::size_t dimension : shape)
    {
        count *= dimension;
    }
    return count;
}

bool NpyArray::is_integer() const
{
    return descr.size() >= 3 && (descr[1] == 'i' || descr[1] == 'u');
}

std::vector<std::int64_t> NpyArray::as_int64() const
{
    if (!is_integer())
    {
        throw WaterError(ErrorType::value, "npy array is not an integer array: " + descr);
    }
    const std::size_t count = element_count();
    const std::size_t width = element_size(descr);
    const bool is_signed = descr[1] == 'i';
    std::vector<std::int64_t> values(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        const std::uint8_t* item = data.data() + index * width;
        std::uint64_t raw = 0;
        for (std::size_t byte = 0; byte < width; ++byte)
        {
            raw |= static_cast<std::uint64_t>(item[byte]) << (8 * byte);
        }
        if (is_signed && width < 8 && (raw >> (8 * width - 1)) & 1u)
        {
            raw |= ~std::uint64_t{0} << (8 * width);
        }
        if (!is_signed && width == 8 && raw > static_cast<std::uint64_t>(INT64_MAX))
        {
            throw WaterError(ErrorType::value, "Detection point index exceeds int64 range");
        }
        values[index] = static_cast<std::int64_t>(raw);
    }
    return values;
}

std::vector<double> NpyArray::as_double() const
{
    const std::size_t count = element_count();
    std::vector<double> values(count);
    if (descr == "<f8")
    {
        std::memcpy(values.data(), data.data(), count * 8);
    }
    else if (descr == "<f4")
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            float value;
            std::memcpy(&value, data.data() + index * 4, 4);
            values[index] = static_cast<double>(value);
        }
    }
    else if (is_integer())
    {
        const std::vector<std::int64_t> integers = as_int64();
        for (std::size_t index = 0; index < count; ++index)
        {
            values[index] = static_cast<double>(integers[index]);
        }
    }
    else
    {
        throw WaterError(ErrorType::value, "Unsupported npy float dtype: " + descr);
    }
    return values;
}

std::string NpyArray::as_unicode() const
{
    if (descr.size() < 3 || descr[1] != 'U')
    {
        throw WaterError(ErrorType::value, "npy member is not a unicode scalar: " + descr);
    }
    const std::size_t width = element_size(descr) / 4;
    std::string text;
    for (std::size_t index = 0; index < width; ++index)
    {
        const std::uint32_t value = read_u32(data.data() + index * 4);
        if (value == 0)
        {
            break;
        }
        append_utf8(text, value);
    }
    return text;
}

NpzReader::NpzReader(const std::filesystem::path& path) : m_path(path)
{
    m_bytes = read_file_bytes(path);
    const std::vector<std::uint8_t>& bytes = m_bytes;
    if (bytes.size() < 22)
    {
        throw WaterError(ErrorType::value, "BadZipFile: file is too small: " + path_utf8(path));
    }
    std::size_t eocd = std::string::npos;
    const std::size_t lowest = bytes.size() > 22 + 65535 ? bytes.size() - 22 - 65535 : 0;
    for (std::size_t position = bytes.size() - 22 + 1; position-- > lowest;)
    {
        if (read_u32(bytes.data() + position) == 0x06054b50)
        {
            eocd = position;
            break;
        }
    }
    if (eocd == std::string::npos)
    {
        throw WaterError(ErrorType::value, "BadZipFile: end of central directory not found: " + path_utf8(path));
    }
    std::uint64_t entry_count = read_u16(bytes.data() + eocd + 10);
    std::uint64_t directory_offset = read_u32(bytes.data() + eocd + 16);
    if ((entry_count == 0xffff || directory_offset == 0xffffffffu) && eocd >= 20 &&
        read_u32(bytes.data() + eocd - 20) == 0x07064b50)
    {
        const std::uint64_t zip64_eocd = read_u64(bytes.data() + eocd - 20 + 8);
        if (zip64_eocd + 56 <= bytes.size() && read_u32(bytes.data() + zip64_eocd) == 0x06064b50)
        {
            entry_count = read_u64(bytes.data() + zip64_eocd + 32);
            directory_offset = read_u64(bytes.data() + zip64_eocd + 48);
        }
    }
    std::size_t position = static_cast<std::size_t>(directory_offset);
    for (std::uint64_t entry = 0; entry < entry_count; ++entry)
    {
        if (position + 46 > bytes.size() || read_u32(bytes.data() + position) != 0x02014b50)
        {
            throw WaterError(ErrorType::value, "BadZipFile: bad central directory: " + path_utf8(path));
        }
        const std::uint8_t* header = bytes.data() + position;
        Entry item;
        item.method = read_u16(header + 10);
        item.compressed = read_u32(header + 20);
        item.uncompressed = read_u32(header + 24);
        const std::size_t name_length = read_u16(header + 28);
        const std::size_t extra_length = read_u16(header + 30);
        const std::size_t comment_length = read_u16(header + 32);
        item.local_offset = read_u32(header + 42);
        const std::string name(reinterpret_cast<const char*>(header + 46), name_length);
        const std::uint8_t* extra = header + 46 + name_length;
        for (std::size_t offset = 0; offset + 4 <= extra_length;)
        {
            const std::uint16_t id = read_u16(extra + offset);
            const std::uint16_t size = read_u16(extra + offset + 2);
            if (id == 0x0001)
            {
                std::size_t field = offset + 4;
                if (item.uncompressed == 0xffffffffu)
                {
                    item.uncompressed = read_u64(extra + field);
                    field += 8;
                }
                if (item.compressed == 0xffffffffu)
                {
                    item.compressed = read_u64(extra + field);
                    field += 8;
                }
                if (item.local_offset == 0xffffffffu)
                {
                    item.local_offset = read_u64(extra + field);
                }
            }
            offset += 4 + size;
        }
        std::string key = name;
        if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0)
        {
            key.resize(key.size() - 4);
        }
        m_entries[key] = item;
        position += 46 + name_length + extra_length + comment_length;
    }
}

bool NpzReader::contains(const std::string& name) const
{
    return m_entries.count(name) != 0;
}

std::vector<std::string> NpzReader::names() const
{
    std::vector<std::string> result;
    for (const auto& item : m_entries)
    {
        result.push_back(item.first);
    }
    return result;
}

NpyArray NpzReader::get(const std::string& name) const
{
    const auto found = m_entries.find(name);
    if (found == m_entries.end())
    {
        throw WaterError(ErrorType::value, "KeyError: " + name + " is not a file in the archive " + path_utf8(m_path));
    }
    const Entry& entry = found->second;
    if (entry.local_offset + 30 > m_bytes.size() || read_u32(m_bytes.data() + entry.local_offset) != 0x04034b50)
    {
        throw WaterError(ErrorType::value, "BadZipFile: bad local header for " + name);
    }
    const std::uint8_t* local = m_bytes.data() + entry.local_offset;
    const std::size_t data_offset =
        static_cast<std::size_t>(entry.local_offset) + 30 + read_u16(local + 26) + read_u16(local + 28);
    if (data_offset + entry.compressed > m_bytes.size())
    {
        throw WaterError(ErrorType::value, "BadZipFile: truncated member " + name);
    }
    std::vector<std::uint8_t> content;
    if (entry.method == 0)
    {
        content.assign(m_bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                       m_bytes.begin() + static_cast<std::ptrdiff_t>(data_offset + entry.compressed));
    }
    else if (entry.method == 8)
    {
        content = inflate_raw(m_bytes.data() + data_offset, static_cast<std::size_t>(entry.compressed),
                              static_cast<std::size_t>(entry.uncompressed));
    }
    else
    {
        throw WaterError(ErrorType::value, "Unsupported zip compression method in " + path_utf8(m_path));
    }
    return parse_npy(content);
}

void NpzWriter::add_int64(const std::string& name, const std::vector<std::int64_t>& values)
{
    m_members.emplace_back(name, make_array("<i8", values, {values.size()}));
}

void NpzWriter::add_uint16(const std::string& name, const std::vector<std::uint16_t>& values)
{
    m_members.emplace_back(name, make_array("<u2", values, {values.size()}));
}

void NpzWriter::add_int32(const std::string& name, const std::vector<std::int32_t>& values)
{
    m_members.emplace_back(name, make_array("<i4", values, {values.size()}));
}

void NpzWriter::add_uint32(const std::string& name, const std::vector<std::uint32_t>& values)
{
    m_members.emplace_back(name, make_array("<u4", values, {values.size()}));
}

void NpzWriter::add_uint8(const std::string& name, const std::vector<std::uint8_t>& values)
{
    m_members.emplace_back(name, make_array("|u1", values, {values.size()}));
}

void NpzWriter::add_float32_1d(const std::string& name, const std::vector<float>& values)
{
    m_members.emplace_back(name, make_array("<f4", values, {values.size()}));
}

void NpzWriter::add_double(const std::string& name, const std::vector<double>& values, std::vector<std::size_t> shape)
{
    m_members.emplace_back(name, make_array("<f8", values, std::move(shape)));
}

void NpzWriter::add_float(const std::string& name, const std::vector<float>& values, std::vector<std::size_t> shape)
{
    m_members.emplace_back(name, make_array("<f4", values, std::move(shape)));
}

void NpzWriter::add_unicode(const std::string& name, const std::string& utf8)
{
    const std::vector<std::uint32_t> points = utf8_to_code_points(utf8);
    NpyArray array;
    array.descr = "<U" + std::to_string(points.size());
    array.data.resize(points.size() * 4);
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        for (int byte = 0; byte < 4; ++byte)
        {
            array.data[index * 4 + static_cast<std::size_t>(byte)] =
                static_cast<std::uint8_t>((points[index] >> (8 * byte)) & 0xff);
        }
    }
    m_members.emplace_back(name, std::move(array));
}

void NpzWriter::add_raw(const std::string& name, NpyArray array)
{
    m_members.emplace_back(name, std::move(array));
}

void NpzWriter::append(NpzWriter&& other)
{
    for (auto& member : other.m_members)
    {
        m_members.push_back(std::move(member));
    }
    other.m_members.clear();
}

void NpzWriter::write(const std::filesystem::path& path, bool compressed) const
{
    std::vector<std::uint8_t> archive;
    std::vector<std::uint8_t> directory;
    const std::uint16_t dos_time = 0;
    const std::uint16_t dos_date = (0 << 9) | (1 << 5) | 1;
    for (const auto& member : m_members)
    {
        const std::string name = member.first + ".npy";
        const std::vector<std::uint8_t> raw = serialize_npy(member.second);
        const std::uint32_t crc = static_cast<std::uint32_t>(crc32(0L, raw.data(), static_cast<uInt>(raw.size())));
        const std::vector<std::uint8_t> stored = compressed ? deflate_raw(raw) : raw;
        if (raw.size() >= 0xffffffffu || stored.size() >= 0xffffffffu || archive.size() >= 0xffffffffu)
        {
            fail_runtime("npz member exceeds the 4 GiB zip limit: " + name);
        }
        const std::uint32_t offset = static_cast<std::uint32_t>(archive.size());
        const std::uint16_t method = compressed ? 8 : 0;
        put_u32(archive, 0x04034b50);
        put_u16(archive, 20);
        put_u16(archive, 0);
        put_u16(archive, method);
        put_u16(archive, dos_time);
        put_u16(archive, dos_date);
        put_u32(archive, crc);
        put_u32(archive, static_cast<std::uint32_t>(stored.size()));
        put_u32(archive, static_cast<std::uint32_t>(raw.size()));
        put_u16(archive, static_cast<std::uint32_t>(name.size()));
        put_u16(archive, 0);
        archive.insert(archive.end(), name.begin(), name.end());
        archive.insert(archive.end(), stored.begin(), stored.end());

        put_u32(directory, 0x02014b50);
        put_u16(directory, 20);
        put_u16(directory, 20);
        put_u16(directory, 0);
        put_u16(directory, method);
        put_u16(directory, dos_time);
        put_u16(directory, dos_date);
        put_u32(directory, crc);
        put_u32(directory, static_cast<std::uint32_t>(stored.size()));
        put_u32(directory, static_cast<std::uint32_t>(raw.size()));
        put_u16(directory, static_cast<std::uint32_t>(name.size()));
        put_u16(directory, 0);
        put_u16(directory, 0);
        put_u16(directory, 0);
        put_u16(directory, 0);
        put_u32(directory, 0x81800000u);
        put_u32(directory, offset);
        directory.insert(directory.end(), name.begin(), name.end());
    }
    const std::uint32_t directory_offset = static_cast<std::uint32_t>(archive.size());
    archive.insert(archive.end(), directory.begin(), directory.end());
    put_u32(archive, 0x06054b50);
    put_u16(archive, 0);
    put_u16(archive, 0);
    put_u16(archive, static_cast<std::uint32_t>(m_members.size()));
    put_u16(archive, static_cast<std::uint32_t>(m_members.size()));
    put_u32(archive, static_cast<std::uint32_t>(directory.size()));
    put_u32(archive, directory_offset);
    put_u16(archive, 0);
    std::filesystem::path temporary = path;
    temporary += "." + std::to_string(process_id()) + ".tmp";
    write_file_atomic(path, archive.data(), archive.size(), temporary);
}

} // namespace ai3d::waterpost
