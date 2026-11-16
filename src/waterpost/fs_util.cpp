#include "fs_util.hpp"

#include "error.hpp"

#include <fstream>
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ai3d::waterpost
{

std::string path_utf8(const std::filesystem::path& path)
{
#if defined(__cpp_char8_t)
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
#else
    return path.u8string();
#endif
}

std::filesystem::path utf8_path(const std::string& text)
{
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
#else
    return std::filesystem::u8path(text);
#endif
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        fail_missing("Cannot open file: " + path_utf8(path));
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size > 0 ? size : 0));
    if (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        fail_runtime("Cannot read file: " + path_utf8(path));
    }
    return bytes;
}

std::string read_file_text(const std::filesystem::path& path)
{
    const std::vector<std::uint8_t> bytes = read_file_bytes(path);
    return std::string(bytes.begin(), bytes.end());
}

void write_file_atomic(const std::filesystem::path& path, const void* data, std::size_t size,
                       const std::filesystem::path& temporary)
{
    std::error_code error;
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path(), error);
    }
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            throw WaterError(ErrorType::permission, "write failed: cannot open " + path_utf8(temporary));
        }
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        stream.flush();
        if (!stream)
        {
            stream.close();
            std::filesystem::remove(temporary, error);
            fail_runtime("write failed: " + path_utf8(temporary));
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        fail_runtime("write failed: cannot replace " + path_utf8(path) + ": " + error.message());
    }
}

std::string canonical_path_text(const std::filesystem::path& path)
{
    // 同一掩码路径会被成千上万次规范化（每个视图、每个水体），和 Python 的 lru_cache 一样记住结果。
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> cache;
    const std::string key = path_utf8(path);
    const bool cacheable = path.is_absolute();
    if (cacheable)
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(key);
        if (found != cache.end())
        {
            return found->second;
        }
    }
    std::error_code error;
    std::filesystem::path absolute = std::filesystem::absolute(path, error);
    if (error)
    {
        absolute = path;
    }
    std::filesystem::path resolved = std::filesystem::weakly_canonical(absolute, error);
    if (error)
    {
        resolved = absolute.lexically_normal();
    }
    std::string text = path_utf8(resolved.make_preferred());
    if (cacheable)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() > 65536)
        {
            cache.clear();
        }
        cache.emplace(key, text);
    }
    return text;
}

void remove_file_if_exists(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::remove(path, error);
}

long process_id()
{
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

} // namespace ai3d::waterpost
