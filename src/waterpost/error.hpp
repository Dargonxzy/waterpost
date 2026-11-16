#pragma once

// 内部错误：算法实现按 Python 版的异常语义抛出 WaterError，管线入口（pipeline.cpp）统一捕获并换成
// core::Result / 退出码。kind 只区分 Python 里有专门分类规则的几种异常（FileNotFoundError、ValueError 等）。


#include <stdexcept>
#include <string>

namespace ai3d::waterpost
{

enum class ErrorType
{
    runtime,
    file_not_found,
    value,
    index,
    stale_index,
    memory,
    permission,
    parse,
};

class WaterError : public std::runtime_error
{
public:
    WaterError(ErrorType type, const std::string& message) : std::runtime_error(message), m_type(type)
    {
    }

    ErrorType type() const
    {
        return m_type;
    }

private:
    ErrorType m_type;
};

[[noreturn]] inline void fail_runtime(const std::string& message)
{
    throw WaterError(ErrorType::runtime, message);
}

[[noreturn]] inline void fail_value(const std::string& message)
{
    throw WaterError(ErrorType::value, message);
}

[[noreturn]] inline void fail_missing(const std::string& message)
{
    throw WaterError(ErrorType::file_not_found, message);
}

} // namespace ai3d::waterpost
