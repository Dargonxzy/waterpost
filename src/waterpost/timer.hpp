#pragma once

#include <chrono>

namespace ai3d::waterpost
{

class Timer
{
public:
    Timer() : m_start(std::chrono::steady_clock::now())
    {
    }

    double seconds() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
    }

private:
    std::chrono::steady_clock::time_point m_start;
};

} // namespace ai3d::waterpost
