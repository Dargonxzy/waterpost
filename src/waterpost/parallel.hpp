#pragma once

// CPU 并行：全进程共用一个限定并发数的 oneTBB arena（--threads）。循环体抛出的异常由 TBB 带回调用线程。

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>


#include <algorithm>
#include <cstddef>
#include <functional>

namespace ai3d::waterpost
{

// 设置并发线程数；0 = 全部硬件线程。
void set_thread_count(int threads);
int thread_count();
tbb::task_arena& arena();

// 对 [0, count) 每个下标调一次 function（粒度 1，按项窃取）。
template <typename Function>
void parallel_for_each(std::size_t count, Function&& function)
{
    if (count == 0)
    {
        return;
    }
    if (count == 1 || thread_count() == 1)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            function(index);
        }
        return;
    }
    arena().execute([&] {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, 1), [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t index = range.begin(); index != range.end(); ++index)
            {
                function(index);
            }
        });
    });
}

// 把 [0, count) 切成至多 grain 大小的区间并行处理：function(begin, end)。
template <typename Function>
void parallel_ranges(std::size_t count, std::size_t grain, Function&& function)
{
    if (count == 0)
    {
        return;
    }
    grain = std::max<std::size_t>(grain, 1);
    if (count <= grain || thread_count() == 1)
    {
        function(std::size_t{0}, count);
        return;
    }
    arena().execute([&] {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, grain),
                          [&](const tbb::blocked_range<std::size_t>& range) { function(range.begin(), range.end()); });
    });
}

// 至多 limit 路并发地对 [0, count) 每个下标调一次 function（独立 arena；用于按内存预算限制同时处理的掩码张数）。
template <typename Function>
void parallel_for_limited(std::size_t count, std::size_t limit, Function&& function)
{
    if (count == 0)
    {
        return;
    }
    limit = std::max<std::size_t>(1, std::min<std::size_t>(limit, static_cast<std::size_t>(thread_count())));
    if (limit == 1 || count == 1)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            function(index);
        }
        return;
    }
    tbb::task_arena limited(static_cast<int>(limit));
    limited.execute([&] {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, 1), [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t index = range.begin(); index != range.end(); ++index)
            {
                function(index);
            }
        });
    });
}


} // namespace ai3d::waterpost
