#include "parallel.hpp"

#include <memory>
#include <mutex>
#include <thread>

namespace ai3d::waterpost
{
namespace
{

int g_threads = 0;
std::unique_ptr<tbb::task_arena> g_arena;
std::mutex g_mutex;

int resolved_threads(int requested)
{
    if (requested > 0)
    {
        return requested;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware == 0 ? 1 : static_cast<int>(hardware);
}

} // namespace

void set_thread_count(int threads)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_threads = resolved_threads(threads);
    g_arena = std::make_unique<tbb::task_arena>(g_threads);
}

int thread_count()
{
    if (g_threads == 0)
    {
        set_thread_count(0);
    }
    return g_threads;
}

tbb::task_arena& arena()
{
    if (!g_arena)
    {
        set_thread_count(0);
    }
    return *g_arena;
}

} // namespace ai3d::waterpost
