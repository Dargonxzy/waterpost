// Small deployment test: execute the actual vote, render and occlusion kernels.
#include "gpu.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace ai3d::waterpost;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv)
{
    try
    {
        gpu_select_device(argc > 1 ? std::stoi(argv[1]) : 0);
        PinholeCamera camera;
        camera.width = camera.height = 32;
        camera.fx = camera.fy = 20;
        camera.cx = camera.cy = 16;
        camera.r[0] = camera.r[4] = camera.r[8] = 1;

        Gaussians g;
        g.count = 3;
        g.xyz = {0, 0, 5, 1, 0, 6, -1, 0, 7};
        g.cov = {.04f, 0, 0, .04f, 0, .04f,
                 .04f, 0, 0, .04f, 0, .04f,
                 .04f, 0, 0, .04f, 0, .04f};
        g.opacity.assign(g.count, .9f);
        g.sigma_max.assign(g.count, .2f);
        g.rgba = {255, 0, 0, 230, 0, 255, 0, 230, 0, 0, 255, 230};
        const std::vector<std::uint8_t> wet(32 * 32, 1), shore(32 * 32, 0);
        const std::vector<float> distance(32 * 32, 32.f);
        DetectGpu detect(g, 1.0);
        require(detect.vote(camera, false, wet.data(), distance.data(), shore.data()) == g.count,
                "Detect projection missed an in-frame Gaussian");
        detect.render(camera, wet.data());
        const auto votes = detect.download();
        for (std::size_t i = 0; i < g.count; ++i)
        {
            require(votes.visible[i] == 1 && votes.water[i] == 1 && votes.land[i] == 0,
                    "Water vote mismatch");
            require(votes.mass[3 * i] > 0 && votes.mass[3 * i + 1] == 0,
                    "Visible water contribution mismatch");
        }
        ColorRenderer renderer(g, 1.0);
        const auto image = renderer.render(camera);
        require(image.size() == 32 * 32 * 3 && *std::max_element(image.begin(), image.end()) > .1f,
                "Colour renderer returned an empty image");
        for (float value : image) require(std::isfinite(value) && value >= 0 && value <= 1, "Invalid render value");

        std::vector<std::uint32_t> support, against;
        std::size_t occluded = 0;
        GridGpu grid({0, 0, 5, 1, 0, 5}, {}, 4);
        grid.signature(camera, wet.data(), .1);
        grid.other(camera, true, nullptr, .1);
        grid.download(support, against, occluded);
        require(support == std::vector<std::uint32_t>({1, 1}) && against == support && occluded == 0,
                "Plane-cell evidence mismatch");
        GridGpu hidden({0, 0, 5}, {0, 0, 2}, 4);
        hidden.signature(camera, wet.data(), .1);
        hidden.download(support, against, occluded);
        require(support[0] == 0 && against[0] == 0 && occluded == 1, "Depth occlusion mismatch");
        std::puts("CUDA smoke passed: detection, visible contributions, colour rendering, grid evidence, occlusion");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "CUDA smoke FAILED: %s\n", error.what());
        return 1;
    }
}
