#pragma once

// CUDA kernels behind plain C++ interfaces (no CUDA types in this header).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ai3d::waterpost
{

struct Gaussians;
struct Occupancy;

// Undistorted pinhole camera, world-to-camera rotation (row major) and camera centre.
struct PinholeCamera
{
    int width = 0;
    int height = 0;
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
    double r[9] = {};
    double center[3] = {};
};

void gpu_select_device(int device);
std::size_t gpu_free_bytes();

// Detect: per-view binary votes (mask_voting._vote) and visible T*alpha support (VisibleSupport).
class DetectGpu
{
public:
    DetectGpu(const Gaussians& gaussians, double unit);
    ~DetectGpu();
    DetectGpu(const DetectGpu&) = delete;
    DetectGpu& operator=(const DetectGpu&) = delete;

    // state: H*W labels 0 land / 1 water / 2 unknown; distance: signed distance field (float);
    // shoreline: H*W 0/1. All three are null for a negative-only view. Returns in-frame point count.
    std::size_t vote(const PinholeCamera& camera, bool negative_only, const std::uint8_t* state, const float* distance,
                     const std::uint8_t* shoreline);
    // Accumulate T*alpha masses of this view for the points marked in-frame by the last vote().
    // state null means an all-land (dry) image.
    void render(const PinholeCamera& camera, const std::uint8_t* state);

    struct Result
    {
        std::vector<std::int32_t> visible, water, land, center_visible, center_water, boundary, water_views;
        std::vector<double> mass; // 3n: water, land, unknown
        std::size_t boundary_checks = 0;
        std::size_t early_uniform = 0;
        std::size_t subdivided_tiles = 0;
    };
    Result download();

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

// Plain colour rendering (degree-0 colours from the BIN, black background) for quality checks.
class ColorRenderer
{
public:
    ColorRenderer(const Gaussians& gaussians, double unit);
    ~ColorRenderer();
    ColorRenderer(const ColorRenderer&) = delete;
    ColorRenderer& operator=(const ColorRenderer&) = delete;
    // Returns H*W*3 RGB in [0, 1].
    std::vector<float> render(const PinholeCamera& camera);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

// Apply: plane-cell evidence against component labels / occupancy, with an occluder z-buffer.
class GridGpu
{
public:
    // cells: 3m world points; occluders: 3k world points (may be empty).
    GridGpu(const std::vector<double>& cells, const std::vector<double>& occluders, int zbuffer_factor);
    ~GridGpu();
    GridGpu(const GridGpu&) = delete;
    GridGpu& operator=(const GridGpu&) = delete;

    std::size_t count() const;
    // Possible support: in-frame cells whose 32 px occupancy cell holds water (all in-frame without occupancy).
    void possible(const PinholeCamera& camera, const Occupancy* occupancy);
    // Exact label support: image is H*W with 1 = this body's component.
    void exact(const PinholeCamera& camera, const std::uint8_t* image);
    // Signature view with occlusion: image 1 = this body, 0 = dry (label 0), 2 = other water.
    void signature(const PinholeCamera& camera, const std::uint8_t* image, double tolerance);
    // Non-signature view with occlusion: dry view, or dry 32 px occupancy cells, are counter-evidence.
    void other(const PinholeCamera& camera, bool dry, const Occupancy* occupancy, double tolerance);

    void download(std::vector<std::uint32_t>& support, std::vector<std::uint32_t>& against,
                  std::size_t& occluded) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

} // namespace ai3d::waterpost
