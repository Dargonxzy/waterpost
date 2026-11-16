// CUDA kernels for water post-processing.
// Detect voting follows water_postprocessing/mask_voting.py; the T*alpha renderer is a port of
// water_postprocessing/cuda_contributions/kernels.cu; grid evidence follows body_vote.grid_evidence.

#include "gpu.hpp"

#include "error.hpp"
#include "types.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace ai3d::waterpost
{
namespace
{

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
    {
        throw WaterError(ErrorType::runtime, std::string("CUDA ") + what + ": " + cudaGetErrorString(status));
    }
}
#define WP_CUDA(call) check_cuda((call), #call)
#define WP_LAUNCH(what) check_cuda(cudaGetLastError(), what)

template <typename T>
struct Buffer
{
    T* p = nullptr;
    std::size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer()
    {
        release();
    }
    void release()
    {
        if (p)
        {
            cudaFree(p);
        }
        p = nullptr;
        n = 0;
    }
    void resize(std::size_t count)
    {
        if (count <= n && p)
        {
            return;
        }
        release();
        WP_CUDA(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)));
        n = count;
    }
    bool try_resize(std::size_t count)
    {
        if (count <= n && p)
        {
            return true;
        }
        release();
        if (cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)) != cudaSuccess)
        {
            (void)cudaGetLastError();
            p = nullptr;
            return false;
        }
        n = count;
        return true;
    }
    void upload(const T* host, std::size_t count)
    {
        resize(count);
        if (count)
        {
            WP_CUDA(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    void zero(std::size_t count)
    {
        if (count)
        {
            WP_CUDA(cudaMemset(p, 0, count * sizeof(T)));
        }
    }
    void download(T* host, std::size_t count) const
    {
        if (count)
        {
            WP_CUDA(cudaMemcpy(host, p, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
};

struct Cam
{
    int width, height;
    float fx, fy, cx, cy;
    float r[9];
    double c[3];
};

Cam device_camera(const PinholeCamera& camera)
{
    Cam out{};
    out.width = camera.width;
    out.height = camera.height;
    out.fx = static_cast<float>(camera.fx);
    out.fy = static_cast<float>(camera.fy);
    out.cx = static_cast<float>(camera.cx);
    out.cy = static_cast<float>(camera.cy);
    for (int k = 0; k < 9; ++k)
    {
        out.r[k] = static_cast<float>(camera.r[k]);
    }
    for (int k = 0; k < 3; ++k)
    {
        out.c[k] = camera.center[k];
    }
    return out;
}

__device__ __forceinline__ void to_camera(const Cam& cam, const double* p, float& x, float& y, float& z)
{
    const float dx = static_cast<float>(p[0] - cam.c[0]);
    const float dy = static_cast<float>(p[1] - cam.c[1]);
    const float dz = static_cast<float>(p[2] - cam.c[2]);
    x = cam.r[0] * dx + cam.r[1] * dy + cam.r[2] * dz;
    y = cam.r[3] * dx + cam.r[4] * dy + cam.r[5] * dz;
    z = cam.r[6] * dx + cam.r[7] * dy + cam.r[8] * dz;
}

__device__ __forceinline__ void rotate_covariance(const Cam& cam, const float* c, float m[3][3])
{
    const float s[3][3] = {{c[0], c[1], c[2]}, {c[1], c[3], c[4]}, {c[2], c[4], c[5]}};
    float left[3][3];
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            left[a][b] = cam.r[a * 3] * s[0][b] + cam.r[a * 3 + 1] * s[1][b] + cam.r[a * 3 + 2] * s[2][b];
        }
    }
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            m[a][b] = left[a][0] * cam.r[b * 3] + left[a][1] * cam.r[b * 3 + 1] + left[a][2] * cam.r[b * 3 + 2];
        }
    }
}

// ------------------------------------------------------------------ detect votes

__constant__ float c_nodes[65 * 2];
__constant__ float c_weights[65];

__global__ void vote_kernel(int n, const double* xyz, const float* cov, const float* sigma, Cam cam, int negative,
                            const std::uint8_t* state, const float* distance, const std::uint8_t* shore,
                            int* visible, int* water, int* land, int* center_visible, int* center_water,
                            int* boundary, std::uint8_t* in_frame, unsigned long long* counters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    in_frame[i] = 0;
    float px, py, pz;
    to_camera(cam, xyz + 3 * static_cast<std::size_t>(i), px, py, pz);
    if (!(pz > 1e-6f))
    {
        return;
    }
    const float u = cam.fx * px / pz + cam.cx - 0.5f;
    const float v = cam.fy * py / pz + cam.cy - 0.5f;
    if (!(u >= -0.5f && v >= -0.5f && u < cam.width - 0.5f && v < cam.height - 0.5f))
    {
        return;
    }
    in_frame[i] = 1;
    atomicAdd(counters, 1ull);
    if (negative)
    {
        visible[i] += 1;
        land[i] += 1;
        center_visible[i] += 1;
        return;
    }
    const int x = min(cam.width - 1, max(0, static_cast<int>(floorf(u + 0.5f))));
    const int y = min(cam.height - 1, max(0, static_cast<int>(floorf(v + 0.5f))));
    const std::size_t pixel = static_cast<std::size_t>(y) * cam.width + x;
    const std::uint8_t label = state[pixel];
    if (label != 2)
    {
        center_visible[i] += 1;
    }
    if (label == 1)
    {
        center_water[i] += 1;
    }
    if (shore[pixel] && xyz[3 * static_cast<std::size_t>(i) + 2] < cam.c[2])
    {
        boundary[i] += 1;
    }
    const float centre = distance[pixel];
    const float limit_x = 1.3f * cam.width / (2.0f * cam.fx);
    const float limit_y = 1.3f * cam.height / (2.0f * cam.fy);
    const float nx = fminf(fmaxf(px / pz, -limit_x), limit_x);
    const float ny = fminf(fmaxf(py / pz, -limit_y), limit_y);
    const float frobenius = sqrtf(cam.fx * cam.fx * (1 + nx * nx) + cam.fy * cam.fy * (1 + ny * ny)) / pz;
    const float bound = 3.0f * sigma[i] * frobenius * (1 + 1e-6f) + 1e-9f + 1.41421356f;
    if (centre > bound)
    {
        visible[i] += 1;
        water[i] += 1;
        atomicAdd(counters + 2, 1ull);
        return;
    }
    if (centre < -bound)
    {
        visible[i] += 1;
        land[i] += 1;
        atomicAdd(counters + 2, 1ull);
        return;
    }
    float m[3][3];
    rotate_covariance(cam, cov + 6 * static_cast<std::size_t>(i), m);
    const float j00 = cam.fx / pz, j02 = -cam.fx * nx / pz, j11 = cam.fy / pz, j12 = -cam.fy * ny / pz;
    const float c00 = j00 * j00 * m[0][0] + 2 * j00 * j02 * m[0][2] + j02 * j02 * m[2][2];
    const float c01 = j00 * j11 * m[0][1] + j00 * j12 * m[0][2] + j02 * j11 * m[2][1] + j02 * j12 * m[2][2];
    const float c11 = j11 * j11 * m[1][1] + 2 * j11 * j12 * m[1][2] + j12 * j12 * m[2][2];
    const float a = fmaxf(c00, 0.0f), b = c01, c = fmaxf(c11, 0.0f);
    const float radius = 3.0f * sqrtf(fmaxf(0.0f, (a + c) / 2 + sqrtf(((a - c) / 2) * ((a - c) / 2) + b * b)));
    if (centre > radius + 1.41421356f)
    {
        visible[i] += 1;
        water[i] += 1;
        return;
    }
    if (centre < -radius - 1.41421356f)
    {
        visible[i] += 1;
        land[i] += 1;
        return;
    }
    atomicAdd(counters + 1, 1ull);
    const float l00 = fmaxf(sqrtf(a), 1e-15f);
    const float l10 = b / l00;
    const float l11 = sqrtf(fmaxf(c - l10 * l10, 0.0f));
    float denominator = 0.0f;
    float wet = 0.0f;
    for (int k = 0; k < 65; ++k)
    {
        const float n0 = c_nodes[2 * k], n1 = c_nodes[2 * k + 1];
        const float qx = u + l00 * n0;
        const float qy = v + l10 * n0 + l11 * n1;
        const int ix = static_cast<int>(floorf(qx + 0.5f));
        const int iy = static_cast<int>(floorf(qy + 0.5f));
        if (ix < 0 || iy < 0 || ix >= cam.width || iy >= cam.height)
        {
            continue;
        }
        const std::uint8_t sample = state[static_cast<std::size_t>(iy) * cam.width + ix];
        if (sample == 2)
        {
            continue;
        }
        denominator += c_weights[k];
        if (sample == 1)
        {
            wet += c_weights[k];
        }
    }
    if (denominator > 0.0f)
    {
        const float fraction = wet / fmaxf(denominator, 1e-30f);
        visible[i] += 1;
        if (fraction >= 0.9f)
        {
            water[i] += 1;
        }
        if (fraction <= 0.1f)
        {
            land[i] += 1;
        }
    }
}

// ------------------------------------------------------------------ T*alpha renderer

constexpr int TILE = 16, THREADS = 256;
constexpr float MIN_ALPHA = 1.0f / 255.0f;

struct RCam
{
    int width, height;
    float focal_x, focal_y, tan_x, tan_y, px, py, ox, oy, kernel, near_z;
    Cam view;
};

__device__ __forceinline__ void rectangle(float2 center, int radius, int gx, int gy, uint2& low, uint2& high)
{
    low = {std::uint32_t(min(gx, max(0, int((center.x - radius) / TILE)))),
           std::uint32_t(min(gy, max(0, int((center.y - radius) / TILE))))};
    high = {std::uint32_t(min(gx, max(0, int((center.x + radius + TILE - 1) / TILE)))),
            std::uint32_t(min(gy, max(0, int((center.y + radius + TILE - 1) / TILE))))};
}

__global__ void project_kernel(int n, const double* xyz, const float* covariance, const float* opacity, RCam camera,
                               float2* centers, float4* conics, float* depths, int* radii)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    radii[i] = 0;
    centers[i] = {0, 0};
    conics[i] = {0, 0, 0, 0};
    depths[i] = 0;
    float x, y, z;
    to_camera(camera.view, xyz + 3 * static_cast<std::size_t>(i), x, y, z);
    if (!(z > camera.near_z))
    {
        return;
    }
    float sigma[3][3];
    rotate_covariance(camera.view, covariance + 6 * static_cast<std::size_t>(i), sigma);
    const float clipped_x = min(1.3f * camera.tan_x, max(-1.3f * camera.tan_x, x / z)) * z;
    const float clipped_y = min(1.3f * camera.tan_y, max(-1.3f * camera.tan_y, y / z)) * z;
    const float j[3][3] = {{camera.focal_x / z, 0.0f, -(camera.focal_x * clipped_x) / (z * z)},
                           {0.0f, camera.focal_y / z, -(camera.focal_y * clipped_y) / (z * z)},
                           {0, 0, 0}};
    float left[2][3];
    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 3; b++)
        {
            left[a][b] = j[a][0] * sigma[0][b] + j[a][1] * sigma[1][b] + j[a][2] * sigma[2][b];
        }
    }
    const float c00 = left[0][0] * j[0][0] + left[0][1] * j[0][1] + left[0][2] * j[0][2];
    const float c01 = left[1][0] * j[0][0] + left[1][1] * j[0][1] + left[1][2] * j[0][2];
    const float c11 = left[1][0] * j[1][0] + left[1][1] * j[1][1] + left[1][2] * j[1][2];
    const float a = c00 + camera.kernel, b = c01, d = c11 + camera.kernel;
    const float det0 = max(1e-6f, c00 * c11 - c01 * c01), det1 = max(1e-6f, a * d - b * b);
    float coefficient = sqrtf(det0 / (det1 + 1e-6f) + 1e-6f);
    if (det0 <= 1e-6f || det1 <= 1e-6f)
    {
        coefficient = 0.0f;
    }
    const float determinant = a * d - b * b;
    if (determinant == 0.0f)
    {
        return;
    }
    const float inverse = 1.0f / determinant;
    const float middle = 0.5f * (a + d), eigen_delta = sqrtf(max(0.1f, middle * middle - determinant));
    const float radius = ceilf(3.0f * sqrtf(max(middle + eigen_delta, middle - eigen_delta)));
    if (!isfinite(radius) || radius > 2147483000.0f)
    {
        return;
    }
    const float inv_w = 1.0f / (z + 0.0000001f);
    const float nx = (camera.px * x + camera.ox * z) * inv_w;
    const float ny = (camera.py * y + camera.oy * z) * inv_w;
    const float2 mean = {float(((nx + 1.0) * camera.width - 1.0) * 0.5), float(((ny + 1.0) * camera.height - 1.0) * 0.5)};
    uint2 low, high;
    rectangle(mean, int(radius), (camera.width + 15) / 16, (camera.height + 15) / 16, low, high);
    if (high.x <= low.x || high.y <= low.y)
    {
        return;
    }
    centers[i] = mean;
    depths[i] = z;
    radii[i] = int(radius);
    conics[i] = {d * inverse, -b * inverse, a * inverse, opacity[i] * coefficient};
}

__device__ __forceinline__ float half_power(float3 q, float x, float y)
{
    return 0.5f * (q.x * x * x + q.z * y * y) + q.y * x * y;
}
__device__ __forceinline__ float segment_power(float2 mean, float3 q, float2 a, float2 b)
{
    const float ux = mean.x - a.x, uy = mean.y - a.y, vx = b.x - a.x, vy = b.y - a.y;
    const float vv = q.x * vx * vx + 2.0f * q.y * vx * vy + q.z * vy * vy;
    const float vu = q.x * vx * ux + q.y * (vx * uy + vy * ux) + q.z * vy * uy;
    const float t = vv > 0.0f ? fminf(fmaxf(vu / vv, 0.0f), 1.0f) : 0.0f;
    return half_power(q, ux - t * vx, uy - t * vy);
}
__device__ __noinline__ float rectangle_power(float2 mean, float3 q, float2 a, float2 b)
{
    if (mean.x >= a.x && mean.x <= b.x && mean.y >= a.y && mean.y <= b.y)
    {
        return 0.0f;
    }
    const float2 c = {b.x, a.y}, d = {a.x, b.y};
    float value = segment_power(mean, q, a, c);
    value = fminf(value, segment_power(mean, q, d, b));
    value = fminf(value, segment_power(mean, q, a, d));
    return fminf(value, segment_power(mean, q, c, b));
}
__device__ __forceinline__ float threshold(float opacity)
{
    return logf(opacity * 255.0f) + 1e-3f;
}
__device__ __forceinline__ bool touches(float2 mean, float4 q, float limit, int tx, int ty)
{
    if (limit == INFINITY)
    {
        return true;
    }
    const float2 low = {float(tx * TILE), float(ty * TILE)}, high = {low.x + 15.0f, low.y + 15.0f};
    return rectangle_power(mean, {q.x, q.y, q.z}, low, high) <= limit;
}

struct Region
{
    int x0, y0, x1, y1, gx, gy;
};

template <bool EMIT>
__global__ void bin_kernel(int n, const float2* centers, const float4* conics, const float* depths, const int* radii,
                           int gx, int gy, Region roi, const long long* offsets, long long* counts,
                           unsigned long long* keys, std::uint32_t* ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    if (!EMIT)
    {
        counts[i] = 0;
    }
    if (radii[i] <= 0)
    {
        return;
    }
    const float2 mean = centers[i];
    const float4 q = conics[i];
    if (!(q.w >= MIN_ALPHA))
    {
        return;
    }
    uint2 low, high;
    rectangle(mean, radii[i], gx, gy, low, high);
    float limit = INFINITY;
    const float determinant = q.x * q.z - q.y * q.y;
    if (determinant > 0.0f)
    {
        const float t = threshold(q.w), ex = sqrtf(2.0f * t * q.z / determinant) + 1e-3f,
                    ey = sqrtf(2.0f * t * q.x / determinant) + 1e-3f;
        if (ex < 1e7f && ey < 1e7f)
        {
            limit = t;
            low.x = max(low.x, (std::uint32_t)fminf(fmaxf(floorf(ceilf(mean.x - ex) / TILE), 0.0f), float(gx)));
            high.x = min(high.x, (std::uint32_t)fminf(fmaxf(floorf(floorf(mean.x + ex) / TILE) + 1.0f, 0.0f), float(gx)));
            low.y = max(low.y, (std::uint32_t)fminf(fmaxf(floorf(ceilf(mean.y - ey) / TILE), 0.0f), float(gy)));
            high.y = min(high.y, (std::uint32_t)fminf(fmaxf(floorf(floorf(mean.y + ey) / TILE) + 1.0f, 0.0f), float(gy)));
        }
    }
    low.x = max(low.x, (std::uint32_t)roi.x0);
    low.y = max(low.y, (std::uint32_t)roi.y0);
    high.x = min(high.x, (std::uint32_t)roi.x1);
    high.y = min(high.y, (std::uint32_t)roi.y1);
    if (low.x >= high.x || low.y >= high.y)
    {
        return;
    }
    long long count = 0, at = EMIT ? (i ? offsets[i - 1] : 0) : 0;
    for (std::uint32_t y = low.y; y < high.y; y++)
    {
        for (std::uint32_t x = low.x; x < high.x; x++)
        {
            if (!touches(mean, q, limit, x, y))
            {
                continue;
            }
            if (EMIT)
            {
                const std::uint32_t tile = (y - roi.y0) * roi.gx + (x - roi.x0);
                keys[at] = ((unsigned long long)tile << 32) | __float_as_uint(depths[i]);
                ids[at] = i;
                at++;
            }
            count++;
        }
    }
    if (!EMIT)
    {
        counts[i] = count;
    }
}

__global__ void tile_ranges(int n, const unsigned long long* keys, uint2* ranges)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (std::uint32_t)n)
    {
        return;
    }
    const std::uint32_t tile = keys[i] >> 32;
    if (i == 0)
    {
        ranges[tile].x = 0;
    }
    else
    {
        const std::uint32_t previous = keys[i - 1] >> 32;
        if (tile != previous)
        {
            ranges[previous].y = i;
            ranges[tile].x = i;
        }
    }
    if (i == (std::uint32_t)n - 1)
    {
        ranges[tile].y = n;
    }
}

__global__ void blend_kernel(const uint2* ranges, const std::uint32_t* ids, const float2* centers, const float4* conics,
                             const std::uint8_t* state, int width, int height, Region roi, float* mass)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const std::uint32_t tid = block.thread_rank(), lane = tid % 32, warp_id = tid / 32;
    const std::uint32_t tile = blockIdx.y * roi.gx + blockIdx.x;
    const std::uint32_t left = (roi.x0 + blockIdx.x) * TILE, top = (roi.y0 + blockIdx.y) * TILE;
    const std::uint32_t sub_x = (warp_id % 2) * 8, sub_y = (warp_id / 2) * 4;
    const std::uint32_t x = left + sub_x + lane % 8, y = top + sub_y + lane / 8;
    const bool inside = x < (std::uint32_t)width && y < (std::uint32_t)height;
    const std::uint8_t label = inside ? (state ? state[(std::size_t)y * width + x] : 0) : 2;
    const bool has_water = warp.any(inside && label == 1), has_land = warp.any(inside && label == 0),
               has_unknown = warp.any(inside && label == 2);
    const float2 sub_low = {float(left + sub_x), float(top + sub_y)}, sub_high = {sub_low.x + 7.0f, sub_low.y + 3.0f};
    bool done = !inside;
    float T = 1.0f;
    const uint2 range = ranges[tile];
    const std::uint32_t total = range.y - range.x;
    __shared__ std::uint32_t batch_ids[THREADS];
    __shared__ float2 batch_xy[THREADS];
    __shared__ float4 batch_q[THREADS];
    for (std::uint32_t base = 0; base < total; base += THREADS)
    {
        if (__syncthreads_count(done) == THREADS)
        {
            break;
        }
        if (base + tid < total)
        {
            const std::uint32_t id = ids[range.x + base + tid];
            batch_ids[tid] = id;
            batch_xy[tid] = centers[id];
            batch_q[tid] = conics[id];
        }
        block.sync();
        const int size = min((std::uint32_t)THREADS, total - base);
        for (int c = 0; c < size; c += 32)
        {
            const int e = c + lane;
            bool candidate = false;
            if (e < size)
            {
                const float4 q = batch_q[e];
                candidate = !(q.x * q.z - q.y * q.y > 0.0f) ||
                            rectangle_power(batch_xy[e], {q.x, q.y, q.z}, sub_low, sub_high) <= threshold(q.w);
            }
            std::uint32_t pending = warp.ballot(candidate);
            while (pending && warp.any(!done))
            {
                const int j = c + __ffs(pending) - 1;
                pending &= pending - 1;
                float weight = 0.0f;
                if (!done)
                {
                    const float2 d = {batch_xy[j].x - float(x), batch_xy[j].y - float(y)};
                    const float4 q = batch_q[j];
                    const float power = -0.5f * (q.x * d.x * d.x + q.z * d.y * d.y) - q.y * d.x * d.y;
                    if (power <= 0.0f)
                    {
                        const float alpha = min(0.99f, q.w * exp(power));
                        if (alpha >= MIN_ALPHA)
                        {
                            const float next = T * (1 - alpha);
                            if (next < 0.0001f)
                            {
                                done = true;
                            }
                            else
                            {
                                weight = alpha * T;
                                T = next;
                            }
                        }
                    }
                }
                const float wet = has_water ? cg::reduce(warp, label == 1 ? weight : 0.0f, cg::plus<float>()) : 0.0f;
                const float dry = has_land ? cg::reduce(warp, label == 0 ? weight : 0.0f, cg::plus<float>()) : 0.0f;
                const float unknown = has_unknown ? cg::reduce(warp, label == 2 ? weight : 0.0f, cg::plus<float>()) : 0.0f;
                if (lane == 0)
                {
                    const std::size_t at = (std::size_t)batch_ids[j] * 3;
                    if (wet)
                    {
                        atomicAdd(mass + at, wet);
                    }
                    if (dry)
                    {
                        atomicAdd(mass + at + 1, dry);
                    }
                    if (unknown)
                    {
                        atomicAdd(mass + at + 2, unknown);
                    }
                }
            }
        }
    }
}

__global__ void color_kernel(const uint2* ranges, const std::uint32_t* ids, const float2* centers, const float4* conics,
                             const float3* colors, int width, int height, Region roi, float* image)
{
    auto block = cg::this_thread_block();
    const std::uint32_t tile = blockIdx.y * roi.gx + blockIdx.x;
    const std::uint32_t x = (roi.x0 + blockIdx.x) * TILE + block.thread_index().x;
    const std::uint32_t y = (roi.y0 + blockIdx.y) * TILE + block.thread_index().y;
    const bool inside = x < (std::uint32_t)width && y < (std::uint32_t)height;
    bool done = !inside;
    const uint2 range = ranges[tile];
    const int total = range.y - range.x;
    __shared__ float2 batch_xy[THREADS];
    __shared__ float4 batch_q[THREADS];
    __shared__ float3 batch_c[THREADS];
    const int tid = block.thread_rank();
    float T = 1.0f;
    float3 c = {0, 0, 0};
    for (int base = 0; base < total; base += THREADS)
    {
        if (__syncthreads_count(done) == THREADS)
        {
            break;
        }
        if (base + tid < total)
        {
            const std::uint32_t id = ids[range.x + base + tid];
            batch_xy[tid] = centers[id];
            batch_q[tid] = conics[id];
            batch_c[tid] = colors[id];
        }
        block.sync();
        const int size = min(THREADS, total - base);
        for (int j = 0; !done && j < size; ++j)
        {
            const float2 d = {batch_xy[j].x - float(x), batch_xy[j].y - float(y)};
            const float4 q = batch_q[j];
            const float power = -0.5f * (q.x * d.x * d.x + q.z * d.y * d.y) - q.y * d.x * d.y;
            if (power > 0.0f)
            {
                continue;
            }
            const float alpha = min(0.99f, q.w * exp(power));
            if (alpha < MIN_ALPHA)
            {
                continue;
            }
            const float next = T * (1 - alpha);
            if (next < 0.0001f)
            {
                done = true;
                continue;
            }
            c.x += batch_c[j].x * alpha * T;
            c.y += batch_c[j].y * alpha * T;
            c.z += batch_c[j].z * alpha * T;
            T = next;
        }
        block.sync();
    }
    if (inside)
    {
        const std::size_t at = (static_cast<std::size_t>(y) * width + x) * 3;
        image[at] = c.x;
        image[at + 1] = c.y;
        image[at + 2] = c.z;
    }
}

__global__ void support_kernel(int n, const std::uint8_t* in_frame, const float* mass, double* total, int* water_views)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !in_frame[i])
    {
        return;
    }
    const double w = mass[3 * static_cast<std::size_t>(i)], l = mass[3 * static_cast<std::size_t>(i) + 1],
                 u = mass[3 * static_cast<std::size_t>(i) + 2];
    const double known = w + l;
    if (known > 0 && w >= 0.9 * known)
    {
        water_views[i] += 1;
    }
    total[3 * static_cast<std::size_t>(i)] += w;
    total[3 * static_cast<std::size_t>(i) + 1] += l;
    total[3 * static_cast<std::size_t>(i) + 2] += u;
}

// ------------------------------------------------------------------ grid evidence

__device__ __forceinline__ bool grid_project(const Cam& cam, const double* p, int& x, int& y, float& depth)
{
    float cx, cy, cz;
    to_camera(cam, p, cx, cy, cz);
    depth = cz;
    if (!(cz > 1e-6f))
    {
        return false;
    }
    const float u = cam.fx * cx / cz + cam.cx;
    const float v = cam.fy * cy / cz + cam.cy;
    if (!(u >= 0.0f && v >= 0.0f && u < float(cam.width) && v < float(cam.height)))
    {
        return false;
    }
    x = min(cam.width - 1, static_cast<int>(u));
    y = min(cam.height - 1, static_cast<int>(v));
    return true;
}

__device__ __forceinline__ bool occupancy_bit(const std::uint8_t* bits, int cols, int x, int y)
{
    const std::size_t bit = static_cast<std::size_t>(y / OCCUPANCY_CELL) * cols + x / OCCUPANCY_CELL;
    return (bits[bit >> 3] >> (7 - (bit & 7))) & 1u;
}

__global__ void possible_kernel(int m, const double* cells, Cam cam, const std::uint8_t* bits, int cols,
                                std::uint32_t* support)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
    {
        return;
    }
    int x, y;
    float depth;
    if (!grid_project(cam, cells + 3 * static_cast<std::size_t>(i), x, y, depth))
    {
        return;
    }
    if (!bits || occupancy_bit(bits, cols, x, y))
    {
        support[i] += 1;
    }
}

__global__ void exact_kernel(int m, const double* cells, Cam cam, const std::uint8_t* image, std::uint32_t* support)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
    {
        return;
    }
    int x, y;
    float depth;
    if (grid_project(cam, cells + 3 * static_cast<std::size_t>(i), x, y, depth) &&
        image[static_cast<std::size_t>(y) * cam.width + x] == 1)
    {
        support[i] += 1;
    }
}

__global__ void zclear_kernel(int count, unsigned int* buffer)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        buffer[i] = 0x7f800000u; // +inf
    }
}

__global__ void zscatter_kernel(int k, const double* points, Cam cam, int factor, int zw, unsigned int* buffer)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
    {
        return;
    }
    int x, y;
    float depth;
    if (grid_project(cam, points + 3 * static_cast<std::size_t>(i), x, y, depth))
    {
        atomicMin(buffer + static_cast<std::size_t>(y / factor) * zw + x / factor, __float_as_uint(depth));
    }
}

__global__ void zpool_kernel(int zw, int zh, const unsigned int* buffer, float* pooled)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= zw * zh)
    {
        return;
    }
    const int x = i % zw, y = i / zw;
    unsigned int best = 0x7f800000u;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int xx = x + dx, yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < zw && yy < zh)
            {
                best = min(best, buffer[yy * zw + xx]);
            }
        }
    }
    pooled[i] = __uint_as_float(best);
}

__global__ void signature_kernel(int m, const double* cells, Cam cam, const std::uint8_t* image, const float* pooled,
                                 int factor, int zw, float tolerance, std::uint32_t* support, std::uint32_t* against,
                                 unsigned long long* occluded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
    {
        return;
    }
    int x, y;
    float depth;
    if (!grid_project(cam, cells + 3 * static_cast<std::size_t>(i), x, y, depth))
    {
        return;
    }
    if (pooled && pooled[static_cast<std::size_t>(y / factor) * zw + x / factor] < depth - tolerance)
    {
        atomicAdd(occluded, 1ull);
        return;
    }
    const std::uint8_t label = image[static_cast<std::size_t>(y) * cam.width + x];
    if (label == 1)
    {
        support[i] += 1;
    }
    else if (label == 0)
    {
        against[i] += 1;
    }
}

__global__ void other_kernel(int m, const double* cells, Cam cam, int dry, const std::uint8_t* bits, int cols,
                             const float* pooled, int factor, int zw, float tolerance, std::uint32_t* against,
                             unsigned long long* occluded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
    {
        return;
    }
    int x, y;
    float depth;
    if (!grid_project(cam, cells + 3 * static_cast<std::size_t>(i), x, y, depth))
    {
        return;
    }
    if (pooled && pooled[static_cast<std::size_t>(y / factor) * zw + x / factor] < depth - tolerance)
    {
        atomicAdd(occluded, 1ull);
        return;
    }
    if (dry || !occupancy_bit(bits, cols, x, y))
    {
        against[i] += 1;
    }
}

int blocks_for(std::size_t count)
{
    return static_cast<int>((count + 255) / 256);
}

} // namespace

void gpu_select_device(int device)
{
    WP_CUDA(cudaSetDevice(device));
}

std::size_t gpu_free_bytes()
{
    std::size_t free = 0, total = 0;
    WP_CUDA(cudaMemGetInfo(&free, &total));
    return free;
}

// ====================================================================== DetectGpu

struct DetectGpu::Impl
{
    int n = 0;
    double unit = 1.0;
    Buffer<double> xyz;
    Buffer<float> cov, sigma, opacity;
    Buffer<int> visible, water, land, center_visible, center_water, boundary, water_views;
    Buffer<std::uint8_t> in_frame;
    Buffer<double> mass_total;
    Buffer<unsigned long long> counters;
    Buffer<std::uint8_t> state;
    Buffer<float> distance;
    Buffer<std::uint8_t> shore;
    // renderer
    Buffer<float2> centers;
    Buffer<float4> conics;
    Buffer<float> depths;
    Buffer<int> radii;
    Buffer<float> mass;
    Buffer<long long> counts, offsets;
    Buffer<unsigned long long> keys, sorted_keys;
    Buffer<std::uint32_t> ids, sorted_ids;
    Buffer<std::uint8_t> temp;
    Buffer<uint2> ranges;
    std::size_t boundary_checks = 0, early = 0, subdivided = 0;

    bool accumulate(int width, int height, int x0, int y0, int x1, int y1, const std::uint8_t* labels, bool can_split);
};

DetectGpu::DetectGpu(const Gaussians& gaussians, double unit) : m(std::make_unique<Impl>())
{
    if (gaussians.count > static_cast<std::size_t>(std::numeric_limits<int>::max() - 255))
    {
        fail_value("Too many Gaussians in one block for the CUDA voter");
    }
    m->n = static_cast<int>(gaussians.count);
    m->unit = unit;
    const std::size_t n = gaussians.count;
    m->xyz.upload(gaussians.xyz.data(), 3 * n);
    m->cov.upload(gaussians.cov.data(), 6 * n);
    m->sigma.upload(gaussians.sigma_max.data(), n);
    m->opacity.upload(gaussians.opacity.data(), n);
    for (Buffer<int>* b : {&m->visible, &m->water, &m->land, &m->center_visible, &m->center_water, &m->boundary,
                           &m->water_views})
    {
        b->resize(n);
        b->zero(n);
    }
    m->in_frame.resize(n);
    m->in_frame.zero(n);
    m->mass_total.resize(3 * n);
    m->mass_total.zero(3 * n);
    m->counters.resize(4);
    // Quadrature: centre plus 4 rings of 16 nodes, weights integrate the circular Gaussian.
    const double radii[5] = {0., .75, 1.5, 2.25, 3.};
    double edges[6] = {0, 0, 0, 0, 0, 3.};
    for (int k = 1; k < 5; ++k)
    {
        edges[k] = (radii[k - 1] + radii[k]) / 2;
    }
    double ring_mass[5];
    double sum = 0;
    for (int k = 0; k < 5; ++k)
    {
        ring_mass[k] = std::exp(-edges[k] * edges[k] / 2) - std::exp(-edges[k + 1] * edges[k + 1] / 2);
        sum += ring_mass[k];
    }
    float nodes[130];
    float weights[65];
    nodes[0] = nodes[1] = 0;
    weights[0] = static_cast<float>(ring_mass[0] / sum);
    int at = 1;
    for (int k = 1; k < 5; ++k)
    {
        for (int a = 0; a < 16; ++a)
        {
            const double angle = (a + .5) * (2 * M_PI / 16);
            nodes[2 * at] = static_cast<float>(radii[k] * std::cos(angle));
            nodes[2 * at + 1] = static_cast<float>(radii[k] * std::sin(angle));
            weights[at] = static_cast<float>(ring_mass[k] / 16 / sum);
            ++at;
        }
    }
    WP_CUDA(cudaMemcpyToSymbol(c_nodes, nodes, sizeof(nodes)));
    WP_CUDA(cudaMemcpyToSymbol(c_weights, weights, sizeof(weights)));
    m->centers.resize(n);
    m->conics.resize(n);
    m->depths.resize(n);
    m->radii.resize(n);
    m->mass.resize(3 * n);
    m->counts.resize(n);
    m->offsets.resize(n);
}

DetectGpu::~DetectGpu() = default;

std::size_t DetectGpu::vote(const PinholeCamera& camera, bool negative_only, const std::uint8_t* state,
                            const float* distance, const std::uint8_t* shoreline)
{
    const std::size_t pixels = static_cast<std::size_t>(camera.width) * camera.height;
    if (!negative_only)
    {
        m->state.upload(state, pixels);
        m->distance.upload(distance, pixels);
        m->shore.upload(shoreline, pixels);
    }
    m->counters.zero(4);
    if (m->n)
    {
        vote_kernel<<<blocks_for(m->n), 256>>>(m->n, m->xyz.p, m->cov.p, m->sigma.p, device_camera(camera),
                                               negative_only ? 1 : 0, negative_only ? nullptr : m->state.p,
                                               negative_only ? nullptr : m->distance.p, negative_only ? nullptr : m->shore.p,
                                               m->visible.p, m->water.p, m->land.p, m->center_visible.p,
                                               m->center_water.p, m->boundary.p, m->in_frame.p, m->counters.p);
        WP_LAUNCH("vote_kernel");
    }
    unsigned long long counters[4] = {};
    m->counters.download(counters, 4);
    m->boundary_checks += counters[1];
    m->early += counters[2];
    return static_cast<std::size_t>(counters[0]);
}

bool DetectGpu::Impl::accumulate(int width, int height, int x0, int y0, int x1, int y1, const std::uint8_t* labels,
                                 bool can_split)
{
    const Region roi = {x0 / 16, y0 / 16, (x1 + 15) / 16, (y1 + 15) / 16, (x1 + 15) / 16 - x0 / 16, (y1 + 15) / 16 - y0 / 16};
    const int gx = (width + 15) / 16, gy = (height + 15) / 16;
    bin_kernel<false><<<blocks_for(n), 256>>>(n, centers.p, conics.p, depths.p, radii.p, gx, gy, roi, nullptr, counts.p,
                                              nullptr, nullptr);
    WP_LAUNCH("bin_kernel(count)");
    std::size_t scan_bytes = 0;
    WP_CUDA(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, counts.p, offsets.p, n));
    temp.resize(scan_bytes);
    WP_CUDA(cub::DeviceScan::InclusiveSum(temp.p, scan_bytes, counts.p, offsets.p, n));
    long long pairs = 0;
    WP_CUDA(cudaMemcpy(&pairs, offsets.p + n - 1, sizeof(pairs), cudaMemcpyDeviceToHost));
    if (pairs <= 0)
    {
        return true;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    WP_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t reserved = keys.n * 24 + temp.n;
    const std::size_t available = std::min<std::size_t>((free_bytes + reserved) / 2, std::size_t{3} << 30);
    const std::size_t budget = std::max<std::size_t>(1, available / 64);
    if (pairs >= std::numeric_limits<int>::max() || (can_split && static_cast<std::size_t>(pairs) > budget))
    {
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(pairs);
    if (!keys.try_resize(count) || !sorted_keys.try_resize(count) || !ids.try_resize(count) || !sorted_ids.try_resize(count))
    {
        if (can_split)
        {
            return false;
        }
        fail_runtime("CUDA memory exhausted for an indivisible 16 px semantic tile");
    }
    bin_kernel<true><<<blocks_for(n), 256>>>(n, centers.p, conics.p, depths.p, radii.p, gx, gy, roi, offsets.p, nullptr,
                                             keys.p, ids.p);
    WP_LAUNCH("bin_kernel(emit)");
    std::size_t sort_bytes = 0;
    WP_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys.p, sorted_keys.p, ids.p, sorted_ids.p,
                                            static_cast<int>(count), 0, 64));
    if (!temp.try_resize(sort_bytes))
    {
        if (can_split)
        {
            return false;
        }
        fail_runtime("CUDA memory exhausted while sorting semantic tiles");
    }
    WP_CUDA(cub::DeviceRadixSort::SortPairs(temp.p, sort_bytes, keys.p, sorted_keys.p, ids.p, sorted_ids.p,
                                            static_cast<int>(count), 0, 64));
    const std::size_t tiles = static_cast<std::size_t>(roi.gx) * roi.gy;
    ranges.resize(tiles);
    WP_CUDA(cudaMemset(ranges.p, 0, tiles * sizeof(uint2)));
    tile_ranges<<<blocks_for(count), 256>>>(static_cast<int>(count), sorted_keys.p, ranges.p);
    WP_LAUNCH("tile_ranges");
    blend_kernel<<<dim3(roi.gx, roi.gy), 256>>>(ranges.p, sorted_ids.p, centers.p, conics.p, labels, width, height, roi,
                                                mass.p);
    WP_LAUNCH("blend_kernel");
    return true;
}

void DetectGpu::render(const PinholeCamera& camera, const std::uint8_t* state)
{
    if (!m->n)
    {
        return;
    }
    const std::size_t pixels = static_cast<std::size_t>(camera.width) * camera.height;
    const std::uint8_t* labels = nullptr;
    if (state)
    {
        m->state.upload(state, pixels);
        labels = m->state.p;
    }
    RCam rc{};
    rc.width = camera.width;
    rc.height = camera.height;
    rc.tan_x = static_cast<float>(camera.width / (2 * camera.fx));
    rc.tan_y = static_cast<float>(camera.height / (2 * camera.fy));
    rc.focal_x = camera.width / (2.0f * rc.tan_x);
    rc.focal_y = camera.height / (2.0f * rc.tan_y);
    rc.px = static_cast<float>(2 * camera.fx / camera.width);
    rc.py = static_cast<float>(2 * camera.fy / camera.height);
    rc.ox = static_cast<float>(2 * camera.cx / camera.width - 1);
    rc.oy = static_cast<float>(2 * camera.cy / camera.height - 1);
    rc.kernel = 0.0f;
    rc.near_z = static_cast<float>(0.2 * m->unit);
    rc.view = device_camera(camera);
    project_kernel<<<blocks_for(m->n), 256>>>(m->n, m->xyz.p, m->cov.p, m->opacity.p, rc, m->centers.p, m->conics.p,
                                              m->depths.p, m->radii.p);
    WP_LAUNCH("project_kernel");
    m->mass.zero(3 * static_cast<std::size_t>(m->n));
    std::vector<std::array<int, 4>> pending = {{0, 0, camera.width, camera.height}};
    while (!pending.empty())
    {
        const auto [x0, y0, x1, y1] = pending.back();
        pending.pop_back();
        const bool can_split = std::max(x1 - x0, y1 - y0) > 16;
        if (m->accumulate(camera.width, camera.height, x0, y0, x1, y1, labels, can_split))
        {
            continue;
        }
        ++m->subdivided;
        if (x1 - x0 >= y1 - y0)
        {
            const int mid = std::max(x0 + 16, ((x0 + x1) / 2 / 16) * 16);
            pending.push_back({mid, y0, x1, y1});
            pending.push_back({x0, y0, mid, y1});
        }
        else
        {
            const int mid = std::max(y0 + 16, ((y0 + y1) / 2 / 16) * 16);
            pending.push_back({x0, mid, x1, y1});
            pending.push_back({x0, y0, x1, mid});
        }
    }
    support_kernel<<<blocks_for(m->n), 256>>>(m->n, m->in_frame.p, m->mass.p, m->mass_total.p, m->water_views.p);
    WP_LAUNCH("support_kernel");
}

DetectGpu::Result DetectGpu::download()
{
    Result r;
    const std::size_t n = static_cast<std::size_t>(m->n);
    for (auto [buffer, out] : {std::make_pair(&m->visible, &r.visible), std::make_pair(&m->water, &r.water),
                               std::make_pair(&m->land, &r.land), std::make_pair(&m->center_visible, &r.center_visible),
                               std::make_pair(&m->center_water, &r.center_water), std::make_pair(&m->boundary, &r.boundary),
                               std::make_pair(&m->water_views, &r.water_views)})
    {
        out->resize(n);
        buffer->download(out->data(), n);
    }
    r.mass.resize(3 * n);
    m->mass_total.download(r.mass.data(), 3 * n);
    r.boundary_checks = m->boundary_checks;
    r.early_uniform = m->early;
    r.subdivided_tiles = m->subdivided;
    return r;
}

// ====================================================================== ColorRenderer

struct ColorRenderer::Impl
{
    int n = 0;
    double unit = 1.0;
    Buffer<double> xyz;
    Buffer<float> cov, opacity;
    Buffer<float3> colors;
    Buffer<float2> centers;
    Buffer<float4> conics;
    Buffer<float> depths;
    Buffer<int> radii;
    Buffer<long long> counts, offsets;
    Buffer<unsigned long long> keys, sorted_keys;
    Buffer<std::uint32_t> ids, sorted_ids;
    Buffer<std::uint8_t> temp;
    Buffer<uint2> ranges;
    Buffer<float> image;

    bool draw(int width, int height, int x0, int y0, int x1, int y1, bool can_split)
    {
        const Region roi = {x0 / 16, y0 / 16, (x1 + 15) / 16, (y1 + 15) / 16, (x1 + 15) / 16 - x0 / 16, (y1 + 15) / 16 - y0 / 16};
        const int gx = (width + 15) / 16, gy = (height + 15) / 16;
        bin_kernel<false><<<blocks_for(n), 256>>>(n, centers.p, conics.p, depths.p, radii.p, gx, gy, roi, nullptr, counts.p, nullptr, nullptr);
        WP_LAUNCH("bin_kernel(count)");
        std::size_t scan_bytes = 0;
        WP_CUDA(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, counts.p, offsets.p, n));
        temp.resize(scan_bytes);
        WP_CUDA(cub::DeviceScan::InclusiveSum(temp.p, scan_bytes, counts.p, offsets.p, n));
        long long pairs = 0;
        WP_CUDA(cudaMemcpy(&pairs, offsets.p + n - 1, sizeof(pairs), cudaMemcpyDeviceToHost));
        if (pairs <= 0)
        {
            return true;
        }
        std::size_t free_bytes = 0, total_bytes = 0;
        WP_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t budget = std::max<std::size_t>(1, std::min<std::size_t>((free_bytes + keys.n * 24 + temp.n) / 2, std::size_t{3} << 30) / 64);
        if (pairs >= std::numeric_limits<int>::max() || (can_split && static_cast<std::size_t>(pairs) > budget))
        {
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(pairs);
        if (!keys.try_resize(count) || !sorted_keys.try_resize(count) || !ids.try_resize(count) || !sorted_ids.try_resize(count))
        {
            if (can_split)
            {
                return false;
            }
            fail_runtime("CUDA memory exhausted for a 16 px colour tile");
        }
        bin_kernel<true><<<blocks_for(n), 256>>>(n, centers.p, conics.p, depths.p, radii.p, gx, gy, roi, offsets.p, nullptr, keys.p, ids.p);
        WP_LAUNCH("bin_kernel(emit)");
        std::size_t sort_bytes = 0;
        WP_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys.p, sorted_keys.p, ids.p, sorted_ids.p, static_cast<int>(count), 0, 64));
        if (!temp.try_resize(sort_bytes))
        {
            if (can_split)
            {
                return false;
            }
            fail_runtime("CUDA memory exhausted while sorting colour tiles");
        }
        WP_CUDA(cub::DeviceRadixSort::SortPairs(temp.p, sort_bytes, keys.p, sorted_keys.p, ids.p, sorted_ids.p, static_cast<int>(count), 0, 64));
        const std::size_t tiles = static_cast<std::size_t>(roi.gx) * roi.gy;
        ranges.resize(tiles);
        WP_CUDA(cudaMemset(ranges.p, 0, tiles * sizeof(uint2)));
        tile_ranges<<<blocks_for(count), 256>>>(static_cast<int>(count), sorted_keys.p, ranges.p);
        WP_LAUNCH("tile_ranges");
        color_kernel<<<dim3(roi.gx, roi.gy), dim3(16, 16)>>>(ranges.p, sorted_ids.p, centers.p, conics.p, colors.p, width, height, roi, image.p);
        WP_LAUNCH("color_kernel");
        return true;
    }
};

ColorRenderer::ColorRenderer(const Gaussians& gaussians, double unit) : m(std::make_unique<Impl>())
{
    m->n = static_cast<int>(gaussians.count);
    m->unit = unit;
    const std::size_t n = gaussians.count;
    m->xyz.upload(gaussians.xyz.data(), 3 * n);
    m->cov.upload(gaussians.cov.data(), 6 * n);
    m->opacity.upload(gaussians.opacity.data(), n);
    std::vector<float3> colors(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        colors[i] = {gaussians.rgba[4 * i] / 255.0f, gaussians.rgba[4 * i + 1] / 255.0f, gaussians.rgba[4 * i + 2] / 255.0f};
    }
    m->colors.upload(colors.data(), n);
    m->centers.resize(n);
    m->conics.resize(n);
    m->depths.resize(n);
    m->radii.resize(n);
    m->counts.resize(n);
    m->offsets.resize(n);
}

ColorRenderer::~ColorRenderer() = default;

std::vector<float> ColorRenderer::render(const PinholeCamera& camera)
{
    const std::size_t pixels = static_cast<std::size_t>(camera.width) * camera.height;
    m->image.resize(3 * pixels);
    m->image.zero(3 * pixels);
    if (m->n)
    {
        RCam rc{};
        rc.width = camera.width;
        rc.height = camera.height;
        rc.tan_x = static_cast<float>(camera.width / (2 * camera.fx));
        rc.tan_y = static_cast<float>(camera.height / (2 * camera.fy));
        rc.focal_x = camera.width / (2.0f * rc.tan_x);
        rc.focal_y = camera.height / (2.0f * rc.tan_y);
        rc.px = static_cast<float>(2 * camera.fx / camera.width);
        rc.py = static_cast<float>(2 * camera.fy / camera.height);
        rc.ox = static_cast<float>(2 * camera.cx / camera.width - 1);
        rc.oy = static_cast<float>(2 * camera.cy / camera.height - 1);
        rc.kernel = 0.0f;
        rc.near_z = static_cast<float>(0.2 * m->unit);
        rc.view = device_camera(camera);
        project_kernel<<<blocks_for(m->n), 256>>>(m->n, m->xyz.p, m->cov.p, m->opacity.p, rc, m->centers.p, m->conics.p,
                                                  m->depths.p, m->radii.p);
        WP_LAUNCH("project_kernel");
        std::vector<std::array<int, 4>> pending = {{0, 0, camera.width, camera.height}};
        while (!pending.empty())
        {
            const auto [x0, y0, x1, y1] = pending.back();
            pending.pop_back();
            if (m->draw(camera.width, camera.height, x0, y0, x1, y1, std::max(x1 - x0, y1 - y0) > 16))
            {
                continue;
            }
            if (x1 - x0 >= y1 - y0)
            {
                const int mid = std::max(x0 + 16, ((x0 + x1) / 2 / 16) * 16);
                pending.push_back({mid, y0, x1, y1});
                pending.push_back({x0, y0, mid, y1});
            }
            else
            {
                const int mid = std::max(y0 + 16, ((y0 + y1) / 2 / 16) * 16);
                pending.push_back({x0, mid, x1, y1});
                pending.push_back({x0, y0, x1, mid});
            }
        }
    }
    std::vector<float> out(3 * pixels);
    m->image.download(out.data(), out.size());
    return out;
}

// ====================================================================== GridGpu

struct GridGpu::Impl
{
    int m = 0;
    int k = 0;
    int factor = 4;
    Buffer<double> cells, occluders;
    Buffer<std::uint32_t> support, against;
    Buffer<std::uint8_t> image, bits;
    Buffer<unsigned int> zbuffer;
    Buffer<float> pooled;
    Buffer<unsigned long long> occluded;

    const float* build_zbuffer(const Cam& cam, int& zw)
    {
        if (!k)
        {
            zw = 0;
            return nullptr;
        }
        zw = (cam.width + factor - 1) / factor;
        const int zh = (cam.height + factor - 1) / factor;
        const std::size_t size = static_cast<std::size_t>(zw) * zh;
        zbuffer.resize(size);
        pooled.resize(size);
        zclear_kernel<<<blocks_for(size), 256>>>(static_cast<int>(size), zbuffer.p);
        WP_LAUNCH("zclear_kernel");
        zscatter_kernel<<<blocks_for(k), 256>>>(k, occluders.p, cam, factor, zw, zbuffer.p);
        WP_LAUNCH("zscatter_kernel");
        zpool_kernel<<<blocks_for(size), 256>>>(zw, zh, zbuffer.p, pooled.p);
        WP_LAUNCH("zpool_kernel");
        return pooled.p;
    }

    void upload_bits(const Occupancy* occupancy)
    {
        if (occupancy && !occupancy->packed.empty())
        {
            bits.upload(occupancy->packed.data(), occupancy->packed.size());
        }
    }
};

GridGpu::GridGpu(const std::vector<double>& cells, const std::vector<double>& occluders, int zbuffer_factor)
    : m(std::make_unique<Impl>())
{
    m->m = static_cast<int>(cells.size() / 3);
    m->k = static_cast<int>(occluders.size() / 3);
    m->factor = std::max(1, zbuffer_factor);
    m->cells.upload(cells.data(), cells.size());
    if (m->k)
    {
        m->occluders.upload(occluders.data(), occluders.size());
    }
    m->support.resize(m->m);
    m->support.zero(m->m);
    m->against.resize(m->m);
    m->against.zero(m->m);
    m->occluded.resize(1);
    m->occluded.zero(1);
}

GridGpu::~GridGpu() = default;

std::size_t GridGpu::count() const
{
    return static_cast<std::size_t>(m->m);
}

void GridGpu::possible(const PinholeCamera& camera, const Occupancy* occupancy)
{
    if (!m->m)
    {
        return;
    }
    m->upload_bits(occupancy);
    const bool use = occupancy && !occupancy->packed.empty();
    possible_kernel<<<blocks_for(m->m), 256>>>(m->m, m->cells.p, device_camera(camera), use ? m->bits.p : nullptr,
                                               use ? occupancy->cols : 0, m->support.p);
    WP_LAUNCH("possible_kernel");
}

void GridGpu::exact(const PinholeCamera& camera, const std::uint8_t* image)
{
    if (!m->m)
    {
        return;
    }
    m->image.upload(image, static_cast<std::size_t>(camera.width) * camera.height);
    exact_kernel<<<blocks_for(m->m), 256>>>(m->m, m->cells.p, device_camera(camera), m->image.p, m->support.p);
    WP_LAUNCH("exact_kernel");
}

void GridGpu::signature(const PinholeCamera& camera, const std::uint8_t* image, double tolerance)
{
    if (!m->m)
    {
        return;
    }
    const Cam cam = device_camera(camera);
    m->image.upload(image, static_cast<std::size_t>(camera.width) * camera.height);
    int zw = 0;
    const float* pooled = m->build_zbuffer(cam, zw);
    signature_kernel<<<blocks_for(m->m), 256>>>(m->m, m->cells.p, cam, m->image.p, pooled, m->factor, zw,
                                                static_cast<float>(tolerance), m->support.p, m->against.p, m->occluded.p);
    WP_LAUNCH("signature_kernel");
}

void GridGpu::other(const PinholeCamera& camera, bool dry, const Occupancy* occupancy, double tolerance)
{
    if (!m->m)
    {
        return;
    }
    const Cam cam = device_camera(camera);
    if (!dry)
    {
        m->upload_bits(occupancy);
    }
    int zw = 0;
    const float* pooled = m->build_zbuffer(cam, zw);
    other_kernel<<<blocks_for(m->m), 256>>>(m->m, m->cells.p, cam, dry ? 1 : 0, dry ? nullptr : m->bits.p,
                                            dry ? 0 : occupancy->cols, pooled, m->factor, zw,
                                            static_cast<float>(tolerance), m->against.p, m->occluded.p);
    WP_LAUNCH("other_kernel");
}

void GridGpu::download(std::vector<std::uint32_t>& support, std::vector<std::uint32_t>& against, std::size_t& occluded) const
{
    support.resize(static_cast<std::size_t>(m->m));
    against.resize(static_cast<std::size_t>(m->m));
    m->support.download(support.data(), support.size());
    m->against.download(against.data(), against.size());
    unsigned long long value = 0;
    m->occluded.download(&value, 1);
    occluded = static_cast<std::size_t>(value);
}

} // namespace ai3d::waterpost
