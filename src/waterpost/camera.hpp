#pragma once

// COLMAP 相机与视图（pre_water_io.View）。cameras.bin / images.bin 由 ai3d-io 读取；投影公式与 Python 版
// camera_to_pixels 逐项相同（只支持 Python 版支持的模型，其余模型在投影时报 Unsupported camera model）。

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai3d::waterpost
{

struct View
{
    std::uint32_t image_id = 0;
    std::string name;
    std::uint32_t camera_id = 0;
    int model_id = 0;
    std::string model;
    int width = 0;
    int height = 0;
    std::vector<double> params;
    // 世界到相机：cam = R * p + t（行主序）。
    std::array<double, 9> r_wc{};
    std::array<double, 3> t_wc{};
    std::array<double, 3> center{};
    std::filesystem::path mask_path;
    // canonical_mask_path(mask_path)。
    std::string mask_key;
    double water_ratio = 0.0;
    // 原始相机带畸变时（位姿文件给的 SIMPLE_RADIAL / RADIAL / OPENCV / FULL_OPENCV）：上面的 model / params 是
    // 去畸变后的针孔相机（内参不变），所有投影都用它；raw_* 是原始相机，只在读掩码 / 照片时用来去畸变。
    int raw_model_id = -1;
    std::vector<double> raw_params;

    // name 里的反斜杠换成正斜杠。
    std::string key() const;
    // 模型无畸变（针孔），可以用投影后的凸包做批量裁剪。
    bool pinhole() const;
    // 掩码 / 照片是在带畸变的原始相机下拍的，读入后要先去畸变。
    bool distorted() const
    {
        return raw_model_id >= 0;
    }
};

// 原始相机的畸变：归一化坐标 (x, y) -> 畸变后的归一化坐标；返回 false 表示模型不支持。
// 公式与 COLMAP 相同（FULL_OPENCV 含 k3..k6 有理项）。
inline bool distort_normalized(int model_id, const std::vector<double>& p, double x, double y, double& xd, double& yd)
{
    const double r2 = x * x + y * y;
    switch (model_id)
    {
    case 2: // SIMPLE_RADIAL: f, cx, cy, k
        xd = x * (1 + p[3] * r2);
        yd = y * (1 + p[3] * r2);
        return true;
    case 3: // RADIAL: f, cx, cy, k1, k2
    {
        const double radial = 1 + p[3] * r2 + p[4] * r2 * r2;
        xd = x * radial;
        yd = y * radial;
        return true;
    }
    case 4: // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    case 6: // FULL_OPENCV: ... k3, k4, k5, k6
    {
        double radial = 1 + p[4] * r2 + p[5] * r2 * r2;
        if (model_id == 6)
        {
            radial = (radial + p[8] * r2 * r2 * r2) / (1 + p[9] * r2 + p[10] * r2 * r2 + p[11] * r2 * r2 * r2);
        }
        xd = x * radial + 2 * p[6] * x * y + p[7] * (r2 + 2 * x * x);
        yd = y * radial + p[6] * (r2 + 2 * y * y) + 2 * p[7] * x * y;
        return true;
    }
    default:
        return false;
    }
}

// 原始相机的焦距与主点（SIMPLE_RADIAL / RADIAL 的 f 同时作 fx、fy）。
inline void raw_intrinsics(int model_id, const std::vector<double>& p, double& fx, double& fy, double& cx, double& cy)
{
    if (model_id == 2 || model_id == 3)
    {
        fx = fy = p[0];
        cx = p[1];
        cy = p[2];
    }
    else
    {
        fx = p[0];
        fy = p[1];
        cx = p[2];
        cy = p[3];
    }
}

// 投影一个点：深度 > 1e-6 时返回 true 并给出像素坐标 (u, v)。
inline bool project_point(const View& view, const double* point, double& u, double& v);

// 读 sparse 目录下的 cameras.bin / images.bin（images 保持文件顺序）。
struct ColmapCatalog
{
    struct Camera
    {
        std::uint32_t id = 0;
        int model_id = 0;
        std::string model;
        std::uint64_t width = 0;
        std::uint64_t height = 0;
        std::vector<double> params;
    };
    struct Image
    {
        std::uint32_t id = 0;
        std::array<double, 4> qvec{};
        std::array<double, 3> tvec{};
        std::uint32_t camera_id = 0;
        std::string name;
    };
    std::vector<Camera> cameras;
    std::vector<Image> images;

    const Camera& camera(std::uint32_t id) const;
};

ColmapCatalog read_colmap_catalog(const std::filesystem::path& sparse_dir);

std::array<double, 9> qvec_to_rotmat(const std::array<double, 4>& q);

// semantic_mask_path：mask_dir / <image name 换后缀 .png>。
std::filesystem::path semantic_mask_path(const std::filesystem::path& mask_dir, const std::string& image_name);

// 由目录项构造视图（不读掩码）。
View make_view(const ColmapCatalog::Image& image, const ColmapCatalog::Camera& camera,
               const std::filesystem::path& mask_path);

[[noreturn]] void unsupported_camera_model(const std::string& model);

// ---------------------------------------------------------------- inline

inline bool project_point(const View& view, const double* point, double& u, double& v)
{
    const std::array<double, 9>& r = view.r_wc;
    const double cx_ = r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + view.t_wc[0];
    const double cy_ = r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + view.t_wc[1];
    const double depth = r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + view.t_wc[2];
    if (!(depth > 1e-6))
    {
        return false;
    }
    const double x = cx_ / depth;
    const double y = cy_ / depth;
    const double* p = view.params.data();
    switch (view.model_id)
    {
    case 0: // SIMPLE_PINHOLE
        u = p[0] * x + p[1];
        v = p[0] * y + p[2];
        return true;
    case 1: // PINHOLE
        u = p[0] * x + p[2];
        v = p[1] * y + p[3];
        return true;
    case 2: // SIMPLE_RADIAL
    {
        const double r2 = x * x + y * y;
        const double radial = 1 + p[3] * r2;
        u = p[0] * x * radial + p[1];
        v = p[0] * y * radial + p[2];
        return true;
    }
    case 3: // RADIAL
    {
        const double r2 = x * x + y * y;
        const double radial = 1 + p[3] * r2 + p[4] * r2 * r2;
        u = p[0] * x * radial + p[1];
        v = p[0] * y * radial + p[2];
        return true;
    }
    case 4: // OPENCV
    case 6: // FULL_OPENCV
    {
        const double fx = p[0], fy = p[1], cx = p[2], cy = p[3], k1 = p[4], k2 = p[5], p1 = p[6], p2 = p[7];
        const double r2 = x * x + y * y;
        const double radial = 1 + k1 * r2 + k2 * r2 * r2;
        const double xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        const double yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        u = fx * xd + cx;
        v = fy * yd + cy;
        return true;
    }
    default: // 鱼眼模型（5 / 8 / 9 …）需要 θ 畸变公式，不用透视公式近似，直接报不支持。
        unsupported_camera_model(view.model);
    }
}

// 投影并检查是否落在影像内；落在内时给出整数像素。
inline bool project_inside(const View& view, const double* point, int& column, int& row)
{
    double u = 0.0;
    double v = 0.0;
    if (!project_point(view, point, u, v))
    {
        return false;
    }
    if (!(u >= 0.0 && v >= 0.0 && u < static_cast<double>(view.width) && v < static_cast<double>(view.height)))
    {
        return false;
    }
    column = static_cast<int>(std::floor(u));
    row = static_cast<int>(std::floor(v));
    return true;
}

} // namespace ai3d::waterpost
