#include "camera.hpp"

#include "error.hpp"
#include "fs_util.hpp"

#include <ai3d/io/colmap.hpp>

#include <unordered_map>

namespace ai3d::waterpost
{
namespace
{

const char* model_name(int id)
{
    switch (id)
    {
    case 0:
        return "SIMPLE_PINHOLE";
    case 1:
        return "PINHOLE";
    case 2:
        return "SIMPLE_RADIAL";
    case 3:
        return "RADIAL";
    case 4:
        return "OPENCV";
    case 5:
        return "OPENCV_FISHEYE";
    case 6:
        return "FULL_OPENCV";
    case 7:
        return "FOV";
    case 8:
        return "SIMPLE_RADIAL_FISHEYE";
    case 9:
        return "RADIAL_FISHEYE";
    case 10:
        return "THIN_PRISM_FISHEYE";
    default:
        return nullptr;
    }
}

} // namespace

std::string View::key() const
{
    std::string result = name;
    for (char& character : result)
    {
        if (character == '\\')
        {
            character = '/';
        }
    }
    return result;
}

bool View::pinhole() const
{
    return model_id == 0 || model_id == 1;
}

void unsupported_camera_model(const std::string& model)
{
    fail_runtime("NotImplementedError: Unsupported camera model: " + model);
}

const ColmapCatalog::Camera& ColmapCatalog::camera(std::uint32_t id) const
{
    for (const Camera& item : cameras)
    {
        if (item.id == id)
        {
            return item;
        }
    }
    fail_runtime("KeyError: COLMAP image references missing camera " + std::to_string(id));
}

ColmapCatalog read_colmap_catalog(const std::filesystem::path& sparse_dir)
{
    ColmapCatalog catalog;
    std::unordered_map<std::uint32_t, std::size_t> camera_index;
    const std::filesystem::path cameras_file = sparse_dir / "cameras.bin";
    const std::filesystem::path images_file = sparse_dir / "images.bin";
    if (!std::filesystem::is_regular_file(cameras_file))
    {
        fail_missing("No such file: cameras.bin: " + path_utf8(cameras_file));
    }
    if (!std::filesystem::is_regular_file(images_file))
    {
        fail_missing("No such file: images.bin: " + path_utf8(images_file));
    }
    const auto cameras = io::visit_cameras_binary(
        cameras_file, [&](const io::Camera& camera, std::uint64_t, std::uint64_t) -> core::Result<void> {
            const int id = static_cast<int>(camera.model);
            const char* name = model_name(id);
            if (name == nullptr)
            {
                return core::fail(core::ErrorKind::format_invalid,
                                  "Unsupported COLMAP camera model id: " + std::to_string(id));
            }
            ColmapCatalog::Camera item;
            item.id = camera.camera_id;
            item.model_id = id;
            item.model = name;
            item.width = camera.width;
            item.height = camera.height;
            item.params = camera.params;
            // Python 用 dict 保存，同 id 后者覆盖前者。
            const auto found = camera_index.find(item.id);
            if (found != camera_index.end())
            {
                catalog.cameras[found->second] = std::move(item);
            }
            else
            {
                camera_index[item.id] = catalog.cameras.size();
                catalog.cameras.push_back(std::move(item));
            }
            return {};
        });
    if (!cameras)
    {
        fail_runtime("colmap cameras.bin: " + core::describe(cameras.error()));
    }
    std::unordered_map<std::uint32_t, std::size_t> image_index;
    const auto images = io::visit_images_binary(
        images_file, [&](const io::View& view, std::uint64_t, std::uint64_t) -> core::Result<void> {
            ColmapCatalog::Image item;
            item.id = view.image_id;
            for (int k = 0; k < 4; ++k)
            {
                item.qvec[static_cast<std::size_t>(k)] = view.qvec[k];
            }
            for (int k = 0; k < 3; ++k)
            {
                item.tvec[static_cast<std::size_t>(k)] = view.tvec[k];
            }
            item.camera_id = view.camera_id;
            item.name = view.name;
            const auto found = image_index.find(item.id);
            if (found != image_index.end())
            {
                catalog.images[found->second] = std::move(item);
            }
            else
            {
                image_index[item.id] = catalog.images.size();
                catalog.images.push_back(std::move(item));
            }
            return {};
        });
    if (!images)
    {
        fail_runtime("colmap images.bin: " + core::describe(images.error()));
    }
    return catalog;
}

std::array<double, 9> qvec_to_rotmat(const std::array<double, 4>& q)
{
    return {
        1 - 2 * (q[2] * q[2]) - 2 * (q[3] * q[3]),
        2 * q[1] * q[2] - 2 * q[0] * q[3],
        2 * q[3] * q[1] + 2 * q[0] * q[2],
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        1 - 2 * (q[1] * q[1]) - 2 * (q[3] * q[3]),
        2 * q[2] * q[3] - 2 * q[0] * q[1],
        2 * q[3] * q[1] - 2 * q[0] * q[2],
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        1 - 2 * (q[1] * q[1]) - 2 * (q[2] * q[2]),
    };
}

std::filesystem::path semantic_mask_path(const std::filesystem::path& mask_dir, const std::string& image_name)
{
    std::string normalized = image_name;
    for (char& character : normalized)
    {
        if (character == '\\')
        {
            character = '/';
        }
    }
    const std::filesystem::path relative = utf8_path(normalized);
    if (relative.is_absolute() || relative.has_root_name() || relative.has_root_directory())
    {
        fail_value("COLMAP image name must be relative: " + image_name);
    }
    for (const std::filesystem::path& part : relative)
    {
        if (part == "..")
        {
            fail_value("COLMAP image name must be relative: " + image_name);
        }
    }
    std::filesystem::path result = relative;
    result.replace_extension(".png");
    return mask_dir / result;
}

View make_view(const ColmapCatalog::Image& image, const ColmapCatalog::Camera& camera,
               const std::filesystem::path& mask_path)
{
    View view;
    view.image_id = image.id;
    view.name = image.name;
    view.camera_id = image.camera_id;
    view.model_id = camera.model_id;
    view.model = camera.model;
    view.width = static_cast<int>(camera.width);
    view.height = static_cast<int>(camera.height);
    view.params = camera.params;
    view.r_wc = qvec_to_rotmat(image.qvec);
    view.t_wc = image.tvec;
    for (std::size_t i = 0; i < 3; ++i)
    {
        view.center[i] = -(view.r_wc[0 * 3 + i] * view.t_wc[0] + view.r_wc[1 * 3 + i] * view.t_wc[1] +
                           view.r_wc[2 * 3 + i] * view.t_wc[2]);
    }
    view.mask_path = mask_path;
    view.mask_key = canonical_path_text(mask_path);
    return view;
}

} // namespace ai3d::waterpost
