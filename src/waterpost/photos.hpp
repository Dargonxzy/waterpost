#pragma once

// 训练照片：按视图读入、对齐到针孔渲染（缩放到视图尺寸，畸变相机先去畸变）。evaluate 与漂浮簇照片核验共用。

#include "camera.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>

namespace ai3d::waterpost
{

// 图像名是存在的绝对路径就直接用；否则在照片目录下找 <图像名去掉前 k 级目录>（k 第一次找到后固定在 cut 里，初值 -1）。
std::filesystem::path photo_path(const std::filesystem::path& images, const std::string& name, long long& cut);
// 读入 BGR 照片；找不到时返回空 Mat。
cv::Mat load_view_photo(const std::filesystem::path& images, const View& view, long long& cut);

} // namespace ai3d::waterpost
