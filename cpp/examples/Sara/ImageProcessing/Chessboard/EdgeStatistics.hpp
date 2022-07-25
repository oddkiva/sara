#pragma once

#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>


auto get_curve_shape_statistics(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
    -> DO::Sara::CurveStatistics;

auto mean_gradient(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
    const DO::Sara::ImageView<float>& Ix,                        //
    const DO::Sara::ImageView<float>& Iy)                        //
    -> std::vector<Eigen::Vector2f>;
