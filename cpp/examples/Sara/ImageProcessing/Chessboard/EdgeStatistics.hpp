#pragma once

#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>


namespace DO::Sara {

  auto get_curve_shape_statistics(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
      -> CurveStatistics;

  auto
  gradient_mean(const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
                const ImageView<float>& Ix,                                  //
                const ImageView<float>& Iy)                                  //
      -> std::vector<Eigen::Vector2f>;

  auto gradient_covariance(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
      const ImageView<float>& Ix,                                  //
      const ImageView<float>& Iy)                                  //
      -> std::vector<Eigen::Matrix2f>;

}  // namespace DO::Sara
