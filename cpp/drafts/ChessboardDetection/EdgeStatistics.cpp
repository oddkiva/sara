// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "EdgeStatistics.hpp"


namespace DO::Sara {

  auto get_curve_shape_statistics(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
      -> CurveStatistics
  {
    auto curves_64f = std::vector<std::vector<Eigen::Vector2d>>{};
    curves_64f.resize(curve_pts.size());
    std::transform(curve_pts.begin(), curve_pts.end(), curves_64f.begin(),
                   [](const auto& points) {
                     auto points_2d = std::vector<Eigen::Vector2d>{};
                     points_2d.resize(points.size());
                     std::transform(points.begin(), points.end(),
                                    points_2d.begin(), [](const auto& p) {
                                      return p.template cast<double>();
                                    });
                     return points_2d;
                   });

    return CurveStatistics{curves_64f};
  }

  auto
  gradient_mean(const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
                const ImageView<float>& Ix, const ImageView<float>& Iy)      //
      -> std::vector<Eigen::Vector2f>
  {
    auto gradient_means = std::vector<Eigen::Vector2f>(curve_pts.size());

    std::transform(
        curve_pts.begin(), curve_pts.end(), gradient_means.begin(),
        [&Ix, &Iy](const std::vector<Eigen::Vector2i>& points) {
          static const Eigen::Vector2f zero2f = Eigen::Vector2f::Zero();
          Eigen::Vector2f mean = std::accumulate(
              points.begin(), points.end(), zero2f,
              [&Ix, &Iy](const Eigen::Vector2f& gradient,
                         const Eigen::Vector2i& point) -> Eigen::Vector2f {
                const Eigen::Vector2f g =
                    gradient + Eigen::Vector2f{Ix(point), Iy(point)};
                return g;
              });
          mean /= points.size();
          return mean;
        });

    return gradient_means;
  }

  auto gradient_covariance(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
      const ImageView<float>& Ix, const ImageView<float>& Iy)      //
      -> std::vector<Eigen::Matrix2f>
  {
    auto gradient_covariances = std::vector<Eigen::Matrix2f>(curve_pts.size());

    std::transform(
        curve_pts.begin(), curve_pts.end(), gradient_covariances.begin(),
        [&Ix, &Iy](const std::vector<Eigen::Vector2i>& points) {
          static const Eigen::Matrix2f zero2f = Eigen::Matrix2f::Zero();
          Eigen::Matrix2f cov = std::accumulate(
              points.begin(), points.end(), zero2f,
              [&Ix, &Iy](const Eigen::Matrix2f& cov,
                         const Eigen::Vector2i& point) -> Eigen::Matrix2f {
                const Eigen::Vector2f g = Eigen::Vector2f{Ix(point), Iy(point)};
                const Eigen::Matrix2f new_cov = cov + g * g.transpose();
                return new_cov;
              });
          cov /= points.size();
          return cov;
        });

    return gradient_covariances;
  }

}  // namespace DO::Sara
