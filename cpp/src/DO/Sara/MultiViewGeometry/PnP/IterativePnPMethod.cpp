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

#include <DO/Sara/MultiViewGeometry/Calibration/OmnidirectionalCameraReprojectionError.hpp>
#include <DO/Sara/MultiViewGeometry/PnP/IterativePnPMethod.hpp>

#include <ceres/ceres.h>


namespace DO::Sara {

  struct IterativePnPMethod::Impl
  {
    // std::unique_ptr<ceres::Problem> _problem;
    std::vector<double> _image_points;
    std::vector<double> _scene_points;
    std::vector<double> _poses;
    std::vector<std::size_t> _image_point_positions;
    std::vector<std::size_t> _scene_point_positions;
    std::vector<std::size_t> _pose_positions;

    auto add_data(const std::vector<Eigen::Vector2d>& /* image_points */,
                  const std::vector<Eigen::Vector3d>& /* scene_points */) -> void
    {
      // for (auto i=0u; i < image_points.size(); ++i)
      // {
      //   const auto& image_point = image_points[i];
      //   const auto& scene_point = scene_points[i];
      //   auto residual = OmnidirectionalCameraReprojectionError::create(
      //       image_point, scene_point);
      //   _problem->AddResidualBlock(residual, nullptr, );
      // }
    }

    auto solve() -> void
    {
      // auto options = ceres::Solver::Options{};
      // ceres::Solve(options, _problem.get(),
    }
  };

  auto IterativePnPMethod::ImplDeleter::operator()(
      const IterativePnPMethod::Impl* p) const -> void
  {
    delete p;
  }

}  // namespace DO::Sara
