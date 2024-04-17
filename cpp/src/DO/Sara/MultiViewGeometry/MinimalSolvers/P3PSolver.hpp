// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include "DO/Sara/Core/Math/UsualFunctions.hpp"
#include <DO/Sara/MultiViewGeometry/PnP/LambdaTwist.hpp>
#include <DO/Sara/MultiViewGeometry/PointRayCorrespondenceList.hpp>


namespace DO::Sara {

  template <typename T>
  struct P3PSolver
  {
    static constexpr auto num_points = 3;
    static constexpr auto num_models = 4;

    using data_point_type = TensorView_<T, 2>;
    using model_type = Eigen::Matrix<T, 3, 4>;

    inline auto operator()(const data_point_type& x) const
        -> std::vector<model_type>
    {
      const Eigen::Matrix3<T> scene_points = x.matrix().leftCols(3);
      const Eigen::Matrix3<T> backprojected_rays = x.matrix().rightCols(3);
      return solve_p3p(scene_points, backprojected_rays);
    }
  };

  //! @brief Joint cheirality and epipolar consistency for RANSAC.
  template <typename CameraModel>
  struct CheiralPnPConsistency
  {
    using Model = Eigen::Matrix<double, 3, 4>;

    const CameraModel* camera = nullptr;
    Model pose;
    double image_reproj_err_max;

    CheiralPnPConsistency() = default;

    CheiralPnPConsistency(const Model& pose)
    {
      set_model(pose);
    }

    auto set_model(const Model& p) -> void
    {
      pose = p;
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& scene_points,
                    const Eigen::MatrixBase<Derived>& rays) const
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      if (camera == nullptr)
        throw std::runtime_error{
            "Error: you must initialize the intrinsic camera parameters!"};

      Eigen::MatrixXd u1n = pose * scene_points;
      auto u1 = Eigen::MatrixXd{2, scene_points.cols()};
      for (auto i = 0; i < u1.cols(); ++i)
        u1.col(i) = camera->project(u1.col(i));

      Eigen::MatrixXd u2{rays.rows(), rays.cols()};
      for (auto i = 0; i < u2.cols(); ++i)
        u2.col(i) = camera->project(rays.col(i));

      // Check the cheirality w.r.t. the candidate pose.
      const auto cheiral = u1n.row(2).array() > 0;

      // Checka the image reprojection errors.
      const auto err_max = square(image_reproj_err_max);
      const auto small_reproj_error =
          (u2 - u1).colwise().squaredNorm().array() < err_max;

      return small_reproj_error && cheiral;
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointRayCorrespondenceSubsetList<T>& m) const
        -> Array<bool, 1, Dynamic>
    {
      const auto& scene_points = m._p1.colmajor_view().matrix();
      const auto& backprojected_rays = m._p2.colmajor_view().matrix();
      return this->operator()(scene_points, backprojected_rays);
    }
  };

}  // namespace DO::Sara
