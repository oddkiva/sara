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
    using PoseMatrix = Eigen::Matrix<double, 3, 4>;
    using Model = PoseMatrix;

    //! @brief The camera model for the image.
    const CameraModel* camera = nullptr;
    //! @brief The pose matrix.
    PoseMatrix T;
    //! @brief Image reprojection error in pixel.
    double ε;

    inline CheiralPnPConsistency() = default;

    inline CheiralPnPConsistency(const PoseMatrix& pose_matrix)
    {
      set_model(pose_matrix);
    }

    inline auto set_model(const PoseMatrix& pose_matrix) -> void
    {
      T = pose_matrix;
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& scene_points,
                    const Eigen::MatrixBase<Derived>& rays) const
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      if (camera == nullptr)
        throw std::runtime_error{
            "Error: you must initialize the intrinsic camera parameters!"};

      const auto& X_world = scene_points;
      const Eigen::MatrixXd X_camera = T * X_world;

      auto u1 = Eigen::MatrixXd{2, scene_points.cols()};
      for (auto i = 0; i < u1.cols(); ++i)
        u1.col(i) = camera->project(X_camera.col(i));

      auto u2 = Eigen::MatrixXd{rays.rows(), rays.cols()};
      for (auto i = 0; i < u2.cols(); ++i)
        u2.col(i) = camera->project(rays.col(i));

      // Check the cheirality w.r.t. the candidate pose.
      const auto cheiral = X_camera.row(2).array() > 0;

      // Check the **squared** image reprojection errors.
      const auto ε_max = square(ε);
      const auto ε_small = (u2 - u1).colwise().squaredNorm().array() < ε_max;

      return ε_small && cheiral;
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointRayCorrespondenceSubsetList<T>& m) const
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      const auto& scene_points = m.x.colmajor_view().matrix();
      const auto& backprojected_rays = m.y.colmajor_view().matrix();
      return this->operator()(scene_points, backprojected_rays);
    }
  };

}  // namespace DO::Sara
