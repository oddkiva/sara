// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/MultiViewGeometry/PnP/LambdaTwist.hpp>
#include <DO/Sara/MultiViewGeometry/PointRayCorrespondenceList.hpp>

#include <fmt/format.h>


namespace DO::Sara {

  template <typename T>
  struct P3PSolver
  {
    static constexpr auto num_points = 3;
    static constexpr auto num_models = 4;

    using tensor_view_type = TensorView_<T, 2>;
    using data_point_type = std::array<TensorView_<T, 2>, 2>;
    using model_type = Eigen::Matrix<T, 3, 4>;

    inline auto
    operator()(const tensor_view_type& scene_points,
               const tensor_view_type& rays) const -> std::vector<model_type>
    {
      const auto sp_mat_ = scene_points.colmajor_view().matrix();

      Eigen::Matrix3<T> sp_mat = sp_mat_.topRows(3);
      if (sp_mat_.cols() == 4)
        sp_mat.array().rowwise() /= sp_mat_.array().row(3);

      const Eigen::Matrix3<T> ray_mat = rays.colmajor_view().matrix();
      return solve_p3p(sp_mat, ray_mat);
    }

    inline auto operator()(const data_point_type& X) -> std::vector<model_type>
    {
      const auto& [scene_points, backprojected_rays] = X;
      return this->operator()(scene_points, backprojected_rays);
    }
  };

  //! @brief Joint cheirality and epipolar consistency for RANSAC.
  template <typename CameraModel>
  struct CheiralPnPConsistency
  {
    using PoseMatrix = Eigen::Matrix<double, 3, 4>;
    using Model = PoseMatrix;

    //! @brief The camera model for the image.
    const CameraModel* C = nullptr;
    //! @brief The pose matrix.
    PoseMatrix T;
    //! @brief Image reprojection error in pixels.
    double ε;

    inline CheiralPnPConsistency() = default;

    inline CheiralPnPConsistency(const PoseMatrix& pose_matrix)
    {
      set_model(pose_matrix);
    }

    inline auto set_camera(const CameraModel& camera_) -> void
    {
      C = &camera_;
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
      if (C == nullptr)
        throw std::runtime_error{
            "Error: you must initialize the intrinsic camera parameters!"};

      if (scene_points.cols() != rays.cols())
        throw std::runtime_error{
            "Error: the number of scene points and rays must be equal!"};

      // Calculate the scene point coordinates in the camera frame.
      const auto& X_world = scene_points;
      auto X_camera = Eigen::MatrixXd{};

      if (X_world.rows() == 3)  // Euclidean coordinates.
        X_camera = T * X_world.colwise().homogeneous();
      else if (X_world.rows() == 4)  // Homogeneous coordinates.
        X_camera = T * X_world;
      else
        throw std::runtime_error{
            "The dimension of scene points is incorrect. "
            "They must either 3D (Euclidean) or 4D (homogeneous)!"};

      fmt::print("Pose:\n{}\n", T);

      // Project the camera coordinates to the image plane.
      //
      // The result is a list of pixel coordinates.
      auto u1 = Eigen::MatrixXd{2, scene_points.cols()};
      for (auto i = 0; i < u1.cols(); ++i)
        u1.col(i) = C->project(X_camera.col(i));

      // Reproject the backprojected rays to the image plane.
      //
      // The result is again a list of pixel coordinates.
      //
      // This is an awkward and potentially wasteful operation.
      // TODO: see if we can do this in the normalization operation:
      //       via the class Normalizer<P3PSolver>
      auto u2 = Eigen::MatrixXd{2, rays.cols()};
      for (auto i = 0; i < u2.cols(); ++i)
        u2.col(i) = C->project(rays.col(i));

      // Check the cheirality w.r.t. the candidate pose.
      const auto cheiral = X_camera.row(2).array() > 0;

      // Check the **squared** image reprojection errors.
      const auto ε_max = square(ε);
      const auto ε_squared = (u2 - u1).colwise().squaredNorm().array();
      const auto ε_small = ε_squared < ε_max;

      const auto ε_debug = Eigen::VectorXd{ε_squared.sqrt()};
      const auto col_max = std::min(Eigen::Index{10}, u2.cols());
      for (auto i = 0; i < col_max; ++i)
      {
        fmt::print("u1[{}]: {}   u2[{}]: {}\n",      //
                   i, u1.col(i).transpose().eval(),  //
                   i, u2.col(i).transpose().eval());
      }
      fmt::print("ε =\n{}\n", ε_debug.head(col_max).eval());
      fmt::print("ε_small.count() = {}\n", ε_small.count());

      return ε_small && cheiral;
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointRayCorrespondenceList<T>& m) const
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      const auto& scene_points = m.x.colmajor_view().matrix();
      const auto& backprojected_rays = m.y.colmajor_view().matrix();
      return this->operator()(scene_points, backprojected_rays);
    }
  };


  //! @}

}  // namespace DO::Sara
