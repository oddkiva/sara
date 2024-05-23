// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>
#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>

#include <ceres/problem.h>

#include <vector>


namespace DO::Sara {

  /*!
   *  @ingroup MultiViewGeometry
   *  @defgroup MultiviewBA Bundle Adjustment
   *  @{
   */

  //! @brief Bundle adjustment class.
  struct BundleAdjustmentData
  {
    //! @brief An observation is a 2D image point.
    Tensor_<double, 2> observations;
    //! @brief the corresponding 3D point index for the observation 'o'
    std::vector<int> point_indices;
    //! @brief the corresponding 3D camera index for the observation 'o'
    std::vector<int> camera_indices;


    //! @brief The parameter is the set of camera parameters and the set of 3D
    //! point coordinates.
    std::vector<double> parameters;
    //! @{
    //! @brief The number of parameters in details.
    int num_cameras;
    int num_intrinsics;
    int num_extrinsics;
    int num_points;
    //! @}
    //! @brief Convenient parameter data views.
    TensorView_<double, 2> intrinsics;
    TensorView_<double, 2> extrinsics;
    TensorView_<double, 2> point_coords;

    auto resize(const int num_image_points, const int num_scene_points,
                const int num_views, const int num_intrinsic_params,
                const int num_extrinsic_params)
    {
      auto& logger = Logger::get();

      SARA_LOGI(logger, "Allocating memory for observation data...");
      observations = Tensor_<double, 2>{{num_image_points, 2}};
      point_indices = std::vector<int>(num_image_points);
      camera_indices = std::vector<int>(num_image_points);

      SARA_LOGI(logger, "Allocating memory for parameter data...");
      // Store the number of parameters.
      num_cameras = num_views;
      num_intrinsics = num_intrinsic_params;
      num_extrinsics = num_extrinsic_params;
      num_points = num_scene_points;
      // Allocate.
      const auto num_parameters =
          num_cameras * (num_intrinsics + num_extrinsics) + num_points * 3;
      parameters = std::vector<double>(num_parameters);

      // Update the memory views.
      auto intrinsics_new = TensorView_<double, 2>{
          parameters.data(),             //
          {num_cameras, num_intrinsics}  //
      };
      auto extrinsics_new = TensorView_<double, 2>{
          parameters.data() + num_cameras * num_intrinsics,  //
          {num_cameras, num_extrinsics}                      //
      };
      auto point_coords_new = TensorView_<double, 2>{
          parameters.data() + num_cameras * (num_extrinsics + num_intrinsics),
          {num_points, 3}};

      intrinsics.swap(intrinsics_new);
      extrinsics.swap(extrinsics_new);
      point_coords.swap(point_coords_new);
    }
  };

  struct BundleAdjustmentProblem
  {
    BundleAdjustmentData ba_data;
    ceres::Problem ba_problem;

    auto populate_image_points(const PointCloudGenerator& point_cloud_generator,
                               const std::vector<FeatureTracker::Track>& tracks)
        -> void
    {
      auto& logger = Logger::get();
      SARA_LOGI(logger, "Populating the BA observation/image point data...");

      auto o = 0;  // observation index.
      for (auto t = std::size_t{}; t < tracks.size(); ++t)
      {
        const auto& track = tracks[t];
        for (const auto& u : track)
        {
          const Eigen::Vector2d pixel_coords = point_cloud_generator  //
                                                   .pixel_coords(u)
                                                   .cast<double>();
          ba_data.observations(o, 0) = pixel_coords.x();
          ba_data.observations(o, 1) = pixel_coords.y();

          ba_data.point_indices[o] = static_cast<int>(t);
          ba_data.camera_indices[o] =
              static_cast<int>(feature_tracker._feature_graph[u].pose_vertex);

          ++o;
        }
      }
    }

    auto populate_scene_points(const PointCloudGenerator& point_cloud_generator,
                               const std::vector<FeatureTracker::Track>& tracks)
        -> void
    {
      auto& logger = Logger::get();
      SARA_LOGI(logger, "Populating the BA (3D) point data...");

      for (auto t = std::size_t{}; t < tracks.size(); ++t)
      {
        // Retrieve the scene point.
        const auto& track = tracks[t];
        const auto scene_point =
            point_cloud_generator.scene_point(track.front());

        // Store.
        const auto tt = static_cast<int>(t);
        ba_data.point_coords[tt].vector() = scene_point->coords();
      }
    }

    auto populate_camera_param(const PointCloudGenerator& point_cloud_generator,
                               const std::vector<FeatureTracker::Track>& tracks)
        -> void
    {
      using PoseVertex = CameraPoseGraph::Vertex;

      auto& logger = Logger::get();

      SARA_LOGI(logger, "Populating the BA camera parameter data...");
      auto extrinsics_params = ba_data.extrinsics.matrix();
      auto intrinsics_params = ba_data.intrinsics.matrix();
      for (auto v = PoseVertex{}; v < pose_graph.num_vertices(); ++v)
      {
        // Angle axis vector.
        auto extrinsics_v = extrinsics_params.row(v);
        auto intrinsics_v = intrinsics_params.row(v);

        // The original data.
        const auto& pose_v = pose_graph[v].pose;
        const auto aaxis_v = Eigen::AngleAxisd{pose_v.q};
        const Eigen::Vector3d aaxis_v_3d = aaxis_v.angle() * aaxis_v.axis();
        // Initialize the absolute rotation.
        extrinsics_v << aaxis_v_3d.transpose(), pose_v.t.transpose();
        SARA_LOGD(logger, "Populating extrinsics[{}]=\n{}", v,
                  extrinsics_v.eval());

        // Initialize the internal camera parameters.
        intrinsics_v(0) = K[v](0, 0);  // fx
        intrinsics_v(1) = K[v](1, 1);  // fy
        intrinsics_v(2) = K[v](0, 2);  // u0
        intrinsics_v(3) = K[v](1, 2);  // v0
        SARA_LOGD(logger, "Populating intrinsics[{}]=\n{}", v,
                  intrinsics_v.eval());
      }
    }

    auto form_problem(const PointCloudGenerator& point_cloud_generator,
                      const std::vector<FeatureTracker::Track>& tracks) -> void
    {
      // Allocate memory for the BA data.
      const auto num_scene_points = static_cast<int>(tracks_filtered.size());
      auto num_image_points = 0;
      for (const auto& track : tracks_filtered)
        num_image_points += static_cast<int>(track.size());

      auto ba_data = sara::BundleAdjustmentData{};
      static constexpr auto num_views = 2;
      static constexpr auto num_intrinsics = 4;  // fx, fy, u0, v0
      static constexpr auto num_extrinsics = 6;
      ba_data.resize(num_image_points, num_scene_points, num_views,  //
                     num_intrinsics, num_extrinsics);

      populate_image_points(point_cloud_generator, tracks);
      populate_scene_points(point_cloud_generator, tracks);
      populate_camera_params(point_cloud_generator, tracks);

      // Solve the bundle adjustment problem with Ceres.
      SARA_LOGI(logger, "Forming the BA problem...");
      auto ba_problem = ceres::Problem{};
      for (auto i = 0; i < num_image_points; ++i)
      {
        SARA_LOGT(logger, "Adding residual with image point {}...", i);

        // Create a cost residual function.
        const auto cost_fn = ReprojectionError::create(
            ba_data.observations(i, 0), ba_data.observations(i, 1));

        // Locate the parameter data.
        const auto camera_idx = ba_data.camera_indices[i];
        const auto point_idx = ba_data.point_indices[i];
        const auto extrinsics_ptr = ba_data.extrinsics[camera_idx].data();
        const auto intrinsics_ptr = ba_data.intrinsics[camera_idx].data();
        const auto scene_point_ptr = ba_data.point_coords[point_idx].data();

        ba_problem.AddResidualBlock(cost_fn, nullptr /* squared loss */,  //
                                    extrinsics_ptr, intrinsics_ptr,
                                    scene_point_ptr);
      }
    }

    auto solve() -> void
    {
      auto options = ceres::Solver::Options{};
      options.linear_solver_type = ceres::DENSE_SCHUR;
      options.minimizer_progress_to_stdout = true;

      auto summary = ceres::Solver::Summary{};
      ceres::Solve(options, &ba_problem, &summary);
      SARA_LOGI(logger, "{}", summary.BriefReport());
    }
  };

  //! @}

} /* namespace DO::Sara */
