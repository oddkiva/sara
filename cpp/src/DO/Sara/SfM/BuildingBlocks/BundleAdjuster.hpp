// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
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
                const int num_extrinsic_params) -> void;
  };


  struct BundleAdjuster
  {
    using PoseVertex = CameraPoseGraph::Vertex;

    BundleAdjustmentData ba_data;
    std::unique_ptr<ceres::Problem> ba_problem;

    auto allocate_data(const CameraPoseGraph& pose_graph,
                       const std::vector<FeatureTracker::Track>& tracks,
                       const int intrinsics_dim_per_camera,
                       const int extrinsics_dim_per_camera) -> void;

    auto populate_image_points(const FeatureTracker& feature_tracker,
                               const PointCloudGenerator& point_cloud_generator,
                               const std::vector<FeatureTracker::Track>& tracks)
        -> void;

    auto populate_scene_points(const PointCloudGenerator& point_cloud_generator,
                               const std::vector<FeatureTracker::Track>& tracks)
        -> void;

    //! @brief N.B.: awkward implementation: we ignore the shear value when
    //! populating the camera parameters.
    auto populate_camera_params(
        const CameraPoseGraph& pose_graph,
        const std::vector<Eigen::Matrix3d>& calibration_matrices,
        const PointCloudGenerator& point_cloud_generator,
        const std::vector<FeatureTracker::Track>& tracks) -> void;

    auto form_problem(const CameraPoseGraph& pose_graph,
                      const FeatureTracker& feature_tracker,
                      const std::vector<Eigen::Matrix3d>& calibration_matrices,
                      const PointCloudGenerator& point_cloud_generator,
                      const std::vector<FeatureTracker::Track>& tracks,
                      const int intrinsics_dim_per_camera,
                      const int extrinsics_dim_per_camera) -> void;

    auto solve() -> void;
  };

  //! @}

} /* namespace DO::Sara */
