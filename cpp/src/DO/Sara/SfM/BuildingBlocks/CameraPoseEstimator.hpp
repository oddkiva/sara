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

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/P3PSolver.hpp>
#include <DO/Sara/MultiViewGeometry/PointRayCorrespondenceList.hpp>
#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>


namespace DO::Sara {

  class CameraPoseEstimator
  {
  public:
    using FeatureTrack = PointCloudGenerator::FeatureTrack;
    using CameraIntrinsicModel = v2::PinholeCamera<double>;
    using PoseMatrix = Eigen::Matrix<double, 3, 4>;
    using Inlier = Tensor_<bool, 1>;
    using MinimalSample = Tensor_<int, 1>;

    //! @brief Constructor
    CameraPoseEstimator()
    {
      set_estimation_params();
    }

    //! @brief Set robust estimation parameters.
    auto
    set_estimation_params([[maybe_unused]] const PixelUnit error_max = 5._px,
                          const int ransac_iter_max = 1000,
                          const double ransac_confidence_min = 0.99) -> void
    {
      _inlier_predicate.ε = 0.1;  // error_max.value;
      _ransac_iter_max = ransac_iter_max;
      _ransac_confidence_min = ransac_confidence_min;
    }

    //! @brief Estimate the absolute pose from the data and the camera intrinsic
    //! parameters.
    auto estimate_pose(const PointRayCorrespondenceList<double>&,
                       const CameraIntrinsicModel&)
        -> std::tuple<PoseMatrix, Inlier, MinimalSample>;

    auto estimate_pose(const std::vector<FeatureTrack>&,
                       const CameraPoseGraph::Vertex,  //
                       const CameraIntrinsicModel&,    //
                       const PointCloudGenerator&)
        -> std::pair<PoseMatrix, bool>;

  private:
    P3PSolver<double> _solver;
    CheiralPnPConsistency<CameraIntrinsicModel> _inlier_predicate;
    int _ransac_inliers_min = 100;
    int _ransac_iter_max;
    double _ransac_confidence_min;
  };

}  // namespace DO::Sara
