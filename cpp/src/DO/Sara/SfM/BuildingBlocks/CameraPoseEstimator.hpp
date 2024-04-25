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
#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/P3PSolver.hpp>
#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>


namespace DO::Sara {

  class CameraPoseEstimator
  {
  public:
    using FeatureTrack = PointCloudGenerator::FeatureTrack;

    using CameraModel = v2::BrownConradyDistortionModel<double>;
    using PoseMatrix = Eigen::Matrix<double, 3, 4>;

    //! @brief Constructor
    CameraPoseEstimator()
    {
      set_estimation_params();
    }

    auto set_estimation_params(const PixelUnit error_max = 0.5_px,
                               const int ransac_iter_max = 1000u,
                               const double ransac_confidence_min = 0.99)
        -> void
    {
      _inlier_predicate.ε = error_max.value;
      _ransac_iter_max = ransac_iter_max;
      _ransac_confidence_min = ransac_confidence_min;
    }

    auto estimate_pose(const std::vector<FeatureTrack>&,
                       const CameraPoseGraph::Vertex,  //
                       const CameraModel&,             //
                       const PointCloudGenerator&)
        -> std::pair<PoseMatrix, bool>;

  private:
    P3PSolver<double> _solver;
    CheiralPnPConsistency<CameraModel> _inlier_predicate;
    int _ransac_inliers_min = 100;
    int _ransac_iter_max;
    double _ransac_confidence_min;
  };

}  // namespace DO::Sara
