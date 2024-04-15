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

#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>
#include <DO/Sara/SfM/Graph/PointCloudManipulator.hpp>


namespace DO::Sara {

  class BundleAdjuster
  {
  public:
    BundleAdjuster(CameraPoseGraph& camera_pose_graph,
                   const FeatureGraph& feature_graph,
                   PointCloudManipulator::PointCloud& point_cloud)
      : _camera_pose_graph{camera_pose_graph}
      , _feature_graph{feature_graph}
      , _point_cloud{point_cloud}
    {
    }

    auto adjust() -> void;

  private:
    CameraPoseGraph& _camera_pose_graph;
    const FeatureGraph& _feature_graph;
    PointCloudManipulator::PointCloud& _point_cloud;
  };

}  // namespace DO::Sara
