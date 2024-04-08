// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/QuaternionBasedPose.hpp>

#include <DO/Sara/SfM/BuildingBlocks/v2/FeatureTracker.hpp>
#include <DO/Sara/SfM/BuildingBlocks/v2/RelativePoseEstimator.hpp>

#include <boost/graph/adjacency_list.hpp>


namespace DO::Sara {

  struct CameraPoseData
  {
    //! @brief The corresponding image frame index.
    int frame_index;

    //! @brief The keypoints detected in the image.
    KeypointList<OERegion, float> keypoints;

    //! @brief "Absolute" pose w.r.t. some reference frame.
    QuaternionBasedPose<double> pose;
  };

  struct RelativeMotionData
  {
    using camera_id_t = int;
    static constexpr auto undefined_camera_id = -1;

    camera_id_t src_camera = undefined_camera_id;
    camera_id_t dst_camera = undefined_camera_id;

    std::vector<Match> matches;
    Tensor_<bool, 1> inliers;

    Motion motion;
  };

  class CameraPoseGraph
  {
  public:
    using GraphImpl =
        boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              CameraPoseData, RelativeMotionData>;

  public:
    using Vertex = boost::graph_traits<GraphImpl>::vertex_descriptor;
    using Edge = boost::graph_traits<GraphImpl>::edge_descriptor;

    auto detect_keypoints(const v2::FeatureTracker& feature_tracker,
                          const ImageView<float>& image,  //
                          const int frame_index) -> void;

    auto estimate_relative_motion(
        const v2::FeatureTracker& feature_tracker,                 //
        const v2::RelativePoseEstimator& relative_pose_estimator,  //
        const Vertex src, const Vertex dst) -> void;

  private:
    //! @brief The graph data structure shortened as g.
    GraphImpl _g;
  };

} /* namespace DO::Sara */
