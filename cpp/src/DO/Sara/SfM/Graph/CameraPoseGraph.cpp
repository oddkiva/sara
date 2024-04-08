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


#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/Logging/Logger.hpp>


using namespace DO::Sara;

auto CameraPoseGraph::detect_keypoints(
    const v2::FeatureTracker& feature_tracker,
    const ImageView<float>& image,  //
    const int frame_index) -> void

{
  auto& logger = Logger::get();

  SARA_LOGI(logger, "Detecting keypoints for image frame {}", frame_index);

  // Grow the pose graph by creating a new camera vertex.
  const auto v = boost::add_vertex(_g);

  // Store the camera pose data.
  auto& camera_pose_data = _g[v];
  camera_pose_data.frame_index = frame_index;
  camera_pose_data.keypoints = compute_sift_keypoints(image);

  const auto& f = features(camera_pose_data.keypoints);
  SARA_LOGI(logger, "Camera vertex: {} keypoints", f.size());
}

auto CameraPoseGraph::estimate_relative_motion(
    const v2::FeatureTracker& feature_tracker,                 //
    const v2::RelativePoseEstimator& relative_pose_estimator,  //
    const Vertex u, const Vertex v) -> void
{
  auto& logger = Logger::get();

  SARA_LOGI(logger, "Match features...");
  const auto& src_keys = _g[u].keypoints;
  const auto& dst_keys = _g[v].keypoints;
  auto matches = feature_tracker.match_features(src_keys, dst_keys);
  if (matches.empty())
    return;

  SARA_LOGI(logger, "Estimating relative pose...");
  auto [geometry, inliers, sample_best] =
      relative_pose_estimator.estimate_relative_pose(src_keys, dst_keys,
                                                     matches);
  const auto num_inliers = inliers.flat_array().count();
  SARA_LOGI(logger, "inlier count: {}", num_inliers);

  const auto success = num_inliers > 100;
  auto e = Edge{};
  auto edge_added = false;
  if (success)
  {
    std::tie(e, edge_added) = boost::add_edge(u, v, _g);
    auto& relative_motion_data = _g[e];
    relative_motion_data.matches = std::move(matches);
    relative_motion_data.inliers = std::move(inliers);
    relative_motion_data.src_camera = u;
    relative_motion_data.dst_camera = v;
  }
}
