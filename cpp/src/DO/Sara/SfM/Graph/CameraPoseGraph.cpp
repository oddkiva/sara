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
#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>


using namespace DO::Sara;


auto CameraPoseGraph::add_absolute_pose(
    KeypointList<OERegion, float>&& keypoints,  //
    const int image_id) -> CameraPoseGraph::Vertex
{
  auto& logger = Logger::get();

  SARA_LOGI(logger, "Detecting keypoints for image frame {}", image_id);

  // Grow the pose graph by creating a new camera vertex.
  const auto v = boost::add_vertex(_g);

  // Store the camera pose data.
  auto& pose_data = _g[v];
  pose_data.image_id = image_id;
  pose_data.keypoints = std::move(keypoints);

  const auto& f = features(pose_data.keypoints);
  SARA_LOGI(logger, "Camera {}: {} keypoints", v, f.size());

  return v;
}

auto CameraPoseGraph::add_relative_pose(
    const v2::RelativePoseEstimator& relative_pose_estimator,  //
    const FeatureParams& feature_params,                       //
    const Vertex u, const Vertex v) -> std::pair<Edge, bool>
{
  auto& logger = Logger::get();

  SARA_LOGI(logger, "Match features...");
  const auto& src_keys = _g[u].keypoints;
  const auto& dst_keys = _g[v].keypoints;
  if (features(src_keys).empty() || features(dst_keys).empty())
    return {{}, false};

  auto matches = match(src_keys, dst_keys, feature_params.sift_nn_ratio);
  if (matches.empty())
    return {{}, false};
  if (matches.size() > feature_params.num_matches_max)
    matches.resize(feature_params.num_matches_max);

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
  }

  return {e, edge_added};
}
