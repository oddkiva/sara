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
    const RelativePoseData& relative_pose_data,  //
    const Vertex u, const Vertex v) -> bool
{
  const auto [e, edge_added] = boost::add_edge(u, v, _g);
  _g[e] = relative_pose_data;
  return edge_added;
}
