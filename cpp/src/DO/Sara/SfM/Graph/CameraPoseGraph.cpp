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


auto CameraPoseGraph::add_absolute_pose(AbsolutePoseData&& data)
    -> CameraPoseGraph::Vertex
{
  auto& logger = Logger::get();

  // Grow the pose graph by creating a new camera vertex.
  const auto v = boost::add_vertex(_g);

  // Store the camera pose data.
  _g[v] = std::move(data);

  SARA_LOGI(logger,
            "[SfM] Added camera absolute pose:\n"
            "- Frame        : {}\n"
            "- Keypoints    : {}\n"
            "- Absolute pose:\n{}\n",          //
            _g[v].image_id,                    //
            features(_g[v].keypoints).size(),  //
            _g[v].pose.matrix34());

  return v;
}

auto CameraPoseGraph::add_relative_pose(const Vertex u, const Vertex v,
                                        RelativePoseData&& relative_pose_data)
    -> CameraPoseGraph::Edge
{
  const auto [e, edge_added] = boost::add_edge(u, v, _g);
  _g[e] = std::move(relative_pose_data);
  return e;
}
