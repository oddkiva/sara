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

#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGID.hpp>


namespace DO::Sara {

  class FeatureGraph
  {
  public:
    using GraphImpl = boost::adjacency_list<           //
        boost::vecS, boost::vecS, boost::undirectedS,  //
        FeatureGID>;

    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = boost::graph_traits<Graph>::edge_descriptor;
    using Track = std::vector<Vertex>;

    auto operator[](Vertex v) -> FeatureGID&
    {
      return _feature_graph[v];
    }

    auto operator[](Vertex v) const -> const FeatureGID&
    {
      return _feature_graph[v];
    }

    auto calculate_feature_tracks() const -> std::vector<Track>;
    auto filter_by_non_max_suppression(const Track&,
                                       const CameraPoseGraph&) const -> Track;
    auto find_vertex_from_camera_view(const Track&,
                                      const CameraPoseGraph::Vertex&) const
        -> Vertex;

  private:
    GraphImpl _feature_graph;
  };

} /* namespace DO::Sara */
