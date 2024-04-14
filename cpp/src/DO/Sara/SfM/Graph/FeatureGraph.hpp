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


namespace DO::Sara {

  //! @brief Feature Global ID (GID).
  struct FeatureGID
  {
    CameraPoseGraph::Vertex pose_vertex;
    int feature_index;

    auto operator==(const FeatureGID& other) const -> bool
    {
      return pose_vertex == other.pose_vertex &&
             feature_index == other.feature_index;
    }

    auto operator<(const FeatureGID& other) const -> bool
    {
      return (pose_vertex < other.pose_vertex) ||
             (pose_vertex == other.pose_vertex &&
              feature_index < other.feature_index);
    }
  };

  //! @brief Match global ID (GID).
  struct MatchGID
  {
    //! @brief Index of the epipolar edge connecting camera i and camera j.
    CameraPoseGraph::Vertex pose_src;
    CameraPoseGraph::Vertex pose_dst;
    //! @brief Local match index.
    std::size_t index;

    auto operator==(const MatchGID& other) const -> bool
    {
      return pose_src == other.pose_src && pose_dst == other.pose_dst &&
             index == other.index;
    }

    auto operator<(const MatchGID& other) const -> bool
    {
      return (pose_src < other.pose_src) ||
             (pose_src == other.pose_src && pose_dst < other.pose_dst) ||
             (pose_src == other.pose_src && pose_dst == other.pose_dst &&
              index < other.index);
    }
  };

  //! @brief Feature Graph.
  class FeatureGraph
  {
  public:
    using Impl = boost::adjacency_list<                //
        boost::vecS, boost::vecS, boost::undirectedS,  //
        FeatureGID, MatchGID>;
    using Vertex = boost::graph_traits<Impl>::vertex_descriptor;
    using VertexIndex = boost::graph_traits<Impl>::vertices_size_type;
    using Edge = boost::graph_traits<Impl>::edge_descriptor;
    using Track = std::vector<Vertex>;

    operator Impl&()
    {
      return _feature_graph;
    }

    operator const Impl&() const
    {
      return _feature_graph;
    }

    auto operator[](Vertex v) -> FeatureGID&
    {
      return _feature_graph[v];
    }

    auto operator[](Vertex v) const -> const FeatureGID&
    {
      return _feature_graph[v];
    }

    auto num_vertices() const -> VertexIndex
    {
      return boost::num_vertices(_feature_graph);
    }

    auto calculate_feature_tracks() const -> std::vector<Track>;
    auto filter_by_non_max_suppression(const Track&,
                                       const CameraPoseGraph&) const -> Track;
    auto find_vertex_from_camera_view(
        const Track&, const CameraPoseGraph::Vertex&) const -> Vertex;

  private:
    Impl _feature_graph;
  };

} /* namespace DO::Sara */
