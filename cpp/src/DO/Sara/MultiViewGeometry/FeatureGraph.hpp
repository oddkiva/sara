// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>


namespace DO::Sara {

//! @brief Feature GID.
struct FeatureGID
{
  int image_id{-1};
  int local_id{-1};

  auto operator==(const FeatureGID& other) const -> bool
  {
    return image_id == other.image_id && local_id == other.local_id;
  }

  auto operator<(const FeatureGID& other) const -> bool
  {
    return std::make_pair(image_id, local_id) <
           std::make_pair(other.image_id, other.local_id);
  }
};


using FeatureGraph = boost::adjacency_list<boost::vecS, boost::vecS,
                                           boost::undirectedS, FeatureGID>;


struct IncrementalConnectedComponentsHelper
{
  using Vertex = boost::graph_traits<FeatureGraph>::vertex_descriptor;
  using VertexIndex = boost::graph_traits<FeatureGraph>::vertices_size_type;

  using Rank = VertexIndex*;
  using Parent = Vertex*;

  using RankList = std::vector<VertexIndex>;
  using ParentList = std::vector<VertexIndex>;

  using DisjointSets = boost::disjoint_sets<Rank, Parent>;

  using Components = boost::component_index<VertexIndex>;

  static auto initialize_ranks(const FeatureGraph& graph)
  {
    return RankList(boost::num_vertices(graph));
  }

  static auto initialize_parents(const FeatureGraph& graph)
  {
    return ParentList(boost::num_vertices(graph));
  }

  static auto initialize_disjoint_sets(RankList& rank, ParentList& parent)
  {
    return DisjointSets(&rank[0], &parent[0]);
  }

  static auto get_components(const ParentList& parents)
  {
    return Components{parents.begin(), parents.end()};
  }

  static auto initialize_incremental_components(FeatureGraph& graph,
                                                DisjointSets& ds)
  {
    boost::initialize_incremental_components(graph, ds);
    boost::incremental_components(graph, ds);
  }
};


DO_SARA_EXPORT
auto populate_feature_gids(
    const std::vector<KeypointList<OERegion, float>>& keypoints)
    -> std::vector<FeatureGID>;

DO_SARA_EXPORT
auto calculate_feature_id_offsets(
    const std::vector<KeypointList<OERegion, float>>& keypoints)
    -> std::vector<int>;

DO_SARA_EXPORT
auto populate_feature_tracks(const ViewAttributes& views,
                             const EpipolarEdgeAttributes& epipolar_edges)
    -> std::pair<FeatureGraph, std::vector<std::vector<int>>>;

DO_SARA_EXPORT
auto filter_feature_tracks(const FeatureGraph& graph,
                           const std::vector<std::vector<int>>& components)
    -> std::set<std::set<FeatureGID>>;


//! @brief write feature graph to HDF5.
DO_SARA_EXPORT
auto write_feature_graph(const FeatureGraph& graph, H5File& file,
                         const std::string& group_name) -> void;

//! @brief read feature graph from HDF5.
DO_SARA_EXPORT
auto read_feature_graph(H5File& file, const std::string& group_name)
    -> FeatureGraph;

} /* namespace DO::Sara */
