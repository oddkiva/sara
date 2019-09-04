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

#include <DO/Sara/Core/HDF5.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>


namespace DO::Sara {

// This is a necessary step for the bundle adjustment step.

//! @brief Feature GID.
struct FeatureGID
{
  int image_id{-1};
  int local_id{-1};
};


template <>
struct CalculateH5Type<FeatureGID>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(FeatureGID)};
    INSERT_MEMBER(h5_comp_type, FeatureGID, image_id);
    INSERT_MEMBER(h5_comp_type, FeatureGID, local_id);
    return h5_comp_type;
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

} /* namespace DO::Sara */
