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

#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/graph/incremental_components.hpp>


using namespace DO::Sara;


auto FeatureGraph::calculate_feature_tracks() const
    -> std::vector<FeatureGraph::Track>
{
  using VertexIndex = boost::graph_traits<FeatureGraph::GraphImpl>::vertices_size_type;

  using Rank = VertexIndex*;
  using Parent = Vertex*;

  using DisjointSets = boost::disjoint_sets<Rank, Parent>;

  using Components = boost::component_index<VertexIndex>;

  const auto num_vertices = boost::num_vertices(_feature_graph);
  auto ranks = std::vector<VertexIndex>(num_vertices);
  auto parents = std::vector<Vertex>(num_vertices);

  // Initialize the disjoint sets.
  auto ds = boost::disjoint_sets<Rank, Parent>(&ranks[0], parents.data());
  boost::initialize_incremental_components(_feature_graph, ds);
  boost::incremental_components(_feature_graph, ds);

  const auto components = Components{parents.begin(), parents.end()};
  for (auto [e, e_end] = boost::edges(_feature_graph); e != e_end; ++e)
  {
    const auto u = boost::source(*e, _feature_graph);
    const auto v = boost::target(*e, _feature_graph);
    ds.union_set(u, v);
  }

  return {};
}
