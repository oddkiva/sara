// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>

#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>


namespace DO::Sara {

  //! @brief Feature disjoint sets for feature tracking.
  struct FeatureDisjointSets
  {
    using Rank = FeatureGraph::Vertex*;
    using Parent = FeatureGraph::Vertex*;
    using Components = boost::component_index<FeatureGraph::VertexIndex>;

    FeatureDisjointSets() = default;

    explicit FeatureDisjointSets(const FeatureGraph& graph)
    {
      const FeatureGraph::Impl& g = graph;

      const auto n = graph.num_vertices();
      _rank.resize(n);
      _parent.resize(n);
      _ds.reset(new boost::disjoint_sets<Rank, Parent>(&_rank[0], &_parent[0]));
      if (_ds.get() == nullptr)
        throw std::runtime_error{
            "Failed to allocate and initialize the feature disjoint sets"};
      boost::initialize_incremental_components(g, *_ds);
      boost::incremental_components(g, *_ds);
    }

    auto components() const -> Components
    {
      return {_parent.begin(), _parent.end()};
    }

    std::vector<FeatureGraph::VertexIndex> _rank;
    std::vector<FeatureGraph::Vertex> _parent;
    std::unique_ptr<boost::disjoint_sets<Rank, Parent>> _ds;
  };

}  // namespace DO::Sara
