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

#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>


namespace DO::Sara {

auto write_feature_graph(const FeatureGraph& graph, H5File& file,
                                const std::string& group_name) -> void
{
  auto features = std::vector<FeatureGID>(boost::num_vertices(graph));
  for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v)
    features[*v] = {graph[*v].image_id, graph[*v].local_id};

  auto matches = std::vector<Vector2i>{};
  for (auto [e, e_end] = boost::edges(graph); e != e_end; ++e)
    matches.push_back({boost::source(*e, graph), boost::target(*e, graph)});

  file.get_group(group_name);
  file.write_dataset(group_name + "/" + "features", tensor_view(features));
  file.write_dataset(group_name + "/" + "matches", tensor_view(matches));
}


auto read_feature_graph(H5File& file, const std::string& group_name)
    -> FeatureGraph
{
  auto features = std::vector<FeatureGID>{};
  auto matches = std::vector<Vector2i>{};

  file.read_dataset(group_name + "/" + "features", features);
  file.read_dataset(group_name + "/" + "matches", matches);

  // Reconstruct the graph.
  auto g = FeatureGraph{};

  for (const auto& v : features)
    boost::add_vertex(v, g);

  for (const auto& e : matches)
    boost::add_edge(e(0), e(1), g);

  return g;
}

} /* namespace DO::Sara */
