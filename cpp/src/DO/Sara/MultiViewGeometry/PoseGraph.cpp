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

#include <DO/Sara/MultiViewGeometry/PoseGraph.hpp>


namespace DO::Sara {


struct EpipolarEdge
{
  int u;
  int v;
  int edge_id;
  double weight;
};


template <>
struct CalculateH5Type<PoseID>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(PoseID)};
    INSERT_MEMBER(h5_comp_type, PoseID, weight);
    return h5_comp_type;
  }
};


template <>
struct CalculateH5Type<EpipolarEdge>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(EpipolarEdge)};
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, u);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, v);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, edge_id);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, weight);
    return h5_comp_type;
  }
};


auto write_pose_graph(const PoseGraph& graph, H5File& file,
                      const std::string& group_name) -> void
{
  auto poses = std::vector<PoseID>(boost::num_vertices(graph));
  for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v)
    poses[*v] = {graph[*v].weight};

  auto epipolar_edges = std::vector<EpipolarEdge>{};
  for (auto [e, e_end] = boost::edges(graph); e != e_end; ++e)
    epipolar_edges.push_back({int(boost::source(*e, graph)),
                              int(boost::target(*e, graph)),  //
                              int(graph[*e].id), graph[*e].weight});

  file.get_group(group_name);
  file.write_dataset(group_name + "/" + "poses", tensor_view(poses));
  file.write_dataset(group_name + "/" + "epipolar_edges", tensor_view(epipolar_edges));
}


auto read_pose_graph(H5File& file, const std::string& group_name) -> PoseGraph
{
  auto poses = std::vector<PoseID>{};
  auto epipolar_edges = std::vector<EpipolarEdge>{};

  file.read_dataset(group_name + "/" + "poses", poses);
  file.read_dataset(group_name + "/" + "epipolar_edges", epipolar_edges);

  // Reconstruct the graph.
  auto g = PoseGraph{};

  for (const auto& v : poses)
    boost::add_vertex(v, g);

  for (const auto& edge : epipolar_edges)
  {
    auto e = boost::add_edge(edge.u, edge.v, g);
    g[e.first] = {edge.edge_id, edge.weight};
  }

  return g;
}

} /* namespace DO::Sara */
