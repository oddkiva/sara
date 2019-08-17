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

#define BOOST_TEST_MODULE "MultiViewGeometry/Geometry/Point Tracks"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_connected_components)
{
  constexpr auto N = 6;
  FeatureGraph G(N);
  boost::add_edge(0, 1, G);
  boost::add_edge(1, 4, G);
  boost::add_edge(4, 0, G);
  boost::add_edge(2, 5, G);

  std::vector<int> c(num_vertices(G));
  int num = boost::connected_components(
      G, make_iterator_property_map(c.begin(),
                                    boost::get(boost::vertex_index, G), c[0]));

  std::cout << std::endl;
  std::cout << "Total number of components: " << num << std::endl;
  for (auto i = c.begin(); i != c.end(); ++i)
    std::cout << "Vertex " << i - c.begin() << " is in component " << *i
              << std::endl;
  std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_incremental_connected_components)
{
  auto graph = FeatureGraph{};
  // The documentation states that the algorithm is incremental w.r.t. to the
  // growing number of edges.
  // 
  // Things become problematic when you also try to grow the number of vertices.
  // So as a workaround two solutions:
  // - preallocate a very big number of vertices before and only add edges.
  // - restart from scratch.
  auto v0 = boost::add_vertex({0, 0}, graph);
  auto v1 = boost::add_vertex({0, 1}, graph);
  auto v2 = boost::add_vertex({0, 2}, graph);
  auto v3 = boost::add_vertex({1, 0}, graph);
  auto v4 = boost::add_vertex({1, 1}, graph);
  auto v5 = boost::add_vertex({1, 2}, graph);
  auto v6 = boost::add_vertex(graph);
  auto v7 = boost::add_vertex(graph);

  // Boilerplate code to initialize for incremental connected components.
  using ICC = IncrementalConnectedComponentsHelper;

  auto rank = ICC::initialize_ranks(graph);
  auto parent = ICC::initialize_parents(graph);
  auto ds = ICC::initialize_disjoint_sets(rank, parent);
  ICC::initialize_incremental_components(graph, ds);

  for (auto r: rank)
    std::cout << "rank = " << r << std::endl;

  for (auto p: parent)
    std::cout << "p = " << p << std::endl;

  auto add_edge = [&](auto u, auto v) {
    boost::add_edge(u, v, graph);
    ds.union_set(u, v);
  };

  auto find_sets = [&](const auto& graph, auto& ds) {
    for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v)
      std::cout << "representative[" << *v << "] = " << ds.find_set(*v)
                << std::endl;
    std::cout << std::endl;
  };

  auto print_graph = [&](const auto& graph) {
    std::cout << "An undirected graph:" << std::endl;
    boost::print_graph(graph, boost::get(boost::vertex_index, graph));
    std::cout << std::endl;
  };

  auto print_components = [&](const auto& components) {
    for (auto c : components)
    {
      std::cout << "component " << c << " contains: ";

      for (auto [child, child_end] = components[c]; child != child_end; ++child)
        std::cout << *child << " ";
      std::cout << std::endl;
    }
  };

  auto calculate_components_as_edges_are_added = [&]() {
    add_edge(v0, v1);
    add_edge(v1, v4);
    add_edge(v4, v0);
    add_edge(v2, v5);

    print_graph(graph);

    find_sets(graph, ds);

    auto components = ICC::get_components(parent);
    print_components(components);
  };

  calculate_components_as_edges_are_added();


  // Now let's see if we add more vertices and edges.
  add_edge(v0, v6);
  add_edge(v6, v7);

  {
    auto components = ICC::get_components(parent);
    std::cout << "OK" << std::endl;
    print_components(components);
  }
}
