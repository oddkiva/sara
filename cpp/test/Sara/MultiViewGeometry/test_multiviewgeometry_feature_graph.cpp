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

#define BOOST_TEST_MODULE "MultiViewGeometry/Geometry/Feature Graph"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
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
  /* auto v3 = */ boost::add_vertex({1, 0}, graph);
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


BOOST_AUTO_TEST_CASE(test_read_write_feature_graph_to_hdf5)
{
  auto graph = FeatureGraph{};
  auto v0 = boost::add_vertex({0, 0}, graph);
  auto v1 = boost::add_vertex({0, 1}, graph);
  /*auto v2 = */ boost::add_vertex({0, 2}, graph);
  auto v3 = boost::add_vertex({1, 0}, graph);
  /* auto v4 = */ boost::add_vertex({1, 1}, graph);
  auto v5 = boost::add_vertex({1, 2}, graph);

  boost::add_edge(v0, v5, graph);
  boost::add_edge(v1, v3, graph);

  const auto filepath = (fs::temp_directory_path() / "feature_graph.h5").string();

  // Write data to HDF5.
  SARA_DEBUG << "WRITE PHASE" << std::endl;
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};
    write_feature_graph(graph, h5file, "feature_graph");
  }

  // Read data to HDF5.
  SARA_DEBUG << "READ PHASE" << std::endl;
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    const auto graph_read = read_feature_graph(h5file, "feature_graph");

    for (auto [v, v_end] = boost::vertices(graph_read); v != v_end; ++v)
    {
      BOOST_CHECK_EQUAL(graph_read[*v].image_id, graph_read[*v].image_id);
      BOOST_CHECK_EQUAL(graph_read[*v].local_id, graph_read[*v].local_id);
    }

    for (auto [e, e_end] = boost::edges(graph_read); e != e_end; ++e)
    {
      const auto u = boost::source(*e, graph_read);
      const auto v = boost::target(*e, graph_read);
      const auto edge_found = boost::edge(u, v, graph).second;
      BOOST_CHECK(edge_found);
    }

    BOOST_CHECK_EQUAL(boost::num_vertices(graph),
                      boost::num_vertices(graph_read));
    BOOST_CHECK_EQUAL(boost::num_edges(graph), boost::num_edges(graph_read));
  }
}


BOOST_AUTO_TEST_CASE(test_populate_feature_gids)
{
  auto keys = std::vector{
    KeypointList<OERegion, float>{},
    KeypointList<OERegion, float>{},
    KeypointList<OERegion, float>{}
  };

  features(keys[0]).resize(3);
  features(keys[1]).resize(1);
  features(keys[2]).resize(2);

  descriptors(keys[0]).resize({3, 10});
  descriptors(keys[1]).resize({1, 10});
  descriptors(keys[2]).resize({2, 10});

  const auto feature_gids = populate_feature_gids(keys);
  const auto true_feature_gids =
      std::vector<FeatureGID>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0}, {2, 1}};
  BOOST_CHECK(feature_gids == true_feature_gids);
}

BOOST_AUTO_TEST_CASE(test_calculate_of_feature_id_offset)
{
  auto keys = std::vector{
    KeypointList<OERegion, float>{},
    KeypointList<OERegion, float>{},
    KeypointList<OERegion, float>{}
  };

  features(keys[0]).resize(3);
  features(keys[1]).resize(1);
  features(keys[2]).resize(2);

  descriptors(keys[0]).resize({3, 10});
  descriptors(keys[1]).resize({1, 10});
  descriptors(keys[2]).resize({2, 10});

  const auto fid_offsets = calculate_feature_id_offsets(keys);
  const auto true_fid_offsets = std::vector{0, 3, 4};
  BOOST_CHECK(fid_offsets == true_fid_offsets);
}

BOOST_AUTO_TEST_CASE(test_populate_feature_tracks)
{
  // Construct a dataset containing 3 views.
  const auto num_views = 3;
  auto views = ViewAttributes{};
  {
    views.keypoints.resize(3);
    features(views.keypoints[0]).resize(3);
    features(views.keypoints[1]).resize(4);
    features(views.keypoints[2]).resize(2);

    descriptors(views.keypoints[0]).resize({3, 10});
    descriptors(views.keypoints[1]).resize({4, 10});
    descriptors(views.keypoints[2]).resize({2, 10});
  }

  // Construct matches.
  auto epipolar_edges = EpipolarEdgeAttributes{};
  epipolar_edges.matches = {
      // Image 0 - Image 1
      //
      // (0, 0) - (1, 0)
      // (0, 1) - (1, 1)
      // (0, 2) - (1, 2)
      {make_index_match(0, 0), make_index_match(1, 1), make_index_match(2, 2)},
      // Image 0 - Image 2
      //
      // (0, 0) - (2, 0)
      // (0, 1) - (2, 1)
      {make_index_match(0, 0), make_index_match(1, 1)},
      // Image 1 - Image 2
      //
      // (1, 0) - (2, 0)
      // (1, 1) - (2, 1)
      {make_index_match(0, 0), make_index_match(1, 1)}};

  // Allocate the edges.
  epipolar_edges.initialize_edges(num_views);
  epipolar_edges.resize_essential_edge_list();

  // Make sure the matches are marked as inliers.
  for (const auto& ij : epipolar_edges.edge_ids)
  {
    const auto& matches_ij = epipolar_edges.matches[ij];
    auto& E_inliers_ij = epipolar_edges.E_inliers[ij];
    E_inliers_ij = Tensor_<bool, 1>{int(matches_ij.size())};
    E_inliers_ij.flat_array().fill(true);
  }

  // Make sure the matches are also marked as cheiral.
  epipolar_edges.two_view_geometries.resize(3);
  for (const auto& ij : epipolar_edges.edge_ids)
  {
    const auto& matches_ij = epipolar_edges.matches[ij];
    auto& cheirality_ij = epipolar_edges.two_view_geometries[ij].cheirality;
    cheirality_ij.resize(matches_ij.size());
    cheirality_ij.fill(true);
  }

  const auto [g_, c_] = populate_feature_tracks(views, epipolar_edges);
  const auto graph = g_;
  const auto components = c_;

  BOOST_CHECK_EQUAL(boost::num_vertices(graph), 9);
  BOOST_CHECK_EQUAL(boost::num_edges(graph), 7);
  BOOST_CHECK_EQUAL(components.size(), 4);

  for (auto c = 0u; c < components.size(); ++c)
  {
    const auto& component = components[c];

    std::cout << "Component " << c << " : ";
    for (const auto& v: component)
      std::cout << "GID[" << v << "] = {" << graph[v].image_id << ", "
                << graph[v].local_id << "}, ";
    std::cout << std::endl;
  }


  const auto feature_tracks_filtered = filter_feature_tracks(graph, components);
  for (const auto& feature_track : feature_tracks_filtered)
  {
    std::cout << "feature track : ";
    for (const auto& f : feature_track)
      std::cout << "{" << f.image_id << ", " << f.local_id << "}, ";
    std::cout << std::endl;
  }

  // Construct the connected components manually from the following matches:
  //
  // Image 0 - Image 1
  // (0, 0) - (1, 0)
  // (0, 1) - (1, 1)
  // (0, 2) - (1, 2)
  //
  // Image 0 - Image 2
  // (0, 0) - (2, 0)
  // (0, 1) - (2, 1)
  //
  // Image 1 - Image 2
  // (1, 0) - (2, 0)
  // (1, 1) - (2, 1)
  //
  //
  // The connected components of size at least 2 are:
  // (0, 0) - (1, 0) - (2, 0)
  // (0, 1) - (1, 1) - (2, 1)
  // (0, 2) - (1, 2)
  auto true_feature_tracks_filtered = std::set<std::set<FeatureGID>>{};
  true_feature_tracks_filtered.insert(
      std::set<FeatureGID>{{0, 0}, {1, 0}, {2, 0}});
  true_feature_tracks_filtered.insert(
      std::set<FeatureGID>{{0, 1}, {1, 1}, {2, 1}});
  true_feature_tracks_filtered.insert(std::set<FeatureGID>{{0, 2}, {1, 2}});

  BOOST_CHECK(feature_tracks_filtered == true_feature_tracks_filtered);
}
