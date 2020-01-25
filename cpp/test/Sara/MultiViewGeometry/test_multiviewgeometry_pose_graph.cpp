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

#define BOOST_TEST_MODULE "MultiViewGeometry/Geometry/Pose Graph"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/MultiViewGeometry/PoseGraph.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_read_write_pose_graph_to_hdf5)
{
  auto graph = PoseGraph{};
  auto v0 = boost::add_vertex({0}, graph);
  auto v1 = boost::add_vertex({1}, graph);
  /* auto v2 = */boost::add_vertex({2}, graph);
  auto v3 = boost::add_vertex({3}, graph);
  /* auto v4 = */ boost::add_vertex({4}, graph);
  auto v5 = boost::add_vertex({5}, graph);

  auto [e0, b0] = boost::add_edge(v0, v5, graph);
  auto [e1, b1] = boost::add_edge(v1, v3, graph);

  (void) b0;
  (void) b1;

  graph[e0].id = 0;
  graph[e0].weight = 6;

  graph[e1].id = 1;
  graph[e1].weight = 4;

  const auto filepath = (fs::temp_directory_path() / "pose_graph.h5").string();

  // Write data to HDF5.
  SARA_DEBUG << "WRITE PHASE" << std::endl;
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};
    write_pose_graph(graph, h5file, "feature_graph");
  }

  // Read data to HDF5.
  SARA_DEBUG << "READ PHASE" << std::endl;
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    const auto graph_read = read_pose_graph(h5file, "feature_graph");

    for (auto [v, v_end] = boost::vertices(graph_read); v != v_end; ++v)
      BOOST_CHECK_EQUAL(graph_read[*v].weight, graph_read[*v].weight);

    for (auto [e, e_end] = boost::edges(graph_read); e != e_end; ++e)
    {
      const auto u = boost::source(*e, graph_read);
      const auto v = boost::target(*e, graph_read);
      const auto [edge, found] = boost::edge(u, v, graph);
      SARA_CHECK(edge);
      SARA_CHECK(u);
      SARA_CHECK(v);
      SARA_CHECK(graph_read[edge].id);
      SARA_CHECK(graph_read[edge].weight);

      BOOST_CHECK_EQUAL(graph[edge].id, graph_read[edge].id);
      BOOST_CHECK_EQUAL(graph[edge].weight, graph_read[edge].weight);
      BOOST_CHECK(found);
    }

    BOOST_CHECK_EQUAL(boost::num_vertices(graph),
                      boost::num_vertices(graph_read));
    BOOST_CHECK_EQUAL(boost::num_edges(graph), boost::num_edges(graph_read));
  }
}
