// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#include "MatchRedundancy.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


using namespace std;


namespace DO::Sara {

  vector<vector<int>>
  compute_redundancy_graph(const vector<Match>& initial_matches,
                           double position_distance_thres)
  {
    auto t = Timer{};
    auto elapsed = 0.;

    auto data = MatrixXd{4, initial_matches.size()};
    for (int i = 0; i < initial_matches.size(); ++i)
      data.col(i) << initial_matches[i].x_pos().cast<double>(),
                     initial_matches[i].y_pos().cast<double>();

    auto indices = vector<vector<int>>{};
    indices.resize(initial_matches.size());
    for (int i = 0; i < initial_matches.size(); ++i)
      indices[i].reserve(200);

    KDTree kdtree(data);
    elapsed = t.elapsed();
    cout << "kdtree build time = " << elapsed << " seconds" << endl;

    t.restart();

    auto indices_i = vector<int>{};
    auto squared_dists_i = vector<double>{};
    indices_i.reserve(200);
    squared_dists_i.reserve(200);

    for (int i = 0; i < initial_matches.size(); ++i)
    {
      const Vector4d query(data.col(i));
      kdtree.radius_search(
          query, position_distance_thres * position_distance_thres * 2,
          indices_i, squared_dists_i);
      indices[i] = indices_i;
    }
    elapsed = t.elapsed();
    cout << "redundancy computation time = " << elapsed << " seconds" << endl;

    return indices;
  }

  void filter_redundant_matches(vector<vector<int>>& redundancy_components,
                                vector<int>& maxima,
                                const vector<Match>& matches,
                                double position_distance_thres)
  {
    const auto redundancy_graph =
        compute_redundancy_graph(matches, position_distance_thres);

    // Construct the graph.
    Graph g(matches.size());
    for (size_t i = 0; i < redundancy_graph.size(); ++i)
      for (vector<int>::size_type j = 0; j < redundancy_graph[i].size(); ++j)
        add_edge(i, redundancy_graph[i][j], g);

    // Compute the connected components.
    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS> Graph;
    vector<int> component(num_vertices(g));
    int num_components = connected_components(g, &component[0]);

    // Store the clusters.
    redundancy_components = vector<vector<int>>(num_components);
    for (vector<int>::size_type i = 0; i != component.size(); ++i)
      redundancy_components.reserve(100);
    for (vector<int>::size_type i = 0; i != component.size(); ++i)
      redundancy_components[component[i]].push_back(i);

    // Store the best representer for each component.
    maxima.resize(num_components);
    for (vector<int>::size_type i = 0; i != redundancy_components.size(); ++i)
    {
      int index_best_match = redundancy_components[i][0];
      for (int j = 0; j < redundancy_components[i].size(); ++j)
      {
        int index = redundancy_components[i][j];
        if (matches[index_best_match].score() > matches[index].score())
          index_best_match = index;
      }
      maxima[i] = index_best_match;
    }
  }

}  // namespace DO::Sara
