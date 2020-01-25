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

  vector<vector<int>> getRedundancies(const vector<Match>& initialMatches,
                                      double thres)
  {
    Timer t;
    double elapsed;

    MatrixXd data(4, initialMatches.size());
    for (int i = 0; i < initialMatches.size(); ++i)
      data.col(i) << initialMatches[i].posX().cast<double>(),
          initialMatches[i].posY().cast<double>();

    vector<vector<int>> indices;
    indices.resize(initialMatches.size());
    for (int i = 0; i < initialMatches.size(); ++i)
      indices[i].reserve(200);

    KDTree kdtree(data);
    elapsed = t.elapsed();
    cout << "kdtree build time = " << elapsed << " seconds" << endl;

    t.restart();
    vector<int> indices_i;
    vector<double> squared_dists_i;
    indices_i.reserve(200);
    squared_dists_i.reserve(200);
    for (int i = 0; i < initialMatches.size(); ++i)
    {
      Vector4d query(data.col(i));
      kdtree.radiusSearch(query, thres * thres * 2, indices_i, squared_dists_i);
      indices[i] = indices_i;
    }
    elapsed = t.elapsed();
    cout << "redundancy computation time = " << elapsed << " seconds" << endl;

    return indices;
  }

  void getRedundancyComponentsAndRepresenters(vector<vector<int>>& components,
                                              vector<int>& representers,
                                              const vector<Match>& matches,
                                              double thres)
  {
    vector<vector<int>> redundancies = getRedundancies(matches, thres);

    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS> Graph;

    // Construct the graph.
    Graph g(matches.size());
    for (size_t i = 0; i < redundancies.size(); ++i)
      for (vector<int>::size_type j = 0; j < redundancies[i].size(); ++j)
        add_edge(i, redundancies[i][j], g);

    // Compute the connected components.
    vector<int> component(num_vertices(g));
    int num_components = connected_components(g, &component[0]);

    // Store the components.
    components = vector<vector<int>>(num_components);
    for (vector<int>::size_type i = 0; i != component.size(); ++i)
      components.reserve(100);
    for (vector<int>::size_type i = 0; i != component.size(); ++i)
      components[component[i]].push_back(i);

    // Store the best representer for each component.
    representers.resize(num_components);
    for (vector<int>::size_type i = 0; i != components.size(); ++i)
    {
      int index_best_match = components[i][0];
      for (int j = 0; j < components[i].size(); ++j)
      {
        int index = components[i][j];
        if (matches[index_best_match].score() > matches[index].score())
          index_best_match = index;
      }
      representers[i] = index_best_match;
    }
  }

}  // namespace DO::Sara
