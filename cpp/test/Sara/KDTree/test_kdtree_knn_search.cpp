// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "KDTree/K-Nearest Neighbor Search"

#include <set>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/KDTree.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


inline vector<int> range(int begin, int end)
{
  vector<int> _range(end - begin);
  for (int i = begin; i < end; ++i)
    _range[i - begin] = i;
  return _range;
}

inline vector<int> range(int end)
{
  return range(0, end);
}


class TestFixtureForKDTree
{
protected:
  MatrixXd data;
  size_t num_points;
  size_t num_points_in_each_circle;

public:
  TestFixtureForKDTree()
  {
    // We construct two sets in points. The first one lives in the
    // zero-centered unit circle and the second in the zero-centered
    // circle with radius 10.
    num_points_in_each_circle = 20;
    num_points = 2 * num_points_in_each_circle;
    data.resize(2, num_points);

    const size_t& N = num_points_in_each_circle;
    for (size_t i = 0; i < N; ++i)
    {
      double theta = (2 * i * M_PI) / N;
      data.col(i) << cos(theta), sin(theta);
    }

    for (size_t i = N; i < 2 * N; ++i)
    {
      double theta = (2 * (i - N) * M_PI) / N;
      data.col(i) << 10 * cos(theta), 10 * sin(theta);
    }
  }
};

BOOST_FIXTURE_TEST_SUITE(TestKDTree, TestFixtureForKDTree)

BOOST_AUTO_TEST_CASE(test_simple_knn_search)
{
  KDTree tree(data);

  Vector2d query = Vector2d::Zero();
  size_t num_nearest_neighbors = num_points_in_each_circle;

  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  tree.knn_search(query, static_cast<int>(num_nearest_neighbors), nn_indices,
                  nn_squared_distances);

  // Check equality of items.
  BOOST_CHECK_ITEMS_EQUAL(nn_indices,
                          range(static_cast<int>(num_points_in_each_circle)));

  // Check the squared distances.
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_points_in_each_circle);
  for (size_t j = 0; j < nn_squared_distances.size(); ++j)
    BOOST_CHECK_SMALL(nn_squared_distances[j] - 1., 1e-10);
}

BOOST_AUTO_TEST_CASE(test_simple_knn_search_with_query_point_in_data)
{
  KDTree tree(data);

  size_t query_index = 0;
  size_t num_nearest_neighbors = num_points_in_each_circle - 1;

  vector<int> indices;
  vector<double> squared_distances;

  tree.knn_search(query_index, static_cast<int>(num_nearest_neighbors), indices,
                  squared_distances);

  // Check the indices of the neighbors.
  BOOST_CHECK_ITEMS_EQUAL(
      indices, range(1, static_cast<int>(num_points_in_each_circle)));

  // Check the squared distances of the neighbors.
  BOOST_CHECK_EQUAL(num_nearest_neighbors, squared_distances.size());
  for (size_t j = 0; j < squared_distances.size(); ++j)
    BOOST_CHECK_LE(squared_distances[j], pow(2. + 1e-5, 2));
}

BOOST_AUTO_TEST_CASE(test_batch_knn_search)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_each_circle;
  const size_t& num_nearest_neighbors = num_points_in_each_circle;
  MatrixXd queries(data.leftCols(num_queries));

  // In-out data.
  vector<vector<int>> indices;
  vector<vector<double>> squared_distances;

  // Use case.
  tree.knn_search(queries, static_cast<int>(num_nearest_neighbors), indices,
                  squared_distances);

  // Check the contents of the retrieval.
  BOOST_CHECK_EQUAL(indices.size(), num_queries);
  BOOST_CHECK_EQUAL(squared_distances.size(), num_queries);

  for (size_t i = 0; i < num_queries; ++i)
  {
    // Check the indices.
    BOOST_CHECK_ITEMS_EQUAL(indices[i],
                            range(static_cast<int>(num_points_in_each_circle)));

    // Check the squared distances.
    BOOST_CHECK_EQUAL(num_nearest_neighbors, squared_distances.size());
    for (size_t j = 0; j < squared_distances[i].size(); ++j)
      BOOST_CHECK_LE(squared_distances[i][j], pow(2. + 1e-5, 2));
  }
}

BOOST_AUTO_TEST_CASE(test_batch_knn_search_with_query_point_in_data)
{
  // Input data.
  KDTree tree(data);
  const size_t num_queries = num_points_in_each_circle;
  const size_t num_nearest_neighbors = num_points_in_each_circle - 1;

  vector<size_t> queries(num_queries);
  for (size_t i = 0; i != queries.size(); ++i)
    queries[i] = i;

  // In-out data.
  vector<vector<int>> indices;
  vector<vector<double>> squared_distances;

  // Use case.
  tree.knn_search(queries, static_cast<int>(num_nearest_neighbors), indices,
                  squared_distances);

  // Check the number of queries.
  BOOST_CHECK_EQUAL(indices.size(), num_queries);
  BOOST_CHECK_EQUAL(squared_distances.size(), num_queries);

  // Check the contents of the retrieval.
  for (size_t i = 0; i != indices.size(); ++i)
  {
    // The correct list of indices is: {0, 1, 2, 3, 4 } \ {i},
    // where i = 0, 1, ... 4.
    auto true_indices = range(static_cast<int>(num_points_in_each_circle));
    true_indices.erase(true_indices.begin() + i);

    BOOST_REQUIRE_ITEMS_EQUAL(indices[i], true_indices);

    // Check the squared distances.
    BOOST_CHECK_EQUAL(squared_distances[i].size(), true_indices.size());
    for (size_t j = 0; j < indices[i].size(); ++j)
      BOOST_CHECK_LE(squared_distances[i][j], pow(2. + 1e-5, 2));
  }
}

BOOST_AUTO_TEST_SUITE_END()
