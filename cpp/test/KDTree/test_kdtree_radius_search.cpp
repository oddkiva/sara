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

//! TODO: this is a messy set of unit tests. It maybe worth reinstating Google
//! mock in the library. We can compare easily vectors.

#define BOOST_TEST_MODULE "KDTree/Radius Search"

#include <set>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/KDTree.hpp>

#include "../AssertHelpers.hpp"


using namespace DO::Sara;
using namespace std;


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
  MatrixXd _data;
  size_t _num_points;
  size_t _num_points_in_each_circle;
  double _small_circle_radius;
  double _large_circle_radius;

public:
  TestFixtureForKDTree()
  {
    // We construct two sets in points. The first one lives in the
    // zero-centered unit circle and the second in the zero-centered
    // circle with radius 10.
    _num_points_in_each_circle = 30;
    _num_points = 2 * _num_points_in_each_circle;
    _small_circle_radius = 2.;
    _large_circle_radius = 10.;
    _data.resize(2, _num_points);

    const auto& N = _num_points_in_each_circle;
    for (auto i = 0u; i < N; ++i)
    {
      auto theta = (2 * i * M_PI) / N;
      _data.col(i) << _small_circle_radius * cos(theta),
          _small_circle_radius * sin(theta);
    }

    for (size_t i = N; i < 2 * N; ++i)
    {
      double theta = (2 * (i - N) * M_PI) / N;
      _data.col(i) << _large_circle_radius * cos(theta),
          _large_circle_radius * sin(theta);
    }
  }
};


BOOST_FIXTURE_TEST_SUITE(TestKDTree, TestFixtureForKDTree)

BOOST_AUTO_TEST_CASE(test_simple_radius_search_default_use)
{
  // Input data.
  KDTree tree(_data);
  Vector2d query = Vector2d::Zero();

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Regular use case.
  size_t num_nearest_neighbors = _num_points_in_each_circle;
  // FLANN is very imprecise. To find points in a zero-centered circle with
  // radius 2. The search radius must be set to very large!
  double squared_search_radius = pow(4., 2);
  int num_found_neighbors = tree.radius_search(
      query, squared_search_radius, nn_indices, nn_squared_distances);

  // Check the number of neighbors.
  BOOST_CHECK_EQUAL(nn_indices.size(), num_nearest_neighbors);
  BOOST_CHECK_EQUAL(num_found_neighbors,
                    static_cast<int>(num_nearest_neighbors));

  // Check the indices of the nearest neighbors.
  BOOST_CHECK_ITEMS_EQUAL(range(_num_points_in_each_circle), nn_indices);

  // Check the squared distances.
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_nearest_neighbors);
  for (size_t j = 0; j < nn_squared_distances.size(); ++j)
    BOOST_REQUIRE_SMALL(nn_squared_distances[j] - pow(_small_circle_radius, 2),
                        1e-10);
}


BOOST_AUTO_TEST_CASE(test_simple_radius_search_with_restricted_num_of_neighbors)
{
  // Input data.
  KDTree tree(_data);
  Vector2d query = Vector2d::Zero();
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  int num_found_neighbors =
      tree.radius_search(query, squared_search_radius, nn_indices,
                         nn_squared_distances, max_num_nearest_neighbors);

  // Check the number of nearest neighbors.
  BOOST_CHECK_EQUAL(nn_indices.size(), max_num_nearest_neighbors);
  BOOST_CHECK_EQUAL(num_found_neighbors,
                    static_cast<int>(max_num_nearest_neighbors));

  // Check the contents of the containers.
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    BOOST_REQUIRE_LT(nn_indices[j],
                     static_cast<int>(_num_points_in_each_circle));
    BOOST_REQUIRE_LT(nn_squared_distances[j] - 4., 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(test_simple_radius_search_with_query_point_in_data_default)
{
  // Input data.
  KDTree tree(_data);
  size_t query = 0;
  size_t num_nearest_neighbors = _num_points_in_each_circle - 1;

  // Squared diameter of the inner circle is: 2**2.
  // Search radius must be coarse. FLANN is not very precise...
  double squared_search_radius = pow(2 * 3, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;

  // Default use case.
  num_found_neighbors = tree.radius_search(query, squared_search_radius,
                                           nn_indices, nn_squared_distances);

  // Check the number of neighbors.
  BOOST_CHECK_EQUAL(nn_indices.size(), num_nearest_neighbors);
  BOOST_CHECK_EQUAL(num_found_neighbors,
                    static_cast<int>(num_nearest_neighbors));

  // Check the indices.
  BOOST_CHECK_ITEMS_EQUAL(nn_indices, range(1, _num_points_in_each_circle));

  // Check the squared distances.
  for (size_t j = 0; j < nn_indices.size(); ++j)
    BOOST_REQUIRE_LE(nn_squared_distances[j], pow(2 * 2., 2));
}


BOOST_AUTO_TEST_CASE(
    test_simple_radius_search_with_query_point_in_data_restricted)
{
  // Input data.
  KDTree tree(_data);
  size_t query = 0;
  double squared_search_radius = pow(2 * 2.0001, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;
  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  num_found_neighbors =
      tree.radius_search(query, squared_search_radius, nn_indices,
                         nn_squared_distances, max_num_nearest_neighbors);

  // Check the number of indices.
  BOOST_CHECK_EQUAL(nn_indices.size(), max_num_nearest_neighbors);
  BOOST_CHECK_EQUAL(num_found_neighbors,
                    static_cast<int>(max_num_nearest_neighbors));

  // Check the number of squared distances.
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), max_num_nearest_neighbors);

  // Check the contents of the retrieval.
  for (size_t j = 0; j < nn_squared_distances.size(); ++j)
  {
    // Check the index value.
    BOOST_REQUIRE(nn_indices[j] != 0);
    BOOST_REQUIRE_LT(nn_indices[j],
                     static_cast<int>(_num_points_in_each_circle));

    // Check the squared distances.
    BOOST_REQUIRE_LT(nn_squared_distances[j], squared_search_radius);
  }
}

BOOST_AUTO_TEST_CASE(test_batch_radius_search_default)
{
  // Input data.
  KDTree tree(_data);
  const auto& num_queries = _num_points_in_each_circle;
  MatrixXd queries(_data.leftCols(num_queries));
  // FLANN is imprecise again...
  double squared_search_radius = pow(2 * 3.5, 2);

  // In-out data.
  vector<vector<int>> nn_indices;
  vector<vector<double>> nn_squared_distances;

  // Use case.
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances);

  // Check the number of queries.
  BOOST_CHECK_EQUAL(nn_indices.size(), num_queries);
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_queries);

  // Check the content of the retrieval.
  for (auto i = 0u; i < num_queries; ++i)
  {
    BOOST_REQUIRE_EQUAL(nn_indices[i].size(), _num_points_in_each_circle);
    BOOST_REQUIRE_ITEMS_EQUAL(nn_indices[i], range(_num_points_in_each_circle));

    BOOST_REQUIRE_EQUAL(nn_squared_distances[i].size(),
                      _num_points_in_each_circle);
    for (size_t j = 0; j < nn_squared_distances[i].size(); ++j)
      BOOST_REQUIRE_LT(nn_squared_distances[i][j], pow(2 * 2.00001, 2));
  }
}

BOOST_AUTO_TEST_CASE(test_batch_radius_search_restricted)
{
  // Input data.
  KDTree tree(_data);
  const size_t& num_queries = _num_points_in_each_circle;
  MatrixXd queries(_data.leftCols(num_queries));
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<vector<int>> nn_indices;
  vector<vector<double>> nn_squared_distances;

  // Use case.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances, max_num_nearest_neighbors);

  // Check the number of queries.
  BOOST_CHECK_EQUAL(nn_indices.size(), num_queries);
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_queries);

  for (size_t i = 0; i < num_queries; ++i)
  {
    BOOST_CHECK_EQUAL(nn_indices[i].size(), max_num_nearest_neighbors);
    BOOST_CHECK_EQUAL(nn_squared_distances[i].size(),
                      max_num_nearest_neighbors);

    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      BOOST_REQUIRE_LT(nn_indices[i][j],
                       static_cast<int>(_num_points_in_each_circle));
      BOOST_REQUIRE_LT(nn_squared_distances[i][j], squared_search_radius);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_batch_radius_search_with_query_point_in_data_default)
{
  // Input data.
  KDTree tree(_data);
  const size_t& num_queries = _num_points_in_each_circle;
  vector<size_t> queries;
  for (size_t i = 0; i < num_queries; ++i)
    queries.push_back(i);
  // FLANN is very imprecise...
  double squared_search_radius = pow(2 * 3.4, 2);

  // In-out data.
  vector<vector<int>> nn_indices;
  vector<vector<double>> nn_squared_distances;

  // Default use case.
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances);

  // Check the number of queries.
  BOOST_CHECK_EQUAL(nn_indices.size(), num_queries);
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_queries);

  // Check the contents of the retrieval.
  for (size_t i = 0; i < num_queries; ++i)
  {
    vector<int> true_indices = range(_num_points_in_each_circle);
    true_indices.erase(true_indices.begin() + i);

    // Check the indices.
    BOOST_REQUIRE_EQUAL(nn_indices[i].size(), _num_points_in_each_circle - 1);
    BOOST_REQUIRE_ITEMS_EQUAL(true_indices, nn_indices[i]);

    // Check the squared distances.
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
      BOOST_REQUIRE_LT(nn_squared_distances[i][j], pow(2 * 2.00000001, 2));
  }
}

BOOST_AUTO_TEST_CASE(
    test_batch_radius_search_with_query_point_in_data_restricted)
{
  // Input data.
  KDTree tree(_data);
  const size_t& num_queries = _num_points_in_each_circle;
  vector<size_t> queries;
  for (size_t i = 0; i < num_queries; ++i)
    queries.push_back(i);
  double squared_search_radius = pow(2 * 2.0001, 2);

  // In-out data.
  vector<vector<int>> nn_indices;
  vector<vector<double>> nn_squared_distances;

  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances, max_num_nearest_neighbors);

  BOOST_CHECK_EQUAL(nn_indices.size(), num_queries);
  BOOST_CHECK_EQUAL(nn_squared_distances.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    BOOST_CHECK_EQUAL(nn_indices[i].size(), max_num_nearest_neighbors);
    BOOST_CHECK_EQUAL(nn_squared_distances[i].size(),
                      max_num_nearest_neighbors);

    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      BOOST_REQUIRE_LT(nn_indices[i][j],
                       static_cast<int>(_num_points_in_each_circle));
      BOOST_REQUIRE(nn_indices[i][j] != static_cast<int>(i));
      BOOST_REQUIRE_LT(nn_squared_distances[i][j], squared_search_radius);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
