// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! TODO: this is a messy set of unit tests. It maybe worth reinstating Google
//! mock in the library. We can compare easily vectors.

#include <set>

#include <gtest/gtest.h>

#include <DO/KDTree.hpp>

#include "../AssertHelpers.hpp"


using namespace DO;
using namespace std;


inline vector<int> range(int begin, int end)
{
  vector<int> _range(end-begin);
  for (int i = begin; i < end; ++i)
    _range[i-begin] = i;
  return _range;
}

inline vector<int> range(int end)
{
  return range(0, end);
}


class TestKDTree : public testing::Test
{
protected:
  MatrixXd data;
  size_t num_points;
  size_t num_points_in_each_circle;

  TestKDTree()
  {
    // We construct two sets in points. The first one lives in the 
    // zero-centered unit circle and the second in the zero-centered
    // circle with radius 10.
    num_points_in_each_circle = 30;
    num_points = 2*num_points_in_each_circle;
    data.resize(2, num_points);

    const size_t& N = num_points_in_each_circle;
    for (size_t i = 0; i < N; ++i)
    {
      double theta = i / (2*N*M_PI);
      data.col(i) << 2*cos(theta), 2*sin(theta);
    }

    for (size_t i = N; i < 2*N; ++i)
    {
      double theta = i / (2*N*M_PI);
      data.col(i) << 10*cos(theta), 10*sin(theta);
    }
  }
};


TEST_F(TestKDTree, test_simple_radius_search_default_use)
{
  // Input data.
  KDTree tree(data, 1, flann::SearchParams());
  Vector2d query = Vector2d::Zero();
  cout << query << endl;

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Regular use case.
  size_t num_nearest_neighbors = num_points_in_each_circle;
  double squared_search_radius = 4.1;
  int num_found_neighbors = tree.radius_search(query,
                                               squared_search_radius,
                                               nn_indices,
                                               nn_squared_distances);

  // Check the number of neighbors.
  EXPECT_EQ(nn_indices.size(), num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, num_nearest_neighbors);

  // Check the indices of the nearest neighbors.
  EXPECT_ITEMS_EQ(range(num_points_in_each_circle), nn_indices);

  // Check the squared distances.
  EXPECT_EQ(nn_squared_distances.size(), num_nearest_neighbors);
  for (size_t j = 0; j < nn_squared_distances.size(); ++j)
    EXPECT_NEAR(nn_squared_distances[j], 4., 1e-10);
}


TEST_F(TestKDTree, test_simple_radius_search_with_restricted_num_of_neighbors)
{
  // Input data.
  KDTree tree(data);
  Vector2d query = Vector2d::Zero();
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  int num_found_neighbors = tree.radius_search(query,
                                               squared_search_radius,
                                               nn_indices,
                                               nn_squared_distances,
                                               max_num_nearest_neighbors);

  // Check the number of nearest neighbors.
  EXPECT_EQ(nn_indices.size(), max_num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, max_num_nearest_neighbors);

  // Check the contents of the containers.
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    EXPECT_LT(nn_indices[j], num_points_in_each_circle);
    EXPECT_NEAR(nn_squared_distances[j], 4., 1e-10);
  }
}

TEST_F(TestKDTree,
       test_simple_radius_search_with_query_point_in_data_default)
{
  // Input data.
  KDTree tree(data);
  size_t query = 0;
  size_t num_nearest_neighbors = num_points_in_each_circle-1;
  // Squared diameter of the inner circle.
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;

  // Default use case.
  num_found_neighbors = tree.radius_search(query,
                                           squared_search_radius,
                                           nn_indices,
                                           nn_squared_distances);

  // Check the number of neighbors.
  EXPECT_EQ(nn_indices.size(), num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, num_nearest_neighbors);

  // Check the indices.
  EXPECT_ITEMS_EQ(nn_indices, range(1, num_points_in_each_circle));

  // Check the squared distances.
  for (size_t j = 0; j < nn_indices.size(); ++j)
    EXPECT_LE(nn_squared_distances[j], squared_search_radius);
}


TEST_F(TestKDTree,
       test_simple_radius_search_with_query_point_in_data_restricted)
{
  // Input data.
  KDTree tree(data);
  size_t query = 0;
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;
  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  num_found_neighbors = tree.radius_search(query,
                                           squared_search_radius,
                                           nn_indices,
                                           nn_squared_distances,
                                           max_num_nearest_neighbors);

  // Check the number of indices.
  EXPECT_EQ(nn_indices.size(), max_num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, max_num_nearest_neighbors);

  // Check the number of squared distances.
  EXPECT_EQ(nn_squared_distances.size(), max_num_nearest_neighbors);

  // Check the contents of the retrieval.
  for (size_t j = 0; j < nn_squared_distances.size(); ++j)
  {
    // Check the index value.
    EXPECT_NE(nn_indices[j], 0);
    EXPECT_LT(nn_indices[j], num_points_in_each_circle);

    // Check the squared distances.
    EXPECT_LE(nn_squared_distances[j], 2.);
  }
}

TEST_F(TestKDTree, test_batch_radius_search_default)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_each_circle;
  MatrixXd queries(data.leftCols(num_queries));
  double squared_search_radius = pow(1.0001, 2);

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;

  // Use case.
  tree.radius_search(
    queries,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  // Check the number of queries.
  EXPECT_EQ(nn_indices.size(), num_queries);
  EXPECT_EQ(nn_squared_distances.size(), num_queries);

  // Check the content of the retrieval.
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), num_points_in_each_circle);
    EXPECT_ITEMS_EQ(nn_indices[i], range(num_points_in_each_circle));

    EXPECT_EQ(nn_squared_distances[i].size(), num_points_in_each_circle);
    for (size_t j = 0; j < nn_squared_distances[i].size(); ++j)
      EXPECT_LE(nn_squared_distances[i][j], squared_search_radius);
  }
}

TEST_F(TestKDTree, test_batch_radius_search_restricted)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_each_circle;
  MatrixXd queries(data.leftCols(num_queries));
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;

  // Use case.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances, max_num_nearest_neighbors);

  // Check the number of queries.
  EXPECT_EQ(nn_indices.size(), num_queries);
  EXPECT_EQ(nn_squared_distances.size(), num_queries);
  
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), max_num_nearest_neighbors);
    EXPECT_EQ(nn_squared_distances[i].size(), max_num_nearest_neighbors);

    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LT(nn_indices[i][j], num_points_in_each_circle);
      EXPECT_LE(nn_squared_distances[i][j], squared_search_radius);
    }
  }
}

TEST_F(TestKDTree, test_batch_radius_search_with_query_point_in_data_default)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_each_circle;
  vector<size_t> queries;
  for (size_t i = 0; i < num_queries; ++i)
    queries.push_back(i);
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;

  // Default use case.
  tree.radius_search(
    queries,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  // Check the number of queries.
  EXPECT_EQ(nn_indices.size(), num_queries);
  EXPECT_EQ(nn_squared_distances.size(), num_queries);

  // Check the contents of the retrieval.
  for (size_t i = 0; i < num_queries; ++i)
  {
    vector<int> true_indices = range(num_points_in_each_circle);
    true_indices.erase(true_indices.begin() + i);

    // Check the indices.
    EXPECT_EQ(nn_indices[i].size(), num_points_in_each_circle-1);
    EXPECT_ITEMS_EQ(true_indices, nn_indices[i]);

    // Check the squared distances.
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
      EXPECT_LE(nn_squared_distances[i][j], squared_search_radius);
  }
}

TEST_F(TestKDTree, test_batch_radius_search_with_query_point_in_data_restricted)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_each_circle;
  vector<size_t> queries;
  for (size_t i = 0; i < num_queries; ++i)
    queries.push_back(i);
  double squared_search_radius = pow(2.0001, 2);

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;

  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
    nn_squared_distances, max_num_nearest_neighbors);

  EXPECT_EQ(nn_indices.size(), num_queries);
  EXPECT_EQ(nn_squared_distances.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), max_num_nearest_neighbors);
    EXPECT_EQ(nn_squared_distances[i].size(), max_num_nearest_neighbors);

    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LT(nn_indices[i][j], num_points_in_each_circle);
      EXPECT_NE(nn_indices[i][j], i);
      EXPECT_LE(nn_squared_distances[i][j], squared_search_radius);
    }
  }
}


int main(int argc, char **argv) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
