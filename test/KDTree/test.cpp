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

#include <gtest/gtest.h>

#include <DO/KDTree.hpp>


using namespace DO;
using namespace std;


class TestKDTree : public testing::Test
{
protected:
  MatrixXd data;
  size_t num_points;
  size_t num_points_in_circle;

  TestKDTree()
  {
    // We construct two sets in points. The first one lives in the 
    // zero-centered unit circle and the second in the zero-centered
    // circle with radius 10.
    num_points_in_circle = 5;
    num_points = 2*num_points_in_circle;
    data.resize(2, num_points);

    const size_t& N = num_points_in_circle;
    for (size_t i = 0; i < N; ++i)
    {
      double theta = i / (2*N*M_PI);
      data.col(i) << cos(theta), sin(theta);
    }

    for (size_t i = N; i < 2*N; ++i)
    {
      double theta = i / (2*N*M_PI);
      data.col(i) << 10*cos(theta), 10*sin(theta);
    }
  }
};


TEST_F(TestKDTree, test_simple_knn_search)
{
  KDTree tree(data);

  Vector2d query = Vector2d::Zero();
  size_t num_nearest_neighbors = num_points_in_circle;

  vector<int> indices;
  vector<double> squared_distances;

  tree.knn_search(query, num_nearest_neighbors, indices,
                  squared_distances);

  EXPECT_EQ(indices.size(), num_nearest_neighbors);
  for (size_t j = 0; j < indices.size(); ++j)
  {
    EXPECT_LE(indices[j], num_nearest_neighbors);
    EXPECT_NEAR(squared_distances[j], 1., 1e-10);
  }
}

TEST_F(TestKDTree, test_simple_radius_search)
{
  // Input data.
  KDTree tree(data);
  Vector2d query = Vector2d::Zero();
  size_t num_nearest_neighbors = num_points_in_circle;
  double squared_search_radius = 1.0001;

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;


  // First use case: we want to examine the squared distances.
  num_found_neighbors = tree.radius_search(
    query,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  EXPECT_EQ(nn_indices.size(), num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, num_nearest_neighbors);
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    EXPECT_LE(nn_indices[j], num_nearest_neighbors);
    EXPECT_NEAR(nn_squared_distances[j], 1., 1e-10);
  }


  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  num_found_neighbors = tree.radius_search(query,
                                           squared_search_radius,
                                           nn_indices,
                                           nn_squared_distances,
                                           max_num_nearest_neighbors);

  EXPECT_EQ(nn_indices.size(), max_num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, max_num_nearest_neighbors);
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    EXPECT_LE(nn_indices[j], num_nearest_neighbors);
    EXPECT_NEAR(nn_squared_distances[j], 1., 1e-10);
  }
}


TEST_F(TestKDTree, test_simple_knn_search_with_query_point_in_data)
{
  KDTree tree(data);

  size_t query_index = 0;
  size_t num_nearest_neighbors = num_points_in_circle-1;

  vector<int> indices;
  vector<double> squared_distances;

  tree.knn_search(query_index, num_nearest_neighbors, indices,
                  squared_distances);

  EXPECT_EQ(indices.size(), num_nearest_neighbors);
  for (size_t j = 0; j < indices.size(); ++j)
  {
    EXPECT_LE(indices[j], num_nearest_neighbors);
    EXPECT_LE(squared_distances[j], 2.);
  }
}

TEST_F(TestKDTree, test_simple_radius_search_with_query_point_in_data)
{
  // Input data.
  KDTree tree(data);
  size_t query = 0;
  size_t num_nearest_neighbors = num_points_in_circle-1;
  double squared_search_radius = 2.0001;

  // In-out data.
  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  // Output data.
  int num_found_neighbors;

  // First use case: we want to examine the squared distances.
  num_found_neighbors = tree.radius_search(
    query,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  EXPECT_EQ(nn_indices.size(), num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, num_nearest_neighbors);
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    EXPECT_LE(nn_indices[j], num_nearest_neighbors);
    EXPECT_LE(nn_squared_distances[j], 2.);
  }


  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  num_found_neighbors = tree.radius_search(query,
                                           squared_search_radius,
                                           nn_indices,
                                           nn_squared_distances,
                                           max_num_nearest_neighbors);

  EXPECT_EQ(nn_indices.size(), max_num_nearest_neighbors);
  EXPECT_EQ(num_found_neighbors, max_num_nearest_neighbors);
  for (size_t j = 0; j < nn_indices.size(); ++j)
  {
    EXPECT_LE(nn_indices[j], num_nearest_neighbors);
    EXPECT_LE(nn_squared_distances[j], 2.);
  }
}


TEST_F(TestKDTree, test_batch_knn_search)
{
  KDTree tree(data);
  const size_t& num_queries = num_points_in_circle;
  const size_t& num_nearest_neighbors = num_points_in_circle;
  MatrixXd queries(data.leftCols(num_queries));

  vector<vector<int> > indices;
  vector<vector<double> > squared_distances;

  tree.knn_search(queries, num_nearest_neighbors, indices, squared_distances);

  EXPECT_EQ(indices.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(indices[i].size(), num_queries);
    for (size_t j = 0; j < indices[i].size(); ++j)
    {
      EXPECT_LE(indices[i][j], num_queries);
      EXPECT_LE(squared_distances[i][j], 2.);
    }
  }
}

TEST_F(TestKDTree, test_batch_radius_search)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_circle;
  MatrixXd queries(data.leftCols(num_queries));
  double squared_search_radius = 2.0001;

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;


  // First use case: we want to examine the squared distances.
  tree.radius_search(
    queries,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  EXPECT_EQ(nn_indices.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), num_queries);
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LE(nn_indices[i][j], num_queries);
      EXPECT_LE(nn_squared_distances[i][j], 2.);
    }
  }


  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
                     nn_squared_distances, max_num_nearest_neighbors);

  EXPECT_EQ(nn_indices.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), max_num_nearest_neighbors);
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LE(nn_indices[i][j], num_queries);
      EXPECT_LE(nn_squared_distances[i][j], 2.);
    }
  }
}


TEST_F(TestKDTree, test_batch_knn_search_with_query_point_in_data)
{
  KDTree tree(data);
  const size_t num_queries = num_points_in_circle;
  const size_t num_nearest_neighbors = num_points_in_circle-1;

  vector<size_t> queries(num_queries);
  for (size_t i = 0; i != queries.size(); ++i)
    queries[i] = i;

  vector<vector<int> > indices;
  vector<vector<double> > squared_distances;

  tree.knn_search(queries, num_nearest_neighbors, indices, squared_distances);

  EXPECT_EQ(indices.size(), num_queries);
  for (size_t i = 0; i != indices.size(); ++i)
  {
    EXPECT_EQ(indices[i].size(), num_nearest_neighbors);
    for (size_t j = 0; j < indices[i].size(); ++j)
    {
      EXPECT_LE(indices[i][j], num_nearest_neighbors);
      EXPECT_LE(squared_distances[i][j], 2.);
    }
  }
}

TEST_F(TestKDTree, test_batch_radius_search_with_query_point_in_data)
{
  // Input data.
  KDTree tree(data);
  const size_t& num_queries = num_points_in_circle;
  vector<size_t> queries;
  for (size_t i = 0; i < num_queries; ++i)
    queries.push_back(i);
  double squared_search_radius = 2.0001;

  // In-out data.
  vector<vector<int> > nn_indices;
  vector<vector<double> > nn_squared_distances;

  // First use case: we want to examine the squared distances.
  tree.radius_search(
    queries,
    squared_search_radius,
    nn_indices,
    nn_squared_distances);

  EXPECT_EQ(nn_indices.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), num_queries-1);
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LE(nn_indices[i][j], num_queries);
      EXPECT_LE(nn_squared_distances[i][j], 2.);
    }
  }


  // Second use case: we want to limit the number of neighbors to return.
  size_t max_num_nearest_neighbors = 2;
  tree.radius_search(queries, squared_search_radius, nn_indices,
    nn_squared_distances, max_num_nearest_neighbors);

  EXPECT_EQ(nn_indices.size(), num_queries);
  for (size_t i = 0; i < num_queries; ++i)
  {
    EXPECT_EQ(nn_indices[i].size(), max_num_nearest_neighbors);
    for (size_t j = 0; j < nn_indices[i].size(); ++j)
    {
      EXPECT_LE(nn_indices[i][j], num_queries);
      EXPECT_LE(nn_squared_distances[i][j], 2.);
    }
  }
}


int main(int argc, char **argv) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}