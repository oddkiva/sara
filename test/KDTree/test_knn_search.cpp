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

template <typename T>
inline set<T> to_set(const vector<T>& v)
{
  return set<T>(v.begin(), v.end());
}

// Define a macro that does something 'self.assertItemsEqual' in Python.
#define EXPECT_ITEMS_EQ(vector1, vector2) \
EXPECT_EQ(to_set(vector1), to_set(vector2))


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

  vector<int> nn_indices;
  vector<double> nn_squared_distances;

  tree.knn_search(query, num_nearest_neighbors, nn_indices,
                  nn_squared_distances);

  // Check equality of items.
  EXPECT_ITEMS_EQ(nn_indices, range(num_points_in_circle));

  // Check the squared distances.
  for (size_t j = 0; j < nn_indices.size(); ++j)
    EXPECT_NEAR(nn_squared_distances[j], 1., 1e-10);
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

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
