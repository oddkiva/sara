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
  size_t N;

  TestKDTree()
  {
    // Generate grid of points.
    N = 10;
    data.resize(2,N*N);
    for (size_t i = 0; i != N ; ++i)
      for (size_t j = 0; j != N ; ++j)
        data.col(i+j*N) = Point2d(i, j);
  }
};


TEST_F(TestKDTree, test_batch_knn_search)
{
  // Batch query search.
  printStage("Batch query search");
  MatrixXd queries(data);
  queries.array() += 0.5;
  KDTree kdTree(data);
  vector<vector<int> > indices;
  vector<vector<double> > sqDists;
  kdTree.knn_search(queries, 10, indices, sqDists);

  for (int i = 0; i != data.cols(); ++i)
  {
    cout << "p[" << i << "] = " << queries.col(i).transpose() << endl;
    for (size_t j = 0; j != indices[i].size(); ++j)
    {
      size_t indj = indices[i][j];
      cout << "knn["<< j << "]\tp[" << indj << "] = " << data.col(indj).transpose() << endl;
    }
  }
}


TEST_F(TestKDTree, test_batch_knn_search_with_query_in_data)
{
  vector<size_t> queries(data.cols());
  for (int i = 0; i != data.cols(); ++i)
    queries[i] = i;
  KDTree kdTree(data);
  vector<vector<int> > indices;
  vector<vector<double> > sqDists;
  kdTree.knn_search(queries, 10, indices, sqDists);

  // Check visually.
  for (int i = 0; i != data.cols(); ++i)
  {
    cout << "p[" << i << "] = " << data.col(queries[i]).transpose() << endl;
    for (size_t j = 0; j != indices[i].size(); ++j)
    {
      size_t indj = indices[i][j];
      cout << "knn["<< j << "]\tp[" << indj << "] = " << data.col(indj).transpose() << endl;
    }
  }
}


int main(int argc, char **argv) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}