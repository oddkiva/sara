// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Clustering/Density Based Clustering 1D"

#include <DO/Sara/Clustering/Clustering1D.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_density_based_clustering_1d)
{
  auto f = std::vector<float>{
  //      2           6        9     11       14       17        20    22
    0, 0, 3, 4, 4, 5, 7, 6, 3, 2, 5, 10, 9, 4, 3, 0, 0, 2, 4, 8, 10, 7, 4, 0};

  const auto clusters = compute_density_based_clustering_1d(f.data(), f.size());
  const auto clusters_expected = std::vector<Cluster1D>{
    {6, {2, 9 + 1}},
    {11, {9, 14 + 1}},
    {20, {17, 22 + 1}}
  };

  BOOST_CHECK(clusters == clusters_expected);

  for (auto& c: clusters)
  {
    SARA_DEBUG << "cluster = " << c.peak  //
               << " [" << c.interval.a << ", " << c.interval.b << "]"
               << std::endl;
  }
}
