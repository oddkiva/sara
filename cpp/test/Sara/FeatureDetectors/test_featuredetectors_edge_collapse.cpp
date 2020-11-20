// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "FeatureDetectors/Edge Collapse"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>


using namespace DO::Sara;

BOOST_AUTO_TEST_CASE(test_edge_collapse)
{
  const auto p = std::vector<Eigen::Vector2d>{
    Eigen::Vector2d(0, 0),
    Eigen::Vector2d(0, 1),
    Eigen::Vector2d(0, 1.1),
    Eigen::Vector2d(0, 1.2),
    Eigen::Vector2d(0, 1.3),
    Eigen::Vector2d(0, 5.),
    Eigen::Vector2d(0, 6.),
    Eigen::Vector2d(0, 7.),
    Eigen::Vector2d(0, 7.2),
    Eigen::Vector2d(0, 7.22),
    Eigen::Vector2d(0, 7.5),
    Eigen::Vector2d(0, 10.)
  };

  auto gradient = Image<float>{12, 12};
  gradient.flat_array().fill(0);

  const auto p2 = collapse(p, gradient, 0.5, false);
  const auto p2_expected = std::vector<Eigen::Vector2d>{
    Eigen::Vector2d(0, 0),
    Eigen::Vector2d(0, 1),
    Eigen::Vector2d(0, 5.),
    Eigen::Vector2d(0, 6.),
    Eigen::Vector2d(0, 7.),
    Eigen::Vector2d(0, 10.)
  };

  BOOST_CHECK(p2 == p2_expected);

}
