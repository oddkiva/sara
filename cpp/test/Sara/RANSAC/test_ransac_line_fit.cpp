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

#define BOOST_TEST_MODULE "RANSAC/Line Fitting"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/LineSolver.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_robust_line_fit)
{
  auto points = PointList<double, 2>{Tensor_<double, 2>{6, 3}};
  // clang-format off
  points.data.matrix() <<               0.00,                      0.00, 1,
                                        1.00,                      1.10, 1,
                                        3.40,                      3.46, 1,
                                        9.80,                     10.10, 1,
                     2 * std::cos(M_PI / 3.), -10 * std::sin(M_PI / 3.), 1,
                         std::cos(M_PI / 6.),       std::sin(M_PI / 6.), 1;
  // clang-format on

  auto line_solver = LineSolver2D<double>{};
  auto inlier_predicate = InlierPredicate<LinePointDistance2D<double>>{
      {}, 0.1  //
  };
  const auto& estimation = ransac(points,            //
                                  line_solver,       //
                                  inlier_predicate,  //
                                  10u);
  const auto& inliers = std::get<1>(estimation);
  BOOST_CHECK_EQUAL(inliers.flat_array().count(), 4);
}
