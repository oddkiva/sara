// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Vanishing Point"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/RANSAC.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>

#include <DO/Sara/MultiViewGeometry/SingleView/VanishingPoint.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_vp_detection)
{
  const auto vp = Eigen::Vector3f(500, 500, 1);
  auto lines = Tensor_<float, 2>{6, 3};
  auto lines_as_matrix = lines.matrix();

  constexpr auto radius = 100.f;
  constexpr auto delta = float(10 / M_PI);

  for (auto i = 0u; i < 6; ++i)
  {
    const auto angle = i * delta;
    const auto v = Eigen::Vector3f(cos(angle), sin(angle), 0);
    const Eigen::Vector3f p2 = vp + radius * v;

    auto line = Projective::line(vp, p2);
    line /= line.head(2).norm();

    lines_as_matrix.row(i) = line.transpose();
  }

  SARA_DEBUG << "lines_as_matrix =\n" << lines_as_matrix << std::endl;

  const auto& [vp_estimated, inliers, best_line_pair] =
      find_dominant_vanishing_point(lines, 5.f, 100u);
  BOOST_CHECK_LE((vp.hnormalized() - vp_estimated.hnormalized()).norm(), 5e-3f);
}
