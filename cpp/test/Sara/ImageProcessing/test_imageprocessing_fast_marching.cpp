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

#define BOOST_TEST_MODULE "ImageProcessing/Level Sets/Fast Marching Method"

#include <DO/Sara/ImageProcessing/LevelSets/FastMarching.hpp>

#include "../AssertHelpers.hpp"

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_min_coeff_in_fast_marching)
{
  constexpr auto N = 3;
  auto us = Eigen::Matrix<float, N, 1>{};
  us << 0, 1, 2;

  // Implementation details check.
  auto umins = Eigen::Matrix<float, N - 1, N>{};
  for (auto j = 0; j < N; ++j)
  {
    if (j == 0)
      umins.col(j) << us.segment(1, N - 1);
    else if (j == N - 1)
      umins.col(j) << us.head(N - 1);
    else
      umins.col(j) << us.head(j), us.segment(j + 1, N - j - 1);
  }

  auto umins_true = Eigen::Matrix<float, N - 1, N>{};
  umins_true <<
    1, 0, 0,
    2, 2, 1;

  BOOST_CHECK(umins == umins_true);

  // Check the helper function.
  const auto min_coeff = sara::FastMarching<float, 3>::find_min_coefficient(us);
  BOOST_CHECK_EQUAL(min_coeff, 0.f);
}

BOOST_AUTO_TEST_CASE(test_fast_marching_2d)
{
  auto gradient = sara::Image<float>(20, 10);
  auto fm = sara::FastMarching{gradient};
}

BOOST_AUTO_TEST_CASE(test_fast_marching_3d)
{
  namespace sara = DO::Sara;
  auto gradient = sara::Image<float, 3>(20, 20, 20);

  auto fm = sara::FastMarching{gradient};

  // Enumerate by hand.
  const auto deltas_true = std::array{
    //               x   y   z
    Eigen::Vector3i{-1, -1, -1},
    Eigen::Vector3i{ 0, -1, -1},
    Eigen::Vector3i{ 1, -1, -1},
    Eigen::Vector3i{-1,  0, -1},
    Eigen::Vector3i{ 0,  0, -1},
    Eigen::Vector3i{ 1,  0, -1},
    Eigen::Vector3i{-1,  1, -1},
    Eigen::Vector3i{ 0,  1, -1},
    Eigen::Vector3i{ 1,  1, -1},
    Eigen::Vector3i{-1, -1,  0},
    Eigen::Vector3i{ 0, -1,  0},
    Eigen::Vector3i{ 1, -1,  0},
    Eigen::Vector3i{-1,  0,  0},
    Eigen::Vector3i{ 1,  0,  0},
    Eigen::Vector3i{-1,  1,  0},
    Eigen::Vector3i{ 0,  1,  0},
    Eigen::Vector3i{ 1,  1,  0},
    Eigen::Vector3i{-1, -1,  1},
    Eigen::Vector3i{ 0, -1,  1},
    Eigen::Vector3i{ 1, -1,  1},
    Eigen::Vector3i{-1,  0,  1},
    Eigen::Vector3i{ 0,  0,  1},
    Eigen::Vector3i{ 1,  0,  1},
    Eigen::Vector3i{-1,  1,  1},
    Eigen::Vector3i{ 0,  1,  1},
    Eigen::Vector3i{ 1,  1,  1}
  };

  BOOST_CHECK_EQUAL(fm._deltas.size(), 26);
  BOOST_CHECK(fm._deltas == deltas_true);
}
