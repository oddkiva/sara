// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "RANSAC/Utility"

#include <DO/Sara/RANSAC/Utility.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestUtility)

BOOST_AUTO_TEST_CASE(test_index_to_point_pairs_from_epipolar_geometry)
{
  // Data dimension
  static constexpr auto data_dimension = 3;

  // Take data from epipolar geometry.
  auto left = sara::Tensor_<double, 2>{8, data_dimension};
  auto right = sara::Tensor_<double, 2>{8, data_dimension};

  // These are  the point correspondences.
  // clang-format off
  left.colmajor_view().matrix() <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  right.colmajor_view().matrix() <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;
  // clang-format on

  // Make simple matches
  auto M = sara::Tensor_<int, 2>{8, 2};
  for (auto i = 0; i < 8; ++i)
  {
    M(i, 0) = i;
    M(i, 1) = i;
  }

  static constexpr auto num_subsets = 3;
  static constexpr auto num_points_per_subset = 5;
  auto I = sara::Tensor_<int, 2>{num_subsets, num_points_per_subset};
  // clang-format off
  I.matrix() <<
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 5,
    3, 0, 1, 7, 6;
  // clang-format on

  const auto X = sara::from_index_pairs_to_point_pairs(M, left, right);
  BOOST_CHECK_EQUAL(X.sizes(), Eigen::Vector3i(8, 2, data_dimension));
  for (auto i = 0; i < 8; ++i)
  {
    BOOST_CHECK(X[i][0].vector() == left[i].vector());
    BOOST_CHECK(X[i][1].vector() == right[i].vector());
  }

  const auto S = sara::from_index_to_point(I, X);
  BOOST_CHECK_EQUAL(S.size(0), num_subsets);
  BOOST_CHECK_EQUAL(S.size(1), num_points_per_subset);
  BOOST_CHECK_EQUAL(S.size(2), 2);
  BOOST_CHECK_EQUAL(S.size(3), data_dimension);

  static constexpr auto first = 0;
  static constexpr auto second = 1;
  // Unwrap the code to check manually the first row.
  BOOST_CHECK_EQUAL(S[0][0][first].vector(), left[0].vector());
  BOOST_CHECK_EQUAL(S[0][0][second].vector(), right[0].vector());

  BOOST_CHECK_EQUAL(S[0][1][first].vector(), left[1].vector());
  BOOST_CHECK_EQUAL(S[0][1][second].vector(), right[1].vector());

  BOOST_CHECK_EQUAL(S[0][2][first].vector(), left[2].vector());
  BOOST_CHECK_EQUAL(S[0][2][second].vector(), right[2].vector());

  BOOST_CHECK_EQUAL(S[0][3][first].vector(), left[3].vector());
  BOOST_CHECK_EQUAL(S[0][3][second].vector(), right[3].vector());

  BOOST_CHECK_EQUAL(S[0][4][first].vector(), left[4].vector());
  BOOST_CHECK_EQUAL(S[0][4][second].vector(), right[4].vector());

  // Check the last row, first column
  BOOST_CHECK_EQUAL(S[2][0][first].vector(), left[3].vector());
  BOOST_CHECK_EQUAL(S[2][0][second].vector(), right[3].vector());
}

BOOST_AUTO_TEST_SUITE_END()
