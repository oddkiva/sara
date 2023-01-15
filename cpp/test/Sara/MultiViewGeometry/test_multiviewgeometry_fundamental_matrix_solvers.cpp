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

#define BOOST_TEST_MODULE "MultiViewGeometry/Eight Point Algorithm"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenFormatInterop.hpp>

#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/FundamentalMatrixSolvers.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>

#include <boost/test/unit_test.hpp>

#include <Eigen/Eigen>

#include <iostream>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestFundamentalMinimalSolver)

BOOST_AUTO_TEST_CASE(test_eight_point_algorithm)
{
  // Check this so that it can be serialized with HDF5.
  static_assert(sizeof(FundamentalMatrix{}) == sizeof(Eigen::Matrix3d{}));

  auto left = Eigen::Matrix<double, 3, 8>{};
  auto right = Eigen::Matrix<double, 3, 8>{};

  // clang-format off
  left <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  right <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;
  // clang-format on

  // Fundamental matrix computation.
  const auto [F] = EightPointAlgorithm{}(left, right);

  // Check the residual errors.
  RowVectorXd errors(8);
  for (int i = 0; i < 8; ++i)
  {
    errors[i] = std::abs(right.col(i).transpose() * F.matrix() * left.col(i));
    BOOST_CHECK_SMALL(errors[i], 1e-3);
  }

  // Also check the batched residual computation as well.
  const auto batched_errors = EpipolarDistance{F}(left, right);
  BOOST_CHECK_LE(batched_errors.norm(), 1e-3);

  // Check that the batched distance computation is consistent for the unbatched
  // version.
  BOOST_CHECK_SMALL((batched_errors - errors).norm() / errors.norm(), 1e-12);

  SARA_DEBUG << "Individual errors = " << errors << std::endl;
  SARA_DEBUG << "Batched errors = " << batched_errors << std::endl;
  SARA_DEBUG << "Inliers = " << (batched_errors.array() < 1e-4) << std::endl;
  SARA_DEBUG << "Inlier count = " << (batched_errors.array() < 1e-4).count()
             << std::endl;

  // Is rank 2?
  BOOST_CHECK(F.rank_two_predicate());

  const auto [el, er] = F.extract_epipoles();

  for (int i = 0; i < 8; ++i)
  {
    const auto xl = left.col(i);
    const auto xr = right.col(i);

    const double err1 = er.transpose() * F.matrix() * xl;
    const double err2 = xr.transpose() * F.matrix() * el;

    BOOST_CHECK_SMALL(err1, 1e-12);
    BOOST_CHECK_SMALL(err2, 1e-12);

    // Check that the right epipole is lying in each right epipolar line.
    const double err1a = er.transpose() * F.right_epipolar_line(xl);

    // Check that the left epipole is lying in each left epipolar line.
    const double err2a = F.left_epipolar_line(xr).transpose() * el;

    BOOST_CHECK_SMALL(err1a, 1e-12);
    BOOST_CHECK_SMALL(err2a, 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_seven_point_algorithm)
{
  // Check this so that it can be serialized with HDF5.
  static_assert(sizeof(FundamentalMatrix{}) == sizeof(Eigen::Matrix3d{}));

  auto left = Eigen::Matrix<double, 2, 7>{};
  auto right = Eigen::Matrix<double, 2, 7>{};

  // clang-format off
  left <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057;

  right <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756;
  // clang-format on

  auto X = Eigen::Matrix<double, 4, 7>{};
  for (auto i = 0; i < 7; ++i)
    X.col(i) << left.col(i), right.col(i);

  // Check the nullspace extraction.
  const auto nullspace =
      SevenPointAlgorithmDoublePrecision::extract_nullspace(X);
  // Check the dimensions of the nullspace.
  BOOST_REQUIRE_EQUAL(nullspace.size(), 2u);
  // Check the residual.
  for (auto i = 0u; i < nullspace.size(); ++i)
  {
    const auto& F = nullspace[i];
    std::cout << fmt::format("Eigen F[{}] =\n {}\n", i, F);

    // Apply in batch.
    const auto dots_batched = (right.colwise().homogeneous().array() *
                               (F * left.colwise().homogeneous()).array())
                                  .colwise()
                                  .sum()
                                  .matrix();

    // Maximum residual value.
    const auto residual_max = dots_batched.lpNorm<Eigen::Infinity>();
    std::cout << fmt::format("residual max = {}\n", residual_max);
    BOOST_CHECK_SMALL(residual_max, 1e-12);
  }

  // Solve the determinant constraint.
  const auto Fs =
      SevenPointAlgorithmDoublePrecision::solve_determinant_constraint(
          nullspace);
  for (const auto& F : Fs)
  {
    std::cout << fmt::format("Candidate F =\n {}\n", F);

    // Apply in batch.
    const auto dots_batched = (right.colwise().homogeneous().array() *
                               (F * left.colwise().homogeneous()).array())
                                  .colwise()
                                  .sum()
                                  .matrix();

    // Maximum residual value.
    const auto residual_max = dots_batched.lpNorm<Eigen::Infinity>();
    BOOST_CHECK_SMALL(residual_max, 1e-12);

    // Try the eighth data point.
    const auto u = Eigen::Vector3d{0.253816, 0.675353, 1.};
    const auto v = Eigen::Vector3d{0.530029, 0.604387, 1.};

    SARA_CHECK(v.transpose() * F * u);
  }
}

BOOST_AUTO_TEST_SUITE_END()
