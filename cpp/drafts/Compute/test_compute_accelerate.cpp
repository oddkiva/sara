// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Compute/Accelerate Backend"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/MultiArray.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include "Accelerate.hpp"


using Test_multiarray_types =
    boost::mpl::list<DO::Sara::MultiArray<float, 2, Eigen::ColMajor>,
                     DO::Sara::MultiArray<float, 2, Eigen::RowMajor>,
                     DO::Sara::MultiArray<double, 2, Eigen::ColMajor>,
                     DO::Sara::MultiArray<double, 2, Eigen::RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_compute_accelerate_gemm, MultiArray,
                              Test_multiarray_types)
{
  using namespace DO::Sara;

  auto A = MultiArray{3, 2};
  A.matrix() << 0, 1, //
                2, 3, //
                4, 5;

  auto B = MultiArray{2, 4};
  B.matrix() << 0, 1, 2, 3, //
                4, 5, 6, 7;

  auto C = MultiArray{3, 4};
  gemm(A, B, C);

  // Compare with Eigen backend.
  auto true_C = MultiArray{3, 4};
  true_C.matrix() = A.matrix() * B.matrix();
  // true_C = np.array([[ 4,  5,  6,  7],
  //                    [12, 17, 22, 27],
  //                    [20, 29, 38, 47]])

  const double rel_err =
      (C.matrix() - true_C.matrix()).norm() / true_C.matrix().norm();
  BOOST_CHECK_SMALL(rel_err, 1e-12);
}
