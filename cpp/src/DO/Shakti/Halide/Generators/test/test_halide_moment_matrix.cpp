// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/Moment Matrix"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_moment_matrix_32f_cpu.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::Halide;


BOOST_AUTO_TEST_CASE(test_moment_matrix)
{
  auto fx = sara::Image<float>{8, 8};
  auto fy = sara::Image<float>{8, 8};

  for (auto y = 0; y < 8; ++y)
    for (auto x = 0; x < 8; ++x)
    {
      fx(x, y) = static_cast<float>(x + y);
      fy(x, y) = static_cast<float>(x - y);
    }

  auto mxx = sara::Image<float>{8, 8};
  auto myy = sara::Image<float>{8, 8};
  auto mxy = sara::Image<float>{8, 8};

  auto fx_buffer = halide::as_runtime_buffer_4d(fx);
  auto fy_buffer = halide::as_runtime_buffer_4d(fy);
  auto mxx_buffer = halide::as_runtime_buffer_4d(mxx);
  auto myy_buffer = halide::as_runtime_buffer_4d(myy);
  auto mxy_buffer = halide::as_runtime_buffer_4d(mxy);

  shakti_moment_matrix_32f_cpu(fx_buffer, fy_buffer, mxx_buffer, myy_buffer, mxy_buffer);

  auto true_mxx = sara::Image<float>{8, 8};
  auto true_myy = sara::Image<float>{8, 8};
  auto true_mxy = sara::Image<float>{8, 8};
  for (auto y = 0; y < 8; ++y)
  {
    for (auto x = 0; x < 8; ++x)
    {
      true_mxx(x, y) = fx(x, y) * fx(x, y);
      true_myy(x, y) = fy(x, y) * fy(x, y);
      true_mxy(x, y) = fx(x, y) * fy(x, y);
    }
  }

  BOOST_CHECK_SMALL(
      (true_mxx.matrix() - mxx.matrix()).lpNorm<Eigen::Infinity>(), 1e-12f);
  BOOST_CHECK_SMALL(
      (true_myy.matrix() - myy.matrix()).lpNorm<Eigen::Infinity>(), 1e-12f);
  BOOST_CHECK_SMALL(
      (true_mxy.matrix() - mxy.matrix()).lpNorm<Eigen::Infinity>(), 1e-12f);
}
