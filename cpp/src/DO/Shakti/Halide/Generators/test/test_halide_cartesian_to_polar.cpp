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

#define BOOST_TEST_MODULE "Halide Backend/Cartesian to Polar Coordinates"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_cartesian_to_polar_32f_cpu.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::Halide;


BOOST_AUTO_TEST_CASE(test_cartesian_to_polar_coordinates)
{
  auto fx = sara::Image<float>{8, 8};
  auto fy = sara::Image<float>{8, 8};

  for (auto y = 0; y < 8; ++y)
    for (auto x = 0; x < 8; ++x)
    {
      fx(x, y) = static_cast<float>(x + y);
      fy(x, y) = static_cast<float>(x - y);
    }

  auto mag = sara::Image<float>{8, 8};
  auto ori = sara::Image<float>{8, 8};

  auto fx_buffer = halide::as_runtime_buffer_4d(fx);
  auto fy_buffer = halide::as_runtime_buffer_4d(fy);
  auto mag_buffer = halide::as_runtime_buffer_4d(mag);
  auto ori_buffer = halide::as_runtime_buffer_4d(ori);

  shakti_cartesian_to_polar_32f_cpu(fx_buffer, fy_buffer, mag_buffer, ori_buffer);

  auto true_mag = sara::Image<float>{8, 8};
  auto true_ori = sara::Image<float>{8, 8};
  for (auto y = 0; y < 8; ++y)
  {
    for (auto x = 0; x < 8; ++x)
    {
      true_mag(x, y) = std::hypot(fx(x, y), fy(x, y));
      true_ori(x, y) = std::atan2(fy(x, y), fx(x, y));
    }
  }

  BOOST_CHECK_SMALL(
      (true_mag.matrix() - mag.matrix()).lpNorm<Eigen::Infinity>(), 1e-12f);
  BOOST_CHECK_SMALL(
      (true_ori.matrix() - ori.matrix()).lpNorm<Eigen::Infinity>(), 5e-4f);
}
