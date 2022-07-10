// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Adaptive Thresholding"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_adaptive_thresholding)
{
  auto src = sara::Image<float>{9, 9};
  // clang-format off
  src.matrix() <<
    0, 0.1f, 0, 0, 0, 0, 0.0f, 0.3f, 0,
    0, 0.1f, 0, 0, 0, 0, 0.0f, 0.1f, 0,
    0, 0.0f, 0, 0, 0, 0, 0.3f, 0.2f, 0,
    0, 0.0f, 0, 0, 0, 0, 0.1f, 0.0f, 0,
    0, 0.0f, 0, 0, 0, 0, 0.2f, 0.0f, 0,
    1, 1.0f, 1, 1, 0, 0, 0.1f, 0.0f, 0,
    1, 0.9f, 1, 1, 0, 0, 0.1f, 0.0f, 0,
    1, 0.8f, 1, 1, 0, 0, 0.1f, 0.0f, 0,
    1, 0.7f, 1, 1, 0, 0, 0.1f, 0.0f, 0;
  // clang-format on

  static constexpr auto sigma = 9.f;
  static constexpr auto gauss_truncate = 4.f;
  static constexpr auto tolerance_parameter = 0.f;

  auto binary_mask = sara::Image<std::uint8_t>{src.sizes()};
  sara::gaussian_adaptive_threshold(src, sigma, gauss_truncate, tolerance_parameter, binary_mask);

  // Keep printing the result so that later on we can still understand the data
  // when we want to perfect the unit test.
  std::cout << binary_mask.matrix().cast<int>() << std::endl;

  auto corner_mask = sara::Image<std::uint8_t>{src.sizes()};
  static constexpr auto v = 255u;
  // clang-format off
  corner_mask.matrix() <<
    0, 0, 0, 0, 0, 0, 0, v, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, v, v, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, v, 0, 0,
    v, v, v, v, 0, 0, 0, 0, 0,
    v, v, v, v, 0, 0, 0, 0, 0,
    v, v, v, v, 0, 0, 0, 0, 0,
    v, v, v, v, 0, 0, 0, 0, 0;
  // clang-format on

  // Flimsy check but it is better than nothing.
  BOOST_CHECK(binary_mask.matrix() == corner_mask.matrix());
}
