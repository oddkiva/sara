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

#include <DO/Sara/ImageProcessing/Otsu.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_adaptive_thresholding)
{
  auto src = sara::Image<float>{9, 9};
  // clang-format off
  src.matrix() <<
    0, 0.1, 0, 0, 0, 0.1, 0.1, 0.2, 0.2,
    0, 0.1, 0, 0, 0, 0.1, 0.0, 0.1, 0.2,
    0, 0.0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2,
    0, 0.0, 0, 0, 0, 0.1, 0.1, 0.0, 0.2,
    0, 0.0, 0, 0, 0, 0.1, 0.2, 0.0, 0.2,
    1, 1.0, 1, 1, 0, 0.1, 0.1, 0.0, 0.2,
    1, 0.5, 1, 1, 0, 0.1, 0.1, 0.0, 0.2,
    1, 0.5, 1, 1, 0, 0.1, 0.1, 0.0, 0.2,
    1, 0.5, 1, 1, 0, 0.1, 0.1, 0.0, 0.2;
  // clang-format on

  // Cheat with smoothing
  const auto mask = sara::otsu_adaptive_binarization(src);

  // Keep printing the result so that later on we can still understand the data
  // when we want to perfect the unit test.
  std::cout << mask.matrix().cast<int>() << std::endl;

  auto expected_mask = sara::Image<std::uint8_t>{src.sizes()};
  // clang-format off
  expected_mask.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0;
  // clang-format on

  BOOST_CHECK(mask.matrix() == expected_mask.matrix());
}
