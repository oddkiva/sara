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

// TODO TODO TODO

#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Gaussian Convolution FP16"

#include <boost/test/unit_test.hpp>

#include <type_traits>

#include <cuda_fp16.h>


BOOST_AUTO_TEST_CASE(test_convolve_fp16)
{
  const half a = __float2half(0.5f);
  const half b = __float2half(0.3f);
  // static_assert(std::is_same_v<decltype(a + b), half>);

  // std::cout << c << std::endl;
}
