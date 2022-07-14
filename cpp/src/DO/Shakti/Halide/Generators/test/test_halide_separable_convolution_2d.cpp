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

#define BOOST_TEST_MODULE "Halide Backend"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_separable_convolution_2d_cpu.h"


using namespace Halide;


BOOST_AUTO_TEST_CASE(test_separable_convolution_2d_with_gaussian_kernel)
{
  namespace sara = DO::Sara;

  static constexpr auto n = 3;

  // Convolve with Dirac.
  auto src = sara::Image<float>{n, n};
  src.flat_array().fill(0.f);
  src(n / 2, n / 2) = 1.f;

  // Check with a non-trivial kernel.
  //
  // This will generate a 3x3 kernel.
  const auto kernel = sara::make_gaussian_kernel(1.f, 1.f);

  // Check the separable convolution with a mini array.
  //
  // This is to make sure that the boundary conditions are taken care correctly
  // in the Halide implementation.
  auto dst = sara::Image<float>{n, n};
  auto src_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(src);
  auto kernel_buffer = DO::Shakti::Halide::as_runtime_buffer(kernel);
  auto dst_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(dst);
  const auto kernel_size = static_cast<std::int32_t>(kernel.size());
  shakti_separable_convolution_2d_cpu(src_buffer, kernel_buffer, kernel_size,
                                      -kernel_size / 2, dst_buffer);

  // Write down the expected convolution.
  auto dst_true = Eigen::Matrix3f{};
  dst_true <<
    exp(-1.0f), exp(-0.5f), exp(-1.0f),
    exp(-0.5f), exp(-0.0f), exp(-0.5f),
    exp(-1.0f), exp(-0.5f), exp(-1.f);
  dst_true /= dst_true.sum();

  BOOST_CHECK_SMALL((dst.matrix() - dst_true).norm(), 1e-5f);
}
