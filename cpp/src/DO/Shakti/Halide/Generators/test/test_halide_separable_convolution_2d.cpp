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
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_separable_convolution_2d_cpu.h"


using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gaussian_blur)
{
  namespace sara = DO::Sara;
  using DO::Sara::square;

  // Create the gaussian kernel.
  const auto truncation_factor = 1;
  const auto sigma = 1.f;
  auto kernel_size = static_cast<int>(2 * sigma * truncation_factor + 1);
  if (kernel_size % 2 == 0)
    kernel_size += 1;

  const auto center = kernel_size / 2;
  auto kernel = std::vector<float>(kernel_size);
  for (int i = 0; i < kernel_size; ++i)
  {
    auto x = float(i - center);
    kernel[i] = exp(-square(x) / (2 * square(sigma)));
  }
  // 2. Calculate the normalizing factor.
  const auto sum_inverse =
      1 / std::accumulate(kernel.begin(), kernel.end(), float{});
  // 3. Rescale the Gaussian kernel.
  std::for_each(kernel.begin(), kernel.end(),
                [sum_inverse](auto& v) { v *= sum_inverse; });


  // Convolve with Dirac.
  const auto n = 3;
  auto src = sara::Image<float>{n, n};
  src.flat_array().fill(0.f);
  src(n / 2, n / 2) = 1.f;

  auto dst = sara::Image<float>{n, n};

  SARA_CHECK(kernel_size);
  SARA_CHECK(-center);
  SARA_DEBUG << "src =\n" << src.matrix() << std::endl;
  SARA_DEBUG << "kernel =\n";
  for (auto i = 0u; i < kernel.size(); ++i)
    std::cout << kernel[i] << " ";
  std::cout << std::endl;

  auto src_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(src);
  auto kernel_buffer = DO::Shakti::Halide::as_runtime_buffer(kernel);
  kernel_buffer.dim(0).set_min(-1);
  auto dst_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(dst);
  shakti_separable_convolution_2d_cpu(src_buffer, kernel_buffer, kernel_size, -center, dst_buffer);

  SARA_DEBUG << "dst =\n" << dst.matrix() << std::endl;

  // Input<Func> input{"input", Float(32), 3};
  // Input<Func> kernel{"kernel", Float(32), 1};
  // Input<int32_t> kernel_size{"kernel_size"};
  // Input<int32_t> kernel_shift{"kernel_shift"};
}

// BOOST_FIXTURE_TEST_CASE(test_gaussian, TestFilters)
// {
//   // Convolve with Dirac.
//   const auto n = _src_image.sizes()[0];
//   _src_image.flat_array().fill(0.f);
//   _src_image(n / 2, n / 2) = 1.f;
//
//   MatrixXf true_matrix(3, 3);
//   true_matrix <<
//     exp(-1.0f), exp(-0.5f), exp(-1.0f),
//     exp(-0.5f), exp(-0.0f), exp(-0.5f),
//     exp(-1.0f), exp(-0.5f), exp(-1.f);
//   true_matrix /= true_matrix.sum();
//
//   auto dst_image = Image<float>{ _src_image.sizes() };
//
//   auto apply_gaussian_filter = shakti::GaussianFilter{ 1.f, 1 };
//   apply_gaussian_filter(
//     dst_image.data(), _src_image.data(), _src_image.sizes().data());
//   BOOST_CHECK_SMALL((true_matrix - dst_image.matrix()).norm(), 1e-5f);
// }
