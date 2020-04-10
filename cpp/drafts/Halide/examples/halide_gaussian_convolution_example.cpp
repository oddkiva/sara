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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <drafts/Halide/Helpers.hpp>
#include <drafts/Halide/Utilities.hpp>

#include "GaussianConvolution.h"
#include "SeparableConvolution2d.h"
#include "shakti_halide_gaussian_blur.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


auto test0(sara::Image<float>& src, sara::Image<float>& dst)
{
  // Define the unnormalized gaussian function.
  auto x = halide::Var{"x"};
  const auto sigma = 20.f;
  const auto truncation_factor = 3;
  SARA_CHECK(truncation_factor);

  // Compute the size of the Gaussian kernel.
  auto kernel_size = static_cast<int>(2 * truncation_factor * sigma + 1);
  // Make sure the Gaussian kernel is at least of size 3 and is of odd size.
  kernel_size = std::max(3, kernel_size);
  if (kernel_size % 2 == 0)
    ++kernel_size;
  const auto kernel_shift = -kernel_size / 2;

  auto gaussian_unnormalized = halide::Func{"gaussian_unnormalized"};
  gaussian_unnormalized(x) = halide::exp(-(x * x) / (2 * sigma * sigma));

  // Define the summation variable `k` defined on a summation domain.
  auto k = halide::RDom(kernel_shift, kernel_size);
  // Calculate the normalization factor by summing with the summation
  // variable.
  auto normalization_factor = halide::sum(gaussian_unnormalized(k));
  auto gaussian = halide::Func{"gaussian"};
  gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

  // Realize the gaussian kernel.
  auto kernel_vec = std::vector<float>(kernel_size);
  auto kernel_buffer = halide::Buffer<float>{kernel_vec.data(), kernel_size};
  kernel_buffer.set_min(kernel_shift);

  {
    for (auto i = 0; i < kernel_size; ++i)
    {
      const auto x = float(i + kernel_shift);
      kernel_vec[i] = std::exp(-0.5f * std::pow(x / sigma, 2));
    }
    auto kernel_sum = std::accumulate(kernel_vec.begin(), kernel_vec.end(), 0.f);
    SARA_CHECK(kernel_sum);

    for (auto i = 0; i < kernel_size; ++i)
      kernel_vec[i] /= kernel_sum;

    kernel_sum = std::accumulate(kernel_vec.begin(), kernel_vec.end(), 0.f);
    SARA_CHECK(kernel_sum);
  }

  // const auto a = kernel_buffer.dim(0).min();
  // const auto b = kernel_buffer.dim(0).max();

  // SARA_DEBUG << "Manual gaussian kernel" << std::endl;
  // for (auto k = a; k <= b; ++k)
  //   SARA_DEBUG << sara::format("g[%d] = %f", k, kernel_buffer(k)) << std::endl;

  // gaussian.realize(kernel_buffer);
  // SARA_DEBUG << "Check realized gaussian kernel" << std::endl;
  // for (auto k = a; k <= b; ++k)
  //   SARA_DEBUG << sara::format("g[%d] = %f", k, kernel_buffer(k)) << std::endl;

  auto src_buffer = halide::as_buffer_3d(src);
  auto dst_buffer = halide::as_buffer_3d(dst);

  {
    using namespace halide;

    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}
    auto input_padded = BoundaryConditions::repeat_edge(src_buffer);

    auto k = halide::RDom{kernel_shift, kernel_size};

    // 1st pass: transpose and convolve the columns.
    auto input_t = Func{"input_transposed"};
    input_t(x, y, c) = input_padded(y, x, c);
    input_t.compute_root();

    auto conv_y_t = Func{"conv_y_transposed"};
    conv_y_t(x, y, c) = sum(input_t(x + k, y, c) * kernel_buffer(k));
    conv_y_t.compute_root();

    // 2nd pass: transpose and convolve the rows.
    auto conv_y = Func{"conv_y"};
    conv_y(x, y, c) = conv_y_t(y, x, c);
    conv_y.compute_root();

    auto conv_x = Func{"conv_x"};
    conv_x(x, y, c) = sum(conv_y(x + k, y, c) * kernel_buffer(k));

    conv_x.realize(dst_buffer);
  }

  // SARA_DEBUG << "Running separable convolution 2D" << std::endl;
  // SeparableConvolution2d(padded_src, kernel_buffer, kernel_size,
  // kernel_shift,
  //                        dst_buffer);
  sara::display(sara::color_rescale(dst));
  sara::get_key();
}

auto test1(sara::Image<float>& src, sara::Image<float>& dst)
{
  sara::display(src);
  sara::get_key();

  auto src_buffer = halide::as_runtime_buffer_3d(src);
  auto dst_buffer = halide::as_runtime_buffer_3d(dst);

  for (auto i = 0; i < 10; ++i)
  {
    sara::tic();
    src_buffer.set_host_dirty();
    GaussianConvolution(src_buffer, 100.f, 4, dst_buffer);
    dst_buffer.copy_to_host();
    sara::toc("[GENERATOR STUB] Gaussian convolution");
  }

  sara::apply_gaussian_filter(src, dst, 100.f, 4);
  sara::display(sara::color_rescale(dst));
  sara::get_key();
}

auto test2(sara::Image<float>& src, sara::Image<float>& dst)
{
  sara::display(src);
  sara::get_key();

  dst.matrix().fill(0);

  auto src_buffer = halide::as_runtime_buffer(src);
  auto dst_buffer = halide::as_runtime_buffer(dst);

  for (auto i = 0; i < 10; ++i)
  {
    sara::tic();
    src_buffer.set_host_dirty();
    shakti_halide_gaussian_blur(src_buffer, 100.f, 4, dst_buffer);
    // sara::apply_gaussian_filter(src, dst, 50.f, 4);
    dst_buffer.copy_to_host();
    sara::toc("[GENERATOR] Gaussian convolution");
  }

  sara::display(sara::color_rescale(dst));
  sara::get_key();
}

GRAPHICS_MAIN()
{
  auto sz = 1025;
  auto src = Image<float>{sz, sz};
  auto dst = Image<float>{src.sizes()};
  src.flat_array().fill(0);
  src(src.sizes() / 2) = 1.f;

  SARA_CHECK(dst.sizes().transpose());
  sara::create_window(src.sizes());

  test2(src, dst);


  return 0;
}
