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

#include "shakti_gaussian_convolution.h"
#include "shakti_gaussian_convolution_v2.h"
#include "shakti_halide_gaussian_blur.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


auto gaussian_convolution_aot_and_stub(sara::Image<float>& src, sara::Image<float>& dst)
{
  const auto sigma = 100.f;
  const auto truncation_factor = 4;

  auto src_buffer = halide::as_runtime_buffer_3d(src);
  auto dst_buffer = halide::as_runtime_buffer_3d(dst);

  for (auto i = 0; i < 100; ++i)
  {
    sara::tic();
    src_buffer.set_host_dirty();
    shakti_gaussian_convolution(src_buffer, sigma, truncation_factor,
                                dst_buffer);
    dst_buffer.copy_to_host();
    sara::toc("[SEPARABLE GENERATOR STUB-BASED] Gaussian convolution");
  }

  sara::display(sara::color_rescale(dst));
}

auto gaussian_convolution_aot(sara::Image<float>& src, sara::Image<float>& dst)
{
  const auto sigma = 100.f;
  const auto truncation_factor = 4;

  dst.matrix().fill(0);

  auto src_buffer = halide::as_runtime_buffer(src);
  auto dst_buffer = halide::as_runtime_buffer(dst);

  for (auto i = 0; i < 100; ++i)
  {
    sara::tic();
    src_buffer.set_host_dirty();
    shakti_halide_gaussian_blur(src_buffer, sigma, truncation_factor,
                                dst_buffer);
    dst_buffer.copy_to_host();
    sara::toc("[GENERATOR] Gaussian convolution");
  }

  sara::display(sara::color_rescale(dst));
}

auto gaussian_convolution_aot_v2(sara::Image<float>& src,
                                 sara::Image<float>& dst)
{
  const auto sigma = 100.f;
  const auto truncation_factor = 4;

  dst.matrix().fill(0);

  auto src_tensor_view =
      tensor_view(src).reshape(Vector4i{1, 1, src.height(), src.width()});
  auto dst_tensor_view =
      tensor_view(dst).reshape(Vector4i{1, 1, dst.height(), dst.width()});

  auto src_buffer = halide::as_runtime_buffer(src_tensor_view);
  auto dst_buffer = halide::as_runtime_buffer(dst_tensor_view);

  for (auto i = 0; i < 100; ++i)
  {
    sara::tic();
    src_buffer.set_host_dirty();
    shakti_gaussian_convolution_v2(src_buffer, sigma, truncation_factor,
                                   dst_buffer);
    dst_buffer.copy_to_host();
    sara::toc("[GENERATOR V2] Gaussian convolution");
  }

  sara::display(sara::color_rescale(dst));
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

  gaussian_convolution_aot_and_stub(src, dst);
  gaussian_convolution_aot(src, dst);
  gaussian_convolution_aot_v2(src, dst);

  sara::get_key();

  return 0;
}
