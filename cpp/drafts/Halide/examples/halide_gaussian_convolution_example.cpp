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
#include "shakti_halide_gaussian_blur.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


GRAPHICS_MAIN()
{
  auto sz = 1025;
  auto src = Image<float>{sz, sz};
  auto dst = Image<float>{src.sizes()};
  src.flat_array().fill(0);
  src(src.sizes() / 2) = 1.f;

  SARA_CHECK(dst.sizes().transpose());
  sara::create_window(src.sizes());

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

  return 0;
}
