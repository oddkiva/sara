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

#define BOOST_TEST_MODULE "Halide Backend/GaussianConvolution"

#include <DO/Sara/Core/Tensor.hpp>

#include <drafts/Halide/Helpers.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <boost/test/unit_test.hpp>

#include "GaussianConvolution.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


BOOST_AUTO_TEST_CASE(test_gaussian_convolution)
{
  auto src = Image<float>{33, 33};
  auto dst = Image<float>{33, 33};
  src.flat_array().fill(0);
  src(16, 16) = 1.f;

  SARA_DEBUG << "src" << std::endl;
  std::cout << src.matrix() << std::endl;

  auto src_buffer = halide::as_runtime_buffer_3d(src);
  auto dst_buffer = halide::as_runtime_buffer_3d(dst);

  src_buffer.set_host_dirty();
  GaussianConvolution(src_buffer, 3.f, 4, dst_buffer);
  dst_buffer.copy_to_host();

  SARA_DEBUG << "dst" << std::endl;
  std::cout << dst.matrix() << std::endl;
}
