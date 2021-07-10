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

#define BOOST_TEST_MODULE "Halide Backend/Helpers"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/TensorDebug.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_halide_cast_to_float.h"

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


BOOST_AUTO_TEST_CASE(test_cast_from_uint8_t_to_float)
{
  {
    auto src = Image<Rgb8>{8, 8};
    auto dst = Image<Rgb32f>{8, 8};
    src.flat_array().fill(Rgb8::Ones() * 255);

    auto src_buffer = halide::as_interleaved_runtime_buffer(src);
    auto dst_buffer = halide::as_interleaved_runtime_buffer(dst);

    src_buffer.set_host_dirty();
    shakti_halide_cast_to_float(src_buffer, dst_buffer);
    dst_buffer.copy_to_host();

    BOOST_CHECK_EQUAL(dst(0, 0), Rgb32f::Ones().eval());
  }

  {
    auto src = Tensor_<std::uint8_t, 3>{3, 8, 8};
    auto dst = Tensor_<float, 3>{3, 8, 8};

    src.flat_array().fill(255);

    auto src_buffer = halide::as_runtime_buffer(src);
    auto dst_buffer = halide::as_runtime_buffer(dst);

    src_buffer.set_host_dirty();
    shakti_halide_cast_to_float(src_buffer, dst_buffer);
    dst_buffer.copy_to_host();

    for (int i = 0; i < 3; ++i)
      BOOST_CHECK_EQUAL(dst[i].matrix(), MatrixXf::Ones(8, 8));
  }
}
