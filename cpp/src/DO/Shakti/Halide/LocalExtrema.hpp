// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_local_max_32f_gpu.h"
#include "shakti_local_scale_space_extremum_32f_gpu_v2.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  inline auto local_max(Sara::ImageView<float>& a,  //
                        Sara::ImageView<float>& b,  //
                        Sara::ImageView<float>& c,  //
                        Sara::ImageView<std::uint8_t>& out)
  {
    auto a_buffer = Shakti::Halide::as_runtime_buffer_4d(a);
    auto b_buffer = Shakti::Halide::as_runtime_buffer_4d(b);
    auto c_buffer = Shakti::Halide::as_runtime_buffer_4d(c);
    auto out_buffer = Shakti::Halide::as_runtime_buffer_4d(out);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();
    shakti_local_max_32f_gpu(a_buffer, b_buffer, c_buffer, out_buffer);
    out_buffer.copy_to_host();
  }

  inline auto local_scale_space_extrema(Sara::ImageView<float>& a,
                                        Sara::ImageView<float>& b,
                                        Sara::ImageView<float>& c,
                                        Sara::ImageView<std::int8_t>& out,
                                        float edge_ratio = 10.0f,
                                        float extremum_thres = 0.03f)
  {
    auto a_buffer = as_runtime_buffer(a);
    auto b_buffer = as_runtime_buffer(b);
    auto c_buffer = as_runtime_buffer(c);
    auto out_buffer = as_runtime_buffer(out);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();

    shakti_local_scale_space_extremum_32f_gpu_v2(
        a_buffer, b_buffer, c_buffer, edge_ratio, extremum_thres, out_buffer);

    out_buffer.copy_to_host();
  }

}}}  // namespace DO::Shakti::HalideBackend
