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

#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_subtract_32f.h"


namespace DO::Shakti::HalideBackend {

  auto subtract(Halide::Runtime::Buffer<float>& a,  //
                Halide::Runtime::Buffer<float>& b,  //
                Halide::Runtime::Buffer<float>& out)
  {
    shakti_subtract_32f(a, b, out);
  }

  auto subtract(Sara::ImageView<float>& a,  //
                Sara::ImageView<float>& b,  //
                Sara::ImageView<float>& out)
  {
    auto a_tensor_view = tensor_view(a).reshape(
        Eigen::Vector4i{1, 1, a.height(), a.width()});
    auto b_tensor_view = tensor_view(b).reshape(
        Eigen::Vector4i{1, 1, b.height(), b.width()});
    auto out_tensor_view = tensor_view(out).reshape(
        Eigen::Vector4i{1, 1, out.height(), out.width()});

    auto a_buffer = as_runtime_buffer(a_tensor_view);
    auto b_buffer = as_runtime_buffer(b_tensor_view);
    auto out_buffer = as_runtime_buffer(out_tensor_view);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    subtract(a_buffer, b_buffer, out_buffer);
    out_buffer.copy_to_host();
  }

}  // namespace DO::Shakti::HalideBackend
