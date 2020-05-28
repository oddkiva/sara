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

#include <drafts/Halide/Utilities.hpp>

#include "shakti_local_max_32f.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  auto local_max(Sara::ImageView<float>& a,  //
                 Sara::ImageView<float>& b,  //
                 Sara::ImageView<float>& c,  //
                 Sara::ImageView<int32_t>& out)
  {
    auto a_tensor_view = tensor_view(a).reshape(
        Eigen::Vector4i{1, 1, a.height(), a.width()});
    auto b_tensor_view = tensor_view(b).reshape(
        Eigen::Vector4i{1, 1, b.height(), b.width()});
    auto c_tensor_view = tensor_view(c).reshape(
        Eigen::Vector4i{1, 1, c.height(), c.width()});
    auto out_tensor_view = tensor_view(out).reshape(
        Eigen::Vector4i{1, 1, out.height(), out.width()});

    auto a_buffer = as_runtime_buffer(a_tensor_view);
    auto b_buffer = as_runtime_buffer(b_tensor_view);
    auto c_buffer = as_runtime_buffer(c_tensor_view);
    auto out_buffer = as_runtime_buffer(out_tensor_view);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();
    shakti_local_max_32f(a_buffer, b_buffer, c_buffer, out_buffer);
    out_buffer.copy_to_host();
  }

}}}  // namespace DO::Shakti::HalideBackend
