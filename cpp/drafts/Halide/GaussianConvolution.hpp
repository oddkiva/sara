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

#include "shakti_gaussian_convolution_v2.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  auto gaussian_convolution(Sara::ImageView<float>& src,         //
                            Sara::ImageView<float>& dst,         //
                            float sigma, int truncation_factor)  //
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = as_runtime_buffer(src_tensor_view);
    auto dst_buffer = as_runtime_buffer(dst_tensor_view);

    src_buffer.set_host_dirty();
    shakti_gaussian_convolution_v2(src_buffer, sigma, truncation_factor,
                                   dst_buffer);
    dst_buffer.copy_to_host();
  }

}}}  // namespace DO::Shakti::HalideBackend
