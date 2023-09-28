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
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_separable_convolution_2d_gpu.h"


namespace DO::Shakti::HalideBackend {

  auto separable_convolution_2d(::Halide::Runtime::Buffer<float>& src,     //
                                ::Halide::Runtime::Buffer<float>& kernel,  //
                                ::Halide::Runtime::Buffer<float>& dst,     //
                                int kernel_size, int kernel_shift)       //
  {
    shakti_separable_convolution_2d_gpu(src, kernel, kernel_size, kernel_shift,
                                        dst);
  }

  auto separable_convolution_2d(const Sara::ImageView<float>& src,
                                const Eigen::VectorXf& kernel,
                                Sara::ImageView<float>& dst, int kernel_shift)
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = as_runtime_buffer(src_tensor_view);
    auto kernel_buffer = DO::Shakti::Halide::as_runtime_buffer(kernel);
    auto dst_buffer = as_runtime_buffer(dst_tensor_view);

    src_buffer.set_host_dirty();
    separable_convolution_2d(src_buffer, kernel_buffer, dst_buffer,
                             static_cast<int>(kernel.size()), kernel_shift);
    dst_buffer.copy_to_host();
  }

}  // namespace DO::Shakti::HalideBackend
