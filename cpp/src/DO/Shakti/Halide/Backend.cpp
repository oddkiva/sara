// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Shakti/Halide/Backend.hpp>
#include <DO/Shakti/Halide/MyHalide.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>


namespace DO::Shakti::Halide::cpu {

  template <typename T>
  auto as_runtime_buffer_4d(const Sara::ImageView<T>& src)
      -> ::Halide::Runtime::Buffer<T>
  {
    auto src_non_const = const_cast<Sara::ImageView<T>&>(src);
    auto src_tensor_view =
        tensor_view(src_non_const)
            .reshape(Eigen::Vector4i{1, 1, src.height(), src.width()});
    return as_runtime_buffer(src_tensor_view);
  }

  auto gaussian_convolution(const Sara::ImageView<float>& src,
                            Sara::ImageView<float>& dst, float sigma,
                            int truncation_factor = 4) -> void
  {
    auto src_buffer = as_runtime_buffer_4d(src);
    auto dst_buffer = as_runtime_buffer_4d(dst);
    shakti_gaussian_convolution_v2_cpu(src_buffer, sigma, truncation_factor,
                                       dst_buffer);
  }

}  // namespace DO::Shakti::Halide::cpu
