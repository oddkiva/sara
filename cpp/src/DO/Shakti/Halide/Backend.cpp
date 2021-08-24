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

#include <DO/Shakti/Halide/Backend.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_convolve_batch_32f_cpu.h"
#include "shakti_gaussian_convolution_cpu.h"


namespace DO::Shakti::Halide::CPU {

  auto convolve(const Sara::TensorView_<float, 4>& src,
                const Sara::TensorView_<float, 4>& kernel,
                Sara::TensorView_<float, 4>& dst) -> void
  {
    auto src_buffer =
        as_runtime_buffer(const_cast<Sara::TensorView_<float, 4>&>(src));
    auto kernel_buffer =
        as_runtime_buffer(const_cast<Sara::TensorView_<float, 4>&>(kernel));
    auto dst_buffer = as_runtime_buffer(dst);
    shakti_convolve_batch_32f_cpu(src_buffer, kernel_buffer, dst_buffer);
  }

  auto gaussian_convolution(const Sara::ImageView<float>& src,
                            Sara::ImageView<float>& dst, float sigma,
                            int truncation_factor) -> void
  {
    auto src_buffer = as_runtime_buffer_4d(src);
    auto dst_buffer = as_runtime_buffer_4d(dst);
    shakti_gaussian_convolution_cpu(src_buffer, sigma, truncation_factor,
                                    dst_buffer);
  }

}  // namespace DO::Shakti::Halide::CPU
