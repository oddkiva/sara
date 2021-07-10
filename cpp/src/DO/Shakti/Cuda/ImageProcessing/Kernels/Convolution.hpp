// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/ImageProcessing/Kernels/Globals.hpp>

#include <DO/Shakti/Cuda/MultiArray/Matrix.hpp>
#include <DO/Shakti/Cuda/MultiArray/Offset.hpp>


namespace DO { namespace Shakti {

  template <typename T>
  __global__
  void apply_column_based_convolution(T *dst)
  {
    const auto p = coords<2>();
    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    const auto i = p.dot(grid_strides<2>());
    const auto kernel_radius = kernel_size / 2;

    auto convolved_value = T{0};
#pragma unroll
    for (int i = 0; i < kernel_size; ++i)
      convolved_value +=
          tex2D(in_float_texture, p.x() - kernel_radius + i, p.y()) * kernel[i];

    dst[i] = convolved_value;
  }

  template <typename T>
  __global__
  void apply_row_based_convolution(T *dst)
  {
    const auto p = coords<2>();
    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    const auto i = p.dot(grid_strides<2>());
    const auto kernel_radius = kernel_size / 2;

    auto convolved_value = T{0};
#pragma unroll
    for (int i = 0; i < kernel_size; ++i)
      convolved_value +=
          tex2D(in_float_texture, p.x(), p.y() - kernel_radius + i) * kernel[i];

    dst[i] = convolved_value;
  }

}}  // namespace DO::Shakti
