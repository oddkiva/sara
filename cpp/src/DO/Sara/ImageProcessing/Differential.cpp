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

#include <DO/Sara/ImageProcessing/Differential.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>
#include "shakti_polar_gradient_2d_32f_cpu.h"
#endif


namespace DO::Sara {

  auto gradient_in_polar_coordinates(const ImageView<float>& src,
                                     ImageView<float>& magnitude,
                                     ImageView<float>& orientation) -> void
  {
    if (src.sizes() != magnitude.sizes() || src.sizes() != orientation.sizes())
      throw std::domain_error{"Polar gradients: image sizes are not equal!"};

#ifdef DO_SARA_USE_HALIDE
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Polar Coordinates] 27.3127 ms

    // For some reason, Halide AOT requires the buffer to have a width of size
    // 64 minimum.
    auto src_buffer = Shakti::Halide::as_runtime_buffer_4d(src);
    auto mag_buffer = Shakti::Halide::as_runtime_buffer_4d(magnitude);
    auto ori_buffer = Shakti::Halide::as_runtime_buffer_4d(orientation);

    src_buffer.set_host_dirty();
    shakti_polar_gradient_2d_32f_cpu(src_buffer, mag_buffer, ori_buffer);
    mag_buffer.copy_to_host();
    ori_buffer.copy_to_host();
#else
    // FALLBACK IMPLEMENTATION.
    //
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Gradient] 36.8563 ms
    // [Polar Coordinates] 165.966 ms
    // This is 7.48 times slower compared to the Halide optimized CPU implementation
    const auto gradient_cartesian = gradient(src);
    magnitude = gradient_cartesian.cwise_transform(
        [](const auto& v) { return v.norm(); });
    orientation = gradient_cartesian.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });
#endif
  }

}  // namespace DO::Sara
