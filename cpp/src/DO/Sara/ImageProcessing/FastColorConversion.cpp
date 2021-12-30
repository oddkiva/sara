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

#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>
#include "shakti_rgb8u_to_gray32f_cpu.h"
#include "shakti_bgra8u_to_gray32f_cpu.h"
#endif


namespace DO::Sara {

  auto from_rgb8_to_gray32f(const ImageView<Rgb8>& src, ImageView<float>& dst) -> void
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
          "Color conversion error: image sizes are not equal!"};

#ifdef DO_SARA_USE_HALIDE
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Grayscale] 0.56932 ms
    auto src_buffer = Shakti::Halide::as_interleaved_runtime_buffer(src);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer(dst);
    shakti_rgb8u_to_gray32f_cpu(src_buffer, dst_buffer);
#else
    // FALLBACK IMPLEMENTATION.
    //
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Grayscale] 8.8687 ms
    // This is 15 times slower compared to the Halide optimized CPU implementation
    DO::Sara::convert(src, dst);
#endif
  }

  auto from_bgra8_to_gray32f(const ImageView<Bgra8>& src, ImageView<float>& dst) -> void
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
          "Color conversion error: image sizes are not equal!"};

#ifdef DO_SARA_USE_HALIDE
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Grayscale] 0.56932 ms
    auto src_buffer = Shakti::Halide::as_interleaved_runtime_buffer(src);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer(dst);
    shakti_bgra8u_to_gray32f_cpu(src_buffer, dst_buffer);
#else
    // FALLBACK IMPLEMENTATION.
    //
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Grayscale] 8.8687 ms
    // This is 15 times slower compared to the Halide optimized CPU implementation
    DO::Sara::convert(src, dst);
#endif
  }


}  // namespace DO::Sara
