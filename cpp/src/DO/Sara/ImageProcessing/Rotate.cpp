// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Sara/ImageProcessing/Rotate.hpp>

#if defined(DO_SARA_USE_HALIDE)
#  include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#  include "shakti_rotate_cw_90_rgb8_cpu.h"
#else
#include <DO/Sara/ImageProcessing/Flip.hpp>
#endif

namespace DO::Sara {

  auto rotate_cw_90(const ImageView<Rgb8>&src, ImageView<Rgb8>&dst) -> void
  {
#ifdef DO_SARA_USE_HALIDE
    // Typical timing with a 4K video on my Desktop CPU:
    // - model name      : Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
    //
    // [Grayscale] 0.56932 ms
    auto src_buffer = Shakti::Halide::as_interleaved_runtime_buffer(src);
    auto dst_buffer = Shakti::Halide::as_interleaved_runtime_buffer(dst);
    shakti_rotate_cw_90_rgb8_cpu(src_buffer, dst_buffer);
#else
    v2::rotate_cw_90(src, dst);
#endif
  }

}  // namespace DO::Sara
