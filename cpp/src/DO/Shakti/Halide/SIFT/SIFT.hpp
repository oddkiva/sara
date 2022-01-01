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

#pragma once

#include <DO/Sara/Core/Image.hpp>

#include <DO/Shakti/Halide/SIFT/V2/Pipeline.hpp>


namespace Shakti::Halide {

  struct SIFT
  {
    SIFT(ImageView<float>& grayscale_image)
      : buffer_gray_4d{halide::as_runtime_buffer(grayscale_image)}
    {
      sift_pipeline.initialize(start_octave_index, num_scales_per_octave,
                               grayscale_image.width(),
                               grayscale_image.height());
    }

    auto update() {
      sara::tic();
      buffer_gray_4d.set_host_dirty();
      sift_pipeline.feed(buffer_gray_4d);
      sara::toc("SIFT");

      sara::tic();
      keys_prev.swap(keys_curr);
      sift_pipeline.get_keypoints(keys_curr);
      sara::toc("Feature Reformatting");
    }

    ::Halide::Runtime::Buffer<float> buffer_gray_4d;
    DO::Shakti::HalideBackend::v2::SiftPyramidPipeline pipeline;
  };

}  // namespace Shakti::Halide
