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
    inline SIFT() = default;

    inline SIFT(const Sara::ImageView<float>& grayscale_image)
    {
      rebind(grayscale_image);
    }

    inline auto rebind(const Sara::ImageView<float>& grayscale_image) -> void
    {
      buffer_gray_4d = as_runtime_buffer(grayscale_image);
      sift_pipeline.initialize(start_octave_index, num_scales_per_octave,
                               grayscale_image.width(),
                               grayscale_image.height());
    }

    inline auto calculate(Sara::KeypointList<Sara::OERegion, float>& keys)
        -> void
    {
      Sara::tic();
      buffer_gray_4d.set_host_dirty();
      pipeline.feed(buffer_gray_4d);
      Sara::toc("SIFT");

      Sara::tic();
      pipeline.get_keypoints(keys);
      Sara::toc("Feature Reformatting");
    }

    int start_octave_index = 0;
    int num_scales_per_octave = 3;
    ::Halide::Runtime::Buffer<float> buffer_gray_4d;
    DO::Shakti::HalideBackend::v2::SiftPyramidPipeline pipeline;
  };

}  // namespace Shakti::Halide
