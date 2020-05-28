// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <algorithm>
#include <cmath>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Utilities.hpp>
#include <drafts/Halide/GaussianPyramid.hpp>

#include "shakti_halide_gray32f_to_rgb.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


using namespace std;


GRAPHICS_MAIN()
{
  const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  auto image = sara::imread<float>(image_filepath);

  // auto pyramid = shakti::HalideBackend::gaussian_pyramid(image);
  auto pyramid = shakti::HalideBackend::difference_of_gaussians_pyramid(image);

  sara::create_window(pyramid(0,0).sizes());
  for (auto o = 0; o < pyramid.num_octaves(); ++o)
    for (auto s = 0; s < pyramid.num_scales_per_octave(); ++s)
    {
      pyramid(s, o).flat_array() = (pyramid(s, o).flat_array() + 1.f) / 2.f;
      auto buffer_gray = halide::as_runtime_buffer<float>(pyramid(s, o));

      auto image_rgb = sara::Image<sara::Rgb8>{pyramid(s, o).sizes()};
      auto buffer_rgb = halide::as_interleaved_runtime_buffer(image_rgb);

      shakti_halide_gray32f_to_rgb(buffer_gray, buffer_rgb);
      sara::display(image_rgb);
    }

  sara::get_key();

  return 0;
}
