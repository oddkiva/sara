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
#include <drafts/Halide/LocalExtrema.hpp>

#include "shakti_halide_gray32f_to_rgb.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


using namespace std;


GRAPHICS_MAIN()
{
  const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  auto image = sara::imread<float>(image_filepath);

  // auto gauss_pyramid = halide::gaussian_pyramid(image);
  auto dog_pyramid = halide::difference_of_gaussians_pyramid(image);
  auto dog_extrema_pyramid = halide::local_scale_space_extrema(dog_pyramid);

  sara::create_window(dog_pyramid(0, 0).sizes());
  sara::set_antialiasing(sara::active_window());

  // Show the DoG pyramid.
  for (auto o = 0; o < dog_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_pyramid.num_scales_per_octave(); ++s)
    {
      auto dog = dog_pyramid(s, o);

      auto image_rgb = sara::Image<sara::Rgb8>{dog.sizes()};
      dog.flat_array() = (dog.flat_array() + 1.f) / 2.f;
      auto buffer_gray = halide::as_runtime_buffer<float>(dog);
      auto buffer_rgb = halide::as_interleaved_runtime_buffer(image_rgb);
      shakti_halide_gray32f_to_rgb(buffer_gray, buffer_rgb);

      sara::display(image_rgb);
    }
  }
  sara::millisleep(1000);

  // Now show the local extrema.
  sara::display(image);
  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    const auto oct_scale = dog_pyramid.octave_scaling_factor(o);

    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      // SARA_CHECK(dog_pyramid.scale(s, o));
      // SARA_CHECK(dog_pyramid.scale_relative_to_octave(s));

      // SARA_CHECK(dog_max_pyramid.scale(s, o));
      // SARA_CHECK(dog_max_pyramid.scale_relative_to_octave(s));

      const auto& dog_ext_map = dog_extrema_pyramid(s, o);
      const auto radius = dog_extrema_pyramid.scale(s, o) * std::sqrt(2.f);

      for (auto y = 0; y < dog_ext_map.height(); ++y)
        for (auto x = 0; x < dog_ext_map.width(); ++x)
        {
          if (dog_ext_map(x, y) == 0)
            continue;

          const auto color = dog_ext_map(x, y) == 1 ? sara::Blue8 : sara::Red8;
          sara::draw_circle(x * oct_scale, y * oct_scale, radius, color, 2);
        }
    }
  }
  sara::get_key();

  return 0;
}
