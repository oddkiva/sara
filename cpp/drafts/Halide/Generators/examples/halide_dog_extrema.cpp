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
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/RefineExtrema.hpp>

#include "shakti_halide_gray32f_to_rgb.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


auto show_dog_pyramid(const sara::ImagePyramid<float>& dog_pyramid)
{
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
}


auto gradient_pyramid(sara::ImagePyramid<float>& gaussian_pyramid) -> void
{
  // auto polar_gradient_pyr = std::vector<Halide::Func>(num_scales);
  // for (auto s = 0; s < num_scales; ++s)
  // {
  //   auto grad = Halide::Func{"gradient_" + std::to_string(s)};
  //   auto g_cart = hal::gradient(gauss_pyr[s].output, x, y);
  //   auto g_mag = Halide::sqrt(g_cart(0) * g_cart(0) + g_cart(1) * g_cart(1));
  //   auto g_ori = Halide::atan2(g_cart(1), g_cart(0));

  //   grad(x, y) = Halide::Tuple(g_mag, g_ori);
  //   polar_gradient_pyr.push_back(grad);
  // }
}


GRAPHICS_MAIN()
{
  auto timer = sara::Timer{};

  const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  // const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  timer.restart();
  auto gauss_pyramid = halide::gaussian_pyramid(image);
  auto dog_pyramid = halide::difference_of_gaussians_pyramid(image);
  SARA_DEBUG << "DoG calculation = " << timer.elapsed_ms() << " ms" << std::endl;

  timer.restart();
  auto dog_extrema_pyramid = halide::local_scale_space_extrema(dog_pyramid);
  SARA_DEBUG << "DoG extrema map = " << timer.elapsed_ms() << " ms" << std::endl;

  // Populate the scale-space extrema.
  auto f0 = halide::populate_local_scale_space_extrema(dog_extrema_pyramid);

  // Refine the scale-space extrema.
  const auto f1 = halide::refine_scale_space_extrema(dog_pyramid, f0);

  // Show the DoG pyramid.
  // sara::create_window(dog_pyramid(0, 0).sizes());
  // sara::set_antialiasing(sara::active_window());
  // show_dog_pyramid(dog_pyramid);

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing(sara::active_window());
  sara::display(image);

  const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };
  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    const auto oct_scale = dog_pyramid.octave_scaling_factor(o);

    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto& dog_ext_map = dog_extrema_pyramid(s, o);

      // const auto& f0_so = f0[at(s, o)];
      const auto& f1_so = f1[at(s, o)];

      for (auto i = 0u; i < f1_so.x.size(); ++i)
      {
        // const auto c0 = f0_so.type[i] == 1 ? sara::Blue8 : sara::Red8;
        // const auto& x0 = f0_so.x[i];
        // const auto& y0 = f0_so.y[i];
        // const auto r0 = dog_extrema_pyramid.scale(s, o) * std::sqrt(2.f);
        // sara::draw_circle(x0 * oct_scale, y0 * oct_scale, r0, c0, 2 + 0);

        const auto c1 = f1_so.type[i] == 1 ? sara::Cyan8 : sara::Magenta8;
        const auto& x1 = f1_so.x[i];
        const auto& y1 = f1_so.y[i];
        const auto r1 = f1_so.s[i] * oct_scale;  // * std::sqrt(2.f);
        sara::draw_circle(x1 * oct_scale, y1 * oct_scale, r1, c1, 2 + 1);
      }
    }
  }
  sara::get_key();

  return 0;
}
