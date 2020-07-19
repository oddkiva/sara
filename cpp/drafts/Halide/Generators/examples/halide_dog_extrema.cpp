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

#include <Eigen/Sparse>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Differential.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/RefineExtrema.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <drafts/Halide/DominantGradientOrientations.hpp>

#include "shakti_halide_gray32f_to_rgb.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


auto show_dog_pyramid(sara::ImagePyramid<float>& dog_pyramid)
{
  for (auto o = 0; o < dog_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_pyramid.num_scales_per_octave(); ++s)
    {
      auto& dog = dog_pyramid(s, o);

      auto image_rgb = sara::Image<sara::Rgb8>{dog.sizes()};
      dog.flat_array() = (dog.flat_array() + 1.f) / 2.f;
      auto buffer_gray = halide::as_runtime_buffer<float>(dog);
      auto buffer_rgb = halide::as_interleaved_runtime_buffer(image_rgb);
      shakti_halide_gray32f_to_rgb(buffer_gray, buffer_rgb);

      sara::display(image_rgb);
    }
  }
}

auto show_pyramid(const sara::ImagePyramid<float>& pyramid)
{
  for (auto o = 0; o < pyramid.num_octaves(); ++o)
    for (auto s = 0; s < pyramid.num_scales_per_octave(); ++s)
      sara::display(sara::color_rescale(pyramid(s, o)));
}


GRAPHICS_MAIN()
{
  auto timer = sara::Timer{};

  const auto image_filepath =
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  // const auto image_filepath =
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  timer.restart();
  auto gauss_pyramid = halide::gaussian_pyramid(image);
  auto dog_pyramid = halide::subtract_pyramid(gauss_pyramid);
  SARA_DEBUG << "DoG pyramid = " << timer.elapsed_ms() << " ms" << std::endl;

  timer.restart();
  auto dog_extrema_pyramid = halide::local_scale_space_extrema(dog_pyramid);
  SARA_DEBUG << "DoG extrema = " << timer.elapsed_ms() << " ms" << std::endl;

  timer.restart();
  auto mag_pyramid = sara::ImagePyramid<float>{};
  auto ori_pyramid = sara::ImagePyramid<float>{};
  std::tie(mag_pyramid, ori_pyramid) = halide::polar_gradient_2d(gauss_pyramid);
  SARA_DEBUG << "Gradient pyramid = " << timer.elapsed_ms() << " ms"
             << std::endl;

  // Populate the DoG extrema.
  auto extrema_quantized =
      halide::populate_local_scale_space_extrema(dog_extrema_pyramid);

  // Refine the scale-space localization of each extremum.
  auto extrema =
      halide::refine_scale_space_extrema(dog_pyramid, extrema_quantized);

  // Estimate the dominant gradient orientations.
  using halide::DominantGradientOrientationMap;
  using halide::Pyramid;
  timer.restart();
  auto dominant_orientations = Pyramid<DominantGradientOrientationMap>{};
  halide::dominant_gradient_orientations(mag_pyramid, ori_pyramid, extrema,
                                         dominant_orientations);
  SARA_DEBUG << "Dominant gradient orientations = " << timer.elapsed_ms()
             << " ms" << std::endl;

  const auto dominant_orientations_sparse = compress(dominant_orientations);

  // TODO: write  call to SIFT.


  // Show the DoG pyramid.
  sara::create_window(dog_pyramid(0, 0).sizes());
  sara::set_antialiasing(sara::active_window());
  show_pyramid(dog_pyramid);
  // sara::get_key();

  SARA_DEBUG << "Gradient magnitude pyramid" << std::endl;
  show_pyramid(mag_pyramid);
  // sara::get_key();

  SARA_DEBUG << "Gradient orientation pyramid" << std::endl;
  show_pyramid(ori_pyramid);
  // sara::get_key();


  // Show the local extrema.
  sara::resize_window(image.sizes());
  sara::set_antialiasing(sara::active_window());
  sara::display(image);
  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    const auto oct_scale = dog_pyramid.octave_scaling_factor(o);

    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto& dog_ext_map = dog_extrema_pyramid(s, o);

      const auto& extrema_quantized_so = extrema_quantized.dict[{s, o}];
      const auto& extrema_so = extrema.dict[{s, o}];
      const auto& ori_so = dominant_orientations.dict[{s, o}];

      SARA_CHECK(extrema_quantized_so.size());
      SARA_CHECK(extrema_so.size());
      SARA_CHECK(ori_so.num_keypoints());
      SARA_CHECK(ori_so.num_orientation_bins());

      SARA_CHECK(ori_so.peak_map.matrix().rows());
      SARA_CHECK(ori_so.peak_map.matrix().cols());
      SARA_CHECK(ori_so.peak_residuals.matrix().rows());
      SARA_CHECK(ori_so.peak_residuals.matrix().cols());

      const auto peak_count =
          ori_so.peak_map.matrix().array().rowwise().count().eval();
      SARA_CHECK(peak_count.rows());
      SARA_CHECK(peak_count.cols());

      for (auto i = 0u; i < extrema_so.x.size(); ++i)
      {
#ifdef SHOW_QUANTIZED_EXTREMA
        const auto c0 =
            extrema_quantized_so.type[i] == 1 ? sara::Blue8 : sara::Red8;
        const auto& x0 = extrema_quantized_so.x[i];
        const auto& y0 = extrema_quantized_so.y[i];
        const auto r0 = dog_extrema_pyramid.scale(s, o) * std::sqrt(2.f);
        sara::draw_circle(x0 * oct_scale, y0 * oct_scale, r0, c0, 2 + 0);
#endif

        const auto c1 = extrema_so.type[i] == 1 ? sara::Cyan8 : sara::Magenta8;
        const auto& x1 = extrema_so.x[i];
        const auto& y1 = extrema_so.y[i];
        // N.B.: the blob radius is the scale multiplied sqrt(2).
        // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
        const auto r1 = extrema_so.s[i] * oct_scale * std::sqrt(2.f);
        sara::draw_circle(x1 * oct_scale, y1 * oct_scale, r1, c1, 2 + 1);

#ifdef DEBUG
        SARA_CHECK(i);
        SARA_CHECK(ori_so.peak_map[i].flat_array().size());
        SARA_CHECK(ori_so.peak_map[i].flat_array().transpose());
        SARA_CHECK(ori_so.peak_residuals[i].flat_array().size());
        SARA_CHECK(ori_so.peak_residuals[i].flat_array().transpose());
#endif

        const auto orientations = dominant_orientations_sparse.dict.at({s, o}).equal_range(i);
        for (auto ori = orientations.first; ori != orientations.second; ++ori)
        {
          const auto &angle = ori->second;
          const Eigen::Vector2f a = Eigen::Vector2f{x1 * oct_scale, y1 * oct_scale};
          const Eigen::Vector2f b =
              a + r1 * Eigen::Vector2f{cos(angle), sin(angle)};
          sara::draw_line(a, b, c1, 2);

#ifdef DEBUG
          std::cout << "index = " << ori->first
                    << "  ori = " << angle * 180 / M_PI << " deg"
                    << std::endl;
#endif
        }
      }
    }
  }
  sara::get_key();

  return 0;
}
