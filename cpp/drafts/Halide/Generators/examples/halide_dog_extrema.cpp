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
#include <drafts/Halide/Differential.hpp>

#include <drafts/Halide/Components/DominantOrientation.hpp>

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


constexpr auto use_gpu = false;
const auto target = use_gpu ? halide::get_gpu_target() : Halide::Target{};
constexpr auto tile_k = 16;
constexpr auto tile_o = 16;

auto k = Halide::Var{"k"};
auto o = Halide::Var{"o"};

auto ko = Halide::Var{"ko"};
auto oo = Halide::Var{"oo"};

auto ki = Halide::Var{"ki"};
auto oi = Halide::Var{"oi"};

// Orientation histograms.
constexpr auto N = 36;


auto make_orientation_histogram_funcs(sara::ImagePyramid<float>& dog_pyramid,
                                      sara::ImagePyramid<float>& mag_pyramid,
                                      sara::ImagePyramid<float>& ori_pyramid,
                                      std::vector<halide::DoGExtremaRefined>& f1)
{
  static const auto at = [&](int s, int o) {
    return o * (dog_pyramid.num_scales_per_octave() - 2) + s;
  };

  auto orientation_histogram_funcs = std::vector<Halide::Func>{};

  for (auto oct = 0; oct < dog_pyramid.num_octaves(); ++oct)
  {
    for (auto s = 1; s < dog_pyramid.num_scales_per_octave() - 1; ++s)
    {
      auto mag_fn = halide::as_buffer(mag_pyramid(s, oct));
      auto ori_fn = halide::as_buffer(ori_pyramid(s, oct));

      auto mag_fn_ext = Halide::BoundaryConditions::constant_exterior(  //
          mag_fn,                                                       //
          0);                                                           //
      auto ori_fn_ext = Halide::BoundaryConditions::constant_exterior(  //
          ori_fn,                                                       //
          0);                                                           //

      auto x = halide::as_buffer(f1[at(s, oct)].x);
      auto y = halide::as_buffer(f1[at(s, oct)].y);
      auto sigma = halide::as_buffer(f1[at(s, oct)].s);

      Halide::Expr length_in_scale_unit = 1.5f;
      Halide::Expr gaussian_truncation_factor = 3.f;
      Halide::Expr scale_max_factor = float(      //
          dog_pyramid.scale_geometric_factor());  //

      const auto residual_sigma =                          //
          sigma(k) /                                       //
          float(dog_pyramid.scale_relative_to_octave(s));  //

      auto hist_fn = Halide::Func{"orientation_histogram_" + std::to_string(s) +
                                  "_" + std::to_string(oct)};
      hist_fn(o, k) = halide::compute_orientation_histogram(  //
          mag_fn_ext, ori_fn_ext,                             //
          x(k), y(k), residual_sigma,                         //
          scale_max_factor,                                   //
          k, o);                                              //
      hist_fn.compute_root();
      halide::schedule_histograms(hist_fn,                               //
                                  o, k, oo, ko, oi, ki, tile_o, tile_k,  //
                                  use_gpu);                              //

      // Smooth histograms using repeated box blurs.
      auto box_blurs = halide::box_blur_histograms(hist_fn,               //
                                                   o, k, oo, ko, oi, ki,  //
                                                   tile_o, tile_k,        //
                                                   N, use_gpu, 6);

      orientation_histogram_funcs.push_back(box_blurs.back());
    }
  }

  return orientation_histogram_funcs;
}

auto realize_orientation_histograms(
    sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid,
    std::vector<Halide::Func>& orientation_histogram_funcs,
    std::vector<halide::DoGExtremaRefined>& f1)
{
  auto orientation_histograms = std::vector<Halide::Buffer<float>>(
      dog_extrema_pyramid.num_scales());

  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto num_dog_extrema = f1[at(s, o)].size();
      if (num_dog_extrema == 0)
      {
        SARA_DEBUG << "NO KEYPOINTS: SKIPPING" << std::endl;
        continue;
      }

      sara::tic();
      auto ori_hist = orientation_histogram_funcs[at(s, o)].realize(
          {N, static_cast<int>(num_dog_extrema)});
      orientation_histograms.emplace_back(ori_hist);
      sara::toc(
          sara::format("Realizing orientation histograms (%d, %d)", s, o));
    }
  }

  return orientation_histograms;
}


auto make_orientation_peak_funcs(sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid,
                                 std::vector<Halide::Buffer<float>>& ori_histograms,
                                 std::vector<halide::DoGExtremaRefined>& f1)
{
  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  auto ori_peak_funcs = std::vector<Halide::Func>{};

  for (auto oct = 0; oct < dog_extrema_pyramid.num_octaves(); ++oct)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      auto ori_peak_fn = Halide::Func{"ori_peak_" + std::to_string(s) + "_" +
                                      std::to_string(oct)};
      halide::localize_peaks(ori_histograms[at(s, oct)],
                             ori_peak_fn,
                             o, k,
                             N, 0.8f);
      halide::schedule_histograms(ori_peak_fn,
                                  o, k,
                                  oo, ko,
                                  oi, ki,
                                  tile_o, tile_k,
                                  use_gpu);
      ori_peak_funcs.push_back(ori_peak_fn);
    }
  }

  return ori_peak_funcs;
}

auto realize_orientation_peaks(sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid,
                               std::vector<halide::DoGExtremaRefined>& f1,
                               std::vector<Halide::Buffer<float>>& ori_histograms,
                               std::vector<Halide::Func>& ori_peak_funcs)
{
  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  auto ori_peak_maps = std::vector<Halide::Buffer<float>>{};

  for (auto oct = 0; oct < dog_extrema_pyramid.num_octaves(); ++oct)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto num_dog_extrema = f1[at(s, oct)].size();
      if (num_dog_extrema == 0)
      {
        SARA_DEBUG << "NO KEYPOINTS: SKIPPING" << std::endl;
        continue;
      }

      sara::tic();
      auto ori_peak_fn = Halide::Func{"ori_peak_" + std::to_string(s) + "_" +
                                      std::to_string(oct)};
      halide::localize_peaks(ori_histograms[at(s, oct)],  //
                             ori_peak_fn,                 //
                             o, k,                        //
                             N, 0.8f);                    //

      auto ori_peak = ori_peak_funcs[at(s, oct)].realize({oct, N});
      ori_peak_maps.emplace_back(ori_peak);
      sara::toc(sara::format("realizing orientation peak localization "
                             "(%d, %d)",
                             s, oct));
    }
  }

  return ori_peak_maps;
}


auto make_orientation_residual_funcs(
    sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid,
    std::vector<Halide::Buffer<float>>& ori_histograms,
    std::vector<Halide::Buffer<float>>& ori_peak_maps)
{
  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  auto ori_residual_funcs = std::vector<Halide::Func>{};

  for (auto oct = 0; oct < dog_extrema_pyramid.num_octaves(); ++oct)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      auto ori_residual_fn = Halide::Func{"ori_residual" + std::to_string(s) +
                                          "_" + std::to_string(oct)};

      ori_residual_fn(o, k) = halide::compute_peak_residual_map(
          ori_histograms[at(s, oct)],
          ori_peak_maps[at(s, oct)],
          o, k, N);
      halide::schedule_histograms(ori_residual_fn,  //
                                  o, k,             //
                                  oo, ko,           //
                                  oi, ki,           //
                                  tile_o, tile_k,   //
                                  use_gpu);         //

      ori_residual_funcs.push_back(ori_residual_fn);
    }
  }

  return ori_residual_funcs;
}

auto realize_orientation_residuals(
    sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid,
    std::vector<Halide::Buffer<float>>& ori_histograms,
    std::vector<Halide::Buffer<float>>& ori_peak_maps,
    std::vector<Halide::Func>& ori_residual_funcs,
    std::vector<halide::DoGExtremaRefined>& f1)
{
  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  auto ori_residuals = std::vector<Halide::Buffer<float>>{};

  for (auto oct = 0; oct < dog_extrema_pyramid.num_octaves(); ++oct)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto num_dog_extrema = f1[at(s, oct)].size();

      if (num_dog_extrema == 0)
      {
        SARA_DEBUG << "NO KEYPOINTS: SKIPPING" << std::endl;
        continue;
      }

      sara::tic();
      auto ori_peak = ori_residual_funcs[at(s, oct)].realize(  //
          {N, static_cast<int>(num_dog_extrema)});
      ori_residuals.emplace_back(ori_peak);
      sara::toc(sara::format("Realizing orientation residuals "
                             "(%d, %d)",
                             s, oct));
    }
  }

  return ori_residuals;
}


GRAPHICS_MAIN()
{
  auto timer = sara::Timer{};

  const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  // const auto image_filepath = "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
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
  SARA_DEBUG << "Gradient pyramid = " << timer.elapsed_ms() << " ms" << std::endl;

  // Populate the DoG extrema.
  auto f0 = halide::populate_local_scale_space_extrema(dog_extrema_pyramid);

  // Refine the scale-space localization of each extremum.
  auto f1 = halide::refine_scale_space_extrema(dog_pyramid, f0);

  // Set an index in the sequel.
  static const auto at = [&](int s, int o) {
    return o * dog_extrema_pyramid.num_scales_per_octave() + s;
  };

  // Calculate the gradient histograms for each extremum.
  auto ori_histogram_funcs = make_orientation_histogram_funcs(
      dog_pyramid, mag_pyramid, ori_pyramid, f1);

  auto ori_histograms = realize_orientation_histograms(  //
      dog_extrema_pyramid,                               //
      ori_histogram_funcs,                               //
      f1);                                               //

  auto ori_peak_funcs = make_orientation_peak_funcs(dog_extrema_pyramid,
                                                    ori_histograms,
                                                    f1);
  auto ori_peak_maps = realize_orientation_peaks(dog_extrema_pyramid,
                                                 f1,
                                                 ori_histograms,
                                                 ori_peak_funcs);

  auto ori_residual_funcs = make_orientation_residual_funcs(dog_extrema_pyramid,
                                                            ori_histograms,
                                                            ori_peak_maps);

  auto ori_residuals = realize_orientation_residuals(dog_extrema_pyramid,
                                                     ori_histograms,
                                                     ori_peak_maps,
                                                     ori_residual_funcs,
                                                     f1);

  struct OrientedDoG {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> sigma;
    std::vector<float> orientation;
  };

  auto oriented_dogs = std::vector<OrientedDoG>(  //
      dog_extrema_pyramid.num_scales());          //
  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto so = at(s, o);
      for (auto k = std::size_t{}; k < f1[so].size(); ++k)
      {
        const auto& x = f1[so].x[k];
        const auto& y = f1[so].y[k];
        const auto& s = f1[so].s[k];

        for (auto ori = 0; ori < N; ++ori)
        {
          if (ori_peak_maps[so](ori, k) == 0)
            continue;

          const auto theta = (ori + ori_residuals[so](ori, k)) * 2 * M_PI;

          oriented_dogs[so].x.push_back(x);
          oriented_dogs[so].y.push_back(y);
          oriented_dogs[so].sigma.push_back(s);
          oriented_dogs[so].orientation.push_back(theta);
        }
      }
    }
  }


  // TODO: write  call to SIFT.



  // Show the DoG pyramid.
  sara::create_window(dog_pyramid(0, 0).sizes());
  sara::set_antialiasing(sara::active_window());
  //show_dog_pyramid(dog_pyramid);

  SARA_DEBUG << "Gradient magnitude pyramid" << std::endl;
  show_pyramid(mag_pyramid);
  sara::get_key();

  SARA_DEBUG << "Gradient orientation pyramid" << std::endl;
  show_pyramid(ori_pyramid);
  sara::get_key();




  // Show the local extrema.
  // sara::create_window(image.sizes());
  // sara::set_antialiasing(sara::active_window());
  sara::resize_window(image.sizes());
  sara::set_antialiasing(sara::active_window());
  sara::display(image);
  for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
  {
    const auto oct_scale = dog_pyramid.octave_scaling_factor(o);

    for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
    {
      const auto& dog_ext_map = dog_extrema_pyramid(s, o);

      const auto& f0_so = f0[at(s, o)];
      const auto& f1_so = f1[at(s, o)];

      for (auto i = 0u; i < f1_so.x.size(); ++i)
      {
        const auto c0 = f0_so.type[i] == 1 ? sara::Blue8 : sara::Red8;
        const auto& x0 = f0_so.x[i];
        const auto& y0 = f0_so.y[i];
        const auto r0 = dog_extrema_pyramid.scale(s, o) * std::sqrt(2.f);
        sara::draw_circle(x0 * oct_scale, y0 * oct_scale, r0, c0, 2 + 0);

        const auto c1 = f1_so.type[i] == 1 ? sara::Cyan8 : sara::Magenta8;
        const auto& x1 = f1_so.x[i];
        const auto& y1 = f1_so.y[i];
        // N.B.: the blob radius is the scale multiplied sqrt(2).
        // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
        const auto r1 = f1_so.s[i] * oct_scale * std::sqrt(2.f);
        sara::draw_circle(x1 * oct_scale, y1 * oct_scale, r1, c1, 2 + 1);
      }
    }
  }
  sara::get_key();

  return 0;
}
