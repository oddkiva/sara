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

#include <omp.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Differential.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/RefineExtrema.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <drafts/Halide/DominantGradientOrientations.hpp>
#include <drafts/Halide/Draw.hpp>
#include <drafts/Halide/Resize.hpp>
#include <drafts/Halide/SIFT.hpp>

#include "shakti_dominant_gradient_orientations_v2.h"
#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_local_scale_space_extremum_32f_v2.h"
#include "shakti_polar_gradient_2d_32f_v2.h"
#include "shakti_refine_scale_space_extrema_v2.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;




// ===========================================================================
// SIFT octave parameters.
//
constexpr auto scale_camera = 1.f;
constexpr auto scale_initial = 1.6f;
constexpr auto scale_factor = std::pow(2.f, 1 / 3.f);
constexpr auto num_scales = 6;
constexpr auto edge_ratio = 10.0f;
constexpr auto extremum_thres = 0.01f;
constexpr auto num_orientation_bins = 36;
constexpr auto gaussian_truncation_factor = 3.f;
constexpr auto scale_multiplying_factor = 1.5f;
constexpr auto peak_ratio_thres = 0.8f;





// ===========================================================================
// Video processing parameters.
//
const auto downscale_factor = 1;





auto impl1() -> int
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // "/home/david/Desktop/GOPR0542.MP4"s;
#endif


  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);


  // Successive Gaussian convolutions.
  auto scales = std::vector<float>(num_scales);
  for (auto i = 0; i < num_scales; ++i)
    scales[i] = scale_initial * std::pow(scale_factor, i);

  auto sigmas = std::vector<float>(num_scales);
  for (auto i = 0u; i < sigmas.size(); ++i)
  {
    sigmas[i] =
        i == 0
            ? std::sqrt(std::pow(scale_initial, 2) - std::pow(scale_camera, 2))
            : std::sqrt(std::pow(scales[i], 2) - std::pow(scales[i - 1], 2));
  }


  // ===========================================================================
  // SARA PIPELINE
  //
  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray = sara::Image<float>{frame.sizes()};
  auto frame_gray_tensor =
      tensor_view(frame_gray)
          .reshape(
              Eigen::Vector4i{1, 1, frame_gray.height(), frame_gray.width()});

  auto frame_conv = sara::Image<float>{frame.sizes() / downscale_factor};
  auto frame_conv_tensor =
      tensor_view(frame_conv)
          .reshape(
              Eigen::Vector4i{1, 1, frame_conv.height(), frame_conv.width()});


  auto extrema_maps = std::vector<sara::Image<std::int8_t>>(
      num_scales - 3, sara::Image<std::int8_t>{frame_conv.sizes()});


  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);

  // Downsampling.
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);
  auto buffer_gray_down_4d = Halide::Runtime::Buffer<float>(
      frame_conv.width(), frame_conv.height(), 1, 1);

  // Octave of Gaussians.
  auto buffer_convs =
      std::vector<Halide::Runtime::Buffer<float>>(sigmas.size());
  for (auto i = 0u; i < buffer_convs.size(); ++i)
    buffer_convs[i] = Halide::Runtime::Buffer<float>(
        buffer_gray_down_4d.width(), buffer_gray_down_4d.height(), 1, 1);

  // Octave of difference of Gaussians.
  auto buffer_dogs =
      std::vector<Halide::Runtime::Buffer<float>>(buffer_convs.size() - 1);
  for (auto i = 0u; i < buffer_dogs.size(); ++i)
  {
    buffer_dogs[i] =
        i != buffer_dogs.size() - 1
            ? Halide::Runtime::Buffer<float>(buffer_gray_down_4d.width(),
                                             buffer_gray_down_4d.height(), 1, 1)
            : halide::as_runtime_buffer(frame_conv_tensor);
  }


  auto buffer_extremas =
      std::vector<Halide::Runtime::Buffer<std::int8_t>>(num_scales - 3);
  for (auto i = 0u; i < extrema_maps.size(); ++i)
    buffer_extremas[i] = Halide::Runtime::Buffer<std::int8_t>{
        extrema_maps[i].data(),                                  //
        extrema_maps[i].width(), extrema_maps[i].height(), 1, 1  //
    };

  auto buffer_mag = std::vector<Halide::Runtime::Buffer<float>>(num_scales - 3);
  auto buffer_ori = std::vector<Halide::Runtime::Buffer<float>>(num_scales - 3);
  for (auto i = 0u; i < buffer_mag.size(); ++i)
  {
    buffer_mag[i] = Halide::Runtime::Buffer<float>{
        extrema_maps[i].width(), extrema_maps[i].height(), 1, 1  //
    };
    buffer_ori[i] = Halide::Runtime::Buffer<float>{
        extrema_maps[i].width(), extrema_maps[i].height(), 1, 1  //
    };
  }


  auto buffer_conv_2d = halide::as_runtime_buffer(frame_conv);

  // Show the local extrema.
  sara::create_window(frame.sizes() / downscale_factor);
  sara::set_antialiasing();

  auto frames_read = 0;

  auto timer = sara::Timer{};
  auto elapsed_ms = double{};

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    SARA_CHECK(frames_read);

    timer.restart();

    sara::tic();
    shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray);
    sara::toc("CPU rgb to grayscale");

    sara::tic();
    buffer_gray_4d.set_host_dirty();
    sara::toc("Set host dirty");

    if (downscale_factor > 1)
    {
      sara::tic();
      halide::scale(buffer_gray_4d, buffer_gray_down_4d);
      sara::toc("Downsample");
    }
    auto& buffer_before_conv =
        downscale_factor > 1 ? buffer_gray_down_4d : buffer_gray_4d;

    for (auto i = 0u; i < buffer_convs.size(); ++i)
    {
      auto& conv_in = i == 0u ? buffer_before_conv : buffer_convs[i - 1];
      auto& conv_out = buffer_convs[i];
      sara::tic();
      halide::gaussian_convolution(conv_in, conv_out, sigmas[i], 4);
      sara::toc("Gaussian convolution " + std::to_string(i) + ": " +
                std::to_string(sigmas[i]));
    }

    for (auto i = 0u; i < buffer_dogs.size(); ++i)
    {
      sara::tic();
      halide::subtract(buffer_convs[i + 1], buffer_convs[i], buffer_dogs[i]);
      sara::toc("DoG " + std::to_string(i));
    }

    for (auto i = 0u; i < buffer_mag.size(); ++i)
    {
      sara::tic();
      shakti_polar_gradient_2d_32f_v2(buffer_convs[i + 1], buffer_mag[i],
                                      buffer_ori[i]);
      sara::toc("Gradients in polar coordinates " + std::to_string(i));
    }

    for (auto i = 0u; i < buffer_extremas.size(); ++i)
    {
      sara::tic();
      shakti_local_scale_space_extremum_32f_v2(
          buffer_dogs[i], buffer_dogs[i + 1], buffer_dogs[i + 2],  //
          edge_ratio, extremum_thres,                              //
          buffer_extremas[i]);
      sara::toc("DoG extremum localization " + std::to_string(i));
    }

#if !defined(__APPLE__)
    sara::tic();
    buffer_dogs.back().copy_to_host();
    sara::toc("Copy last DoG buffer to host");
#endif

    sara::tic();
    for (auto i = 0u; i < buffer_extremas.size(); ++i)
      buffer_extremas[i].copy_to_host();
    sara::toc("Copy extrema map buffers to host");


    sara::tic();
    auto extrema_quantized =
        std::vector<halide::QuantizedExtremumArray>(extrema_maps.size());
#pragma omp parallel for
    for (auto s = 0u; s < extrema_maps.size(); ++s)
    {
      const auto& dog_ext_map = extrema_maps[s];
      const auto num_extrema = std::count_if(      //
          dog_ext_map.begin(), dog_ext_map.end(),  //
          [](const auto& v) { return v != 0; }     //
      );

      if (num_extrema == 0)
        continue;

      // Populate the list of extrema for the corresponding scale.
      extrema_quantized[s].resize(num_extrema);
      extrema_quantized[s].scale = scales[s + 1];
      extrema_quantized[s].scale_geometric_factor = scale_factor;

      auto i = 0;
      for (auto y = 0; y < dog_ext_map.height(); ++y)
      {
        for (auto x = 0; x < dog_ext_map.width(); ++x)
        {
          if (dog_ext_map(x, y) == 0)
            continue;

          extrema_quantized[s].x[i] = x;
          extrema_quantized[s].y[i] = y;
          extrema_quantized[s].type[i] = dog_ext_map(x, y);
          ++i;
        }
      }
    }
    sara::toc("Populating list of extrema");


    sara::tic();
    auto extrema = std::vector<halide::ExtremumArray>(extrema_maps.size());
    for (auto s = 0u; s < extrema_quantized.size(); ++s)
    {
      auto& e = extrema_quantized[s];
      if (e.x.empty())
        continue;

      auto x_buffer = halide::as_runtime_buffer(e.x);
      auto y_buffer = halide::as_runtime_buffer(e.y);
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();

      auto& e_refined = extrema[s];
      e_refined.resize(e.x.size());
      e_refined.type = e.type;
      auto xf_buffer = halide::as_runtime_buffer(e_refined.x);
      auto yf_buffer = halide::as_runtime_buffer(e_refined.y);
      auto sf_buffer = halide::as_runtime_buffer(e_refined.s);
      auto value_buffer = halide::as_runtime_buffer(e_refined.value);

      shakti_refine_scale_space_extrema_v2(
          buffer_dogs[s], buffer_dogs[s + 1], buffer_dogs[s + 2],  //
          x_buffer, y_buffer,                                      //
          buffer_dogs[s].width(), buffer_dogs[s].height(),         //
          e.scale,                                                 //
          e.scale_geometric_factor,                                //
          xf_buffer,                                               //
          yf_buffer,                                               //
          sf_buffer,                                               //
          value_buffer);                                           //

      xf_buffer.copy_to_host();
      yf_buffer.copy_to_host();
      sf_buffer.copy_to_host();
      value_buffer.copy_to_host();
    }
    sara::toc("Refined extrema");


    sara::tic();
    auto dominant_orientations_dense =
        std::vector<halide::DominantOrientationDenseMap>(extrema.size());
    for (auto s = 0u; s < dominant_orientations_dense.size(); ++s)
    {
      // Inputs.
      auto& e = extrema[s];
      if (e.x.size() == 0)
        continue;

      auto x_buffer = halide::as_runtime_buffer(e.x);
      auto y_buffer = halide::as_runtime_buffer(e.y);
      auto scale_buffer = halide::as_runtime_buffer(e.s);
      const auto& scale_max = *std::max_element(e.s.begin(), e.s.end());

      // TODO: optimize this.
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();

      //  Outputs.
      auto& d = dominant_orientations_dense[s];
      d.resize(static_cast<std::int32_t>(e.size()), num_orientation_bins);
      auto peak_map_buffer = halide::as_runtime_buffer(d.peak_map);
      auto peak_residuals_buffer = halide::as_runtime_buffer(d.peak_residuals);

      shakti_dominant_gradient_orientations_v2(buffer_mag[s], buffer_ori[s],  //
                                               x_buffer,                      //
                                               y_buffer,                      //
                                               scale_buffer,                  //
                                               scale_max,                     //
                                               num_orientation_bins,          //
                                               gaussian_truncation_factor,    //
                                               scale_multiplying_factor,      //
                                               peak_ratio_thres,              //
                                               peak_map_buffer,               //
                                               peak_residuals_buffer);
      peak_map_buffer.copy_to_host();
      peak_residuals_buffer.copy_to_host();
    }
    sara::toc("Dense dominant gradient orientations");

    sara::tic();
    auto dominant_orientations =
        std::vector<halide::DominantOrientationMap>(extrema.size());
#pragma omp parallel for
    for (auto s = 0u; s < dominant_orientations_dense.size(); ++s)
    {
      auto& d = dominant_orientations_dense[s];
      auto& dsparse = dominant_orientations[s];
      dsparse = halide::DominantOrientationMap{halide::compress(d)};
    }
    sara::toc("Sparse dominant gradient orientations");


    sara::tic();
    auto oriented_extrema =
        std::vector<halide::OrientedExtremumArray>(extrema.size());
#pragma omp parallel for
    for (auto s = 0u; s < oriented_extrema.size(); ++s)
      oriented_extrema[s] = halide::to_oriented_extremum_array(
          extrema[s], dominant_orientations[s]);
    sara::toc("Populating oriented extrema");


    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[" << frames_read
               << "] total computation time = " << elapsed_ms << " ms"
               << std::endl;


    sara::tic();
    sara::clear_window();
    for (const auto& e : oriented_extrema)
    {
      for (auto i = 0u; i < e.x.size(); ++i)
      {
        const auto& color = e[i].type == 1 ? sara::Cyan8 : sara::Magenta8;
        const Eigen::Vector2f xy = {e[i].x, e[i].y};
        const auto& theta = e[i].orientation;

        // N.B.: the blob radius is the scale multiplied sqrt(2).
        // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
        const float r = e[i].s * M_SQRT2;
        const auto& p1 = xy;
        const Eigen::Vector2f& p2 =
            xy + r * Eigen::Vector2f{cos(theta), sin(theta)};

        sara::draw_line(p1, p2, color, 2);
        sara::draw_circle(xy, r, color, 2);
      }
    }
    sara::toc("Display");
  }

  return 0;
}


auto impl2() -> int
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // "/home/david/Desktop/GOPR0542.MP4"s;
#endif


  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);


  // Successive Gaussian convolutions.
  auto scales = std::vector<float>(num_scales);
  for (auto i = 0; i < num_scales; ++i)
    scales[i] = scale_initial * std::pow(scale_factor, i);

  auto sigmas = std::vector<float>(num_scales);
  for (auto i = 0u; i < sigmas.size(); ++i)
  {
    sigmas[i] =
        i == 0
            ? std::sqrt(std::pow(scale_initial, 2) - std::pow(scale_camera, 2))
            : std::sqrt(std::pow(scales[i], 2) - std::pow(scales[i - 1], 2));
  }


  // ===========================================================================
  // SARA PIPELINE
  //
  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray = sara::Image<float>{frame.sizes()};
  auto frame_gray_tensor =
      tensor_view(frame_gray)
          .reshape(
              Eigen::Vector4i{1, 1, frame_gray.height(), frame_gray.width()});

  auto frame_conv = sara::Image<float>{frame.sizes() / downscale_factor};
  auto frame_conv_tensor =
      tensor_view(frame_conv)
          .reshape(
              Eigen::Vector4i{1, 1, frame_conv.height(), frame_conv.width()});


  auto extrema_maps = std::vector<sara::Image<std::int8_t>>(
      num_scales - 3, sara::Image<std::int8_t>{frame_conv.sizes()});


  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);

  // Downsampling.
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);
  auto buffer_gray_down_4d = Halide::Runtime::Buffer<float>(
      frame_conv.width(), frame_conv.height(), 1, 1);


  // Show the local extrema.
  sara::create_window(frame.sizes() / downscale_factor);
  sara::set_antialiasing();

  auto frames_read = 0;

  auto timer = sara::Timer{};
  auto elapsed_ms = double{};

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    SARA_CHECK(frames_read);

    timer.restart();

    sara::tic();
    shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray);
    sara::toc("CPU rgb to grayscale");





    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[" << frames_read
               << "] total computation time = " << elapsed_ms << " ms"
               << std::endl;


    sara::tic();
    sara::clear_window();
    for (const auto& e : oriented_extrema)
    {
      for (auto i = 0u; i < e.x.size(); ++i)
      {
        const auto& color = e[i].type == 1 ? sara::Cyan8 : sara::Magenta8;
        const Eigen::Vector2f xy = {e[i].x, e[i].y};
        const auto& theta = e[i].orientation;

        // N.B.: the blob radius is the scale multiplied sqrt(2).
        // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
        const float r = e[i].s * M_SQRT2;
        const auto& p1 = xy;
        const Eigen::Vector2f& p2 =
            xy + r * Eigen::Vector2f{cos(theta), sin(theta)};

        sara::draw_line(p1, p2, color, 2);
        sara::draw_circle(xy, r, color, 2);
      }
    }
    sara::toc("Display");
  }

  return 0;
}

GRAPHICS_MAIN()
{
  return impl1();
}
