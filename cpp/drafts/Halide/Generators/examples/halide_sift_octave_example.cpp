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

#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_local_scale_space_extremum_32f_v2.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
#else
  const auto video_filepath =
      "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
      // "/home/david/Desktop/GOPR0542.MP4"s;
#endif


  // SIFT octave parameters.
  const auto scale_camera = 1.f;
  const auto scale_initial = 1.6f;
  const auto scale_factor = std::pow(2.f, 1 / 3.f);
  const auto num_scales = 6;
  const auto edge_ratio = 10.0f;
  const auto extremum_thres = 0.03f;

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

  const auto downscale_factor = 1;
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
      frame.width() / downscale_factor, frame.height() / downscale_factor, 1, 1);

  // Octave of Gaussians.
  auto buffer_convs =
      std::vector<Halide::Runtime::Buffer<float>>(sigmas.size());
  for (auto i = 0u; i < buffer_convs.size(); ++i)
    buffer_convs[i] = Halide::Runtime::Buffer<float>(
        buffer_gray_down_4d.width(), buffer_gray_4d.height(), 1, 1);

  // Octave of difference of Gaussians.
  auto buffer_dogs =
      std::vector<Halide::Runtime::Buffer<float>>(buffer_convs.size() - 1);
  for (auto i = 0u; i < buffer_dogs.size(); ++i)
  {
    buffer_dogs[i] =
        i != buffer_dogs.size() - 1
            ? Halide::Runtime::Buffer<float>(buffer_gray_down_4d.width(),
                                             buffer_gray_4d.height(), 1, 1)
            : halide::as_runtime_buffer(frame_conv_tensor);
  }

  auto buffer_extremas = std::vector<Halide::Runtime::Buffer<std::int8_t>>(num_scales - 3);
  for (auto i = 0u; i < extrema_maps.size(); ++i)
    buffer_extremas[i] = Halide::Runtime::Buffer<std::int8_t>{
        extrema_maps[i].data(),                                  //
        extrema_maps[i].width(), extrema_maps[i].height(), 1, 1  //
    };


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

    if (downscale_factor != 1)
    {
      sara::tic();
      halide::scale(buffer_gray_4d, buffer_gray_down_4d);
      sara::toc("Downsample");
    }
    auto& buffer_before_conv =
        downscale_factor == 1 ? buffer_gray_4d : buffer_gray_down_4d;

    for (auto i = 0u; i < buffer_convs.size(); ++i)
    {
      auto& conv_in = i == 0u? buffer_before_conv : buffer_convs[i - 1];
      auto& conv_out = buffer_convs[i];
      sara::tic();
      halide::gaussian_convolution(conv_in, conv_out, sigmas[i], 4);
      sara::toc("Gaussian convolution " + std::to_string(i) + ": " + std::to_string(sigmas[i]));
    }

    for (auto i = 0u; i < buffer_dogs.size(); ++i)
    {
      sara::tic();
      halide::subtract(buffer_convs[i + 1], buffer_convs[i], buffer_dogs[i]);
      sara::toc("DoG " + std::to_string(i));
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

    sara::tic();
    buffer_dogs.back().copy_to_host();
    sara::toc("Copy last DoG buffer to host");

    sara::tic();
    for (auto i = 0u; i < buffer_extremas.size(); ++i)
      buffer_extremas[i].copy_to_host();
    sara::toc("Copy extrema map buffers to host");

    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[" << frames_read
               << "] total computation time = " << elapsed_ms << " ms"
               << std::endl;


    sara::tic();
    sara::display(frame);
    sara::toc("Display");
  }

  return 0;
}
