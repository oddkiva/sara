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
#include "shakti_sift_octave.h"


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
      // "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/home/david/Desktop/GOPR0542.MP4"s;
#endif






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

  auto frame_conv_as_rgb = sara::Image<sara::Rgb8>{frame_conv.sizes()};





  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);

  // Downsampling.
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);
  // auto buffer_gray_down_4d = halide::as_runtime_buffer(frame_down_tensor);
  auto buffer_gray_down_4d = Halide::Runtime::Buffer<float>(
      frame.width() / downscale_factor, frame.height() / downscale_factor, 1, 1);

  // Successive Gaussian convolutions.
  const auto scale_initial = 1.6f;
  const auto scale_factor = std::pow(2.f, 1 / 3.f);
  const auto num_scales = 6;
  auto scales = std::vector<float>(num_scales);
  for (auto i = 0; i < num_scales; ++i)
    scales[i] = scale_initial * std::pow(scale_factor, i);

  auto sigmas = std::vector<float>(num_scales - 1);
  for (auto i = 0u; i < sigmas.size(); ++i)
    sigmas[i] = std::sqrt(std::pow(scales[i + 1], 2) - std::pow(scales[i], 2));

  auto buffer_convs =
      std::vector<Halide::Runtime::Buffer<float>>(sigmas.size());
  for (auto i = 0u; i < buffer_convs.size(); ++i)
  {
    buffer_convs[i] =
        i != buffer_convs.size() - 1
            ? Halide::Runtime::Buffer<float>(buffer_gray_down_4d.width(),
                                             buffer_gray_4d.height(), 1, 1)
            : halide::as_runtime_buffer(frame_conv_tensor);
  }

  auto buffer_conv_2d = halide::as_runtime_buffer(frame_conv);
  auto buffer_conv_as_rgb =
      halide::as_interleaved_runtime_buffer(frame_conv_as_rgb);

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

    for (auto i = 0u; i < sigmas.size(); ++i)
    {
      auto& conv_in = i == 0u? buffer_before_conv : buffer_convs[i - 1];
      auto& conv_out = buffer_convs[i];
      sara::tic();
      halide::gaussian_convolution(conv_in, conv_out, sigmas[i], 4);
      sara::toc("Gaussian convolution " + std::to_string(i) + ": " + std::to_string(sigmas[i]));
    }

    sara::tic();
    buffer_convs.back().copy_to_host();
    sara::toc("Copy to host");

    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[" << frames_read
               << "] total computation time = " << elapsed_ms << " ms"
               << std::endl;


    sara::tic();
    shakti_halide_gray32f_to_rgb(buffer_conv_2d, buffer_conv_as_rgb);
    sara::toc("Convert conv to RGB");

    sara::tic();
    sara::display(frame_conv_as_rgb);
    sara::toc("Display");
  }

  return 0;
}
