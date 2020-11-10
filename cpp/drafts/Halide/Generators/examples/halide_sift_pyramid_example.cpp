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

#include <drafts/Halide/Draw.hpp>
#include <drafts/Halide/SIFTPipeline.hpp>

#include "shakti_halide_rgb_to_gray.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  auto image = sara::imread<float>(image_filepath);
  auto image_tensor = tensor_view(image).reshape(
      Eigen::Vector4i{1, 1, image.height(), image.width()});
  auto buffer_4d = halide::as_runtime_buffer(image_tensor);

  auto sift_pipeline = halide::v2::SiftPyramidPipeline{};
  sift_pipeline.initialize(-1, image.width(), image.height());

  auto timer = sara::Timer{};

  timer.restart();
  {
    buffer_4d.set_host_dirty();
    sift_pipeline.feed(buffer_4d);
  }
  const auto elapsed_ms = timer.elapsed_ms();
  SARA_DEBUG << "SIFT pipeline: " << elapsed_ms << " ms" << std::endl;


#ifdef CHECK_INPUT_UPSCALED
  if (sift_pipeline.start_octave_index < 0)
  {
    auto input_upscaled = sift_pipeline.input_upscaled_view();
    sara::create_window(input_upscaled.sizes());
    sara::display(input_upscaled);
    sara::get_key();
    sara::resize_window(image.sizes());
  }
#endif

  if (!sara::active_window())
    sara::create_window(image.sizes());

// #define CHECK_PYRAMIDS
#ifdef CHECK_PYRAMIDS
  for (auto& octave : sift_pipeline.octaves)
    for (auto s = 0; s < octave.params.num_scales + 3; ++s)
      sara::display(octave.gaussian_view(s));
  sara::get_key();

  for (auto& octave : sift_pipeline.octaves)
    for (auto s = 0; s < octave.params.num_scales + 2; ++s)
    {
      sara::display(sara::color_rescale(octave.dog_view(s)));
      sara::get_key();
    }
  sara::get_key();
#endif

  sara::set_antialiasing();
  sara::display(image);
  for (auto o = 0u; o < sift_pipeline.octaves.size(); ++o)
  {
    auto& octave = sift_pipeline.octaves[o];
    for (auto s = 0u; s < octave.extrema_oriented.size(); ++s)
    {
      SARA_DEBUG << sara::format("[o = %d, s = %d] Num extrema = %d",
                                 sift_pipeline.start_octave_index + o, s,
                                 octave.extrema_oriented[s].size())
                 << std::endl;
      draw_oriented_extrema(octave.extrema_oriented[s],
                            sift_pipeline.octave_scaling_factor(
                                sift_pipeline.start_octave_index + o));
    }
  }

  while (sara::get_key() != sara::KEY_ESCAPE)
    ;
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath =
      // "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/home/david/Desktop/GOPR0542.MP4"s;
      // "/home/david/Desktop/Datasets/ha/turn_bikes.mp4"s;
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

  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);

  auto sift_pipeline = halide::v2::SiftPyramidPipeline{};

  const auto start_octave_index = 0;
  sift_pipeline.initialize(start_octave_index, frame.width(), frame.height());


  // Show the local extrema.
  sara::create_window(frame.sizes());
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

    // if(frames_read % 3 != 0)
    //   continue;

    timer.restart();
    {
      sara::tic();
      shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray);
      sara::toc("CPU RGB to grayscale");

      buffer_gray_4d.set_host_dirty();
      sift_pipeline.feed(buffer_gray_4d);
    }
    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[Frame: " << frames_read << "] "
               << "total computation time = " << elapsed_ms << " ms"
               << std::endl;

    sara::tic();
    for (auto o = 0u; o < sift_pipeline.octaves.size(); ++o)
    {
      auto& octave = sift_pipeline.octaves[o];
      for (auto s = 0u; s < octave.extrema_oriented.size(); ++s)
        draw_oriented_extrema(frame, octave.extrema_oriented[s],
                              sift_pipeline.octave_scaling_factor(
                                  sift_pipeline.start_octave_index + o));
    }
    sara::display(frame);
    sara::toc("Display");
  }
}


GRAPHICS_MAIN()
{
  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);

  // test_on_image();
  test_on_video();
  return 0;
}
