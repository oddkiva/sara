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

#include <DO/Shakti/Halide/Draw.hpp>
#include <DO/Shakti/Halide/SIFT/PipelineV2.hpp>

#include "shakti_rgb8u_to_gray32f_cpu.h"


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

  auto sift_octave_pipeline = halide::v2::SiftOctavePipeline{};
  sift_octave_pipeline.initialize_buffers(image.width(), image.height());

  auto timer = sara::Timer{};
  timer.restart();
  {
    buffer_4d.set_host_dirty();
    sift_octave_pipeline.feed(buffer_4d);
  }
  const auto elapsed_ms = timer.elapsed_ms();
  SARA_DEBUG << "SIFT octave: " << elapsed_ms << " ms" << std::endl;


  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::tic();
  sara::display(image);
  for (auto s = 0u; s < sift_octave_pipeline.extrema_oriented.size(); ++s)
    draw_oriented_extrema(sift_octave_pipeline.extrema_oriented[s]);
  sara::toc("Display");

  sara::get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath = "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath =
      // "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/home/david/Desktop/Datasets/sfm/oddkiva/bali-excursion.mp4"s;
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

  auto sift_octave_pipeline = halide::v2::SiftOctavePipeline{};
  sift_octave_pipeline.initialize_buffers(frame.width(), frame.height());


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

    sara::tic();
    shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray);
    sara::toc("CPU rgb to grayscale");

    sara::tic();
    buffer_gray_4d.set_host_dirty();
    sara::toc("Set host dirty");

    timer.restart();
    sift_octave_pipeline.feed(buffer_gray_4d);
    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[" << frames_read
               << "] total computation time = " << elapsed_ms << " ms"
               << std::endl;

    sara::tic();
#ifdef CHECK_GAUSSIANS
    for (auto i = 0; i < sift_octave_pipeline.params.num_scales + 3; ++i)
    {
      sara::display(sift_octave_pipeline.gaussian_view(i));
      sara::get_key();
    }
#endif

#ifdef CHECK_DOGS
    for (auto i = 0; i < sift_octave_pipeline.params.num_scales + 2; ++i)
    {
      sara::display(sara::color_rescale(sift_octave_pipeline.dog_view(i)));
      sara::get_key();
    }
#endif

#ifdef CHECK_QUANTIZED_EXTREMA
    for (auto s = 0u; s < sift_octave_pipeline.extrema_quantized.size(); ++s)
      draw_quantized_extrema(sift_octave_pipeline.extrema_quantized[s],
                             sift_octave_pipeline.params.scales[s + 1]);
#endif

#ifdef CHECK_REFINED_EXTREMA
    for (auto s = 0u; s < sift_octave_pipeline.extrema.size(); ++s)
    {
      sift_octave_pipeline.extrema[s].x.copy_to_host();
      sift_octave_pipeline.extrema[s].y.copy_to_host();
      sift_octave_pipeline.extrema[s].s.copy_to_host();
      draw_extrema(sift_octave_pipeline.extrema[s]);
    }
#endif

    for (auto s = 0u; s < sift_octave_pipeline.extrema_oriented.size(); ++s)
      draw_oriented_extrema(frame, sift_octave_pipeline.extrema_oriented[s]);
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
