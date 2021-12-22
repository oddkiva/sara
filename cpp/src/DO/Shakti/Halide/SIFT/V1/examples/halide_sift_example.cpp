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

#ifdef USE_SHAKTI_CUDA_VIDEOIO
#  include <DO/Shakti/Cuda/VideoIO.hpp>
#endif

#include <DO/Shakti/Halide/SIFT/Draw.hpp>
#include <DO/Shakti/Halide/SIFT/V1/Pipeline.hpp>

#include "shakti_bgra8u_to_gray32f_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"


namespace shakti = DO::Shakti;


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  // const auto image_filepath =
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  auto sift_extractor = halide::SIFTExtractor{};
  sift_extractor.params.initial_pyramid_octave = -1;
  auto timer = sara::Timer{};

  timer.restart();
  sift_extractor(image);
  SARA_DEBUG << "Halide SIFT computation time: "  //
             << timer.elapsed_ms() << " ms" << std::endl;
  SARA_CHECK(sift_extractor.pipeline.num_keypoints());

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_extrema(sift_extractor.pipeline.oriented_extrema);
  sara::get_key();
}

auto test_on_video(int argc, char** argv)
{
  using namespace std::string_literals;

  const auto video_filepath =
      argc < 2
          ?
#ifdef _WIN32
          "C:/Users/David/Desktop/GOPR0542.MP4"s
#elif __APPLE__
          "/Users/david/Desktop/Datasets/videos/sample10.mp4"s
#else
          "/home/david/Desktop/Datasets/sfm/Family.mp4"s
          // "/home/david/Desktop/GOPR0542.MP4"s;
#endif
          : argv[1];

  // Input and output from Sara.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  // Initialize CUDA driver.
  DriverApi::init();

  // Create a CUDA context so that we can use the GPU device.
  const auto gpu_id = 0;
  auto cuda_context = DriverApi::CudaContext{gpu_id};
  cuda_context.make_current();

  // nVidia's hardware accelerated video decoder.
  shakti::VideoStream video_stream{video_filepath, cuda_context};
  auto frame =
      sara::Image<sara::Bgra8>{video_stream.width(), video_stream.height()};
  auto device_bgra_buffer =
      DriverApi::DeviceBgraBuffer{video_stream.width(), video_stream.height()};
#else
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
#endif
  auto frame_gray32f = sara::Image<float>{frame.sizes()};

  const auto scale_factor = argc < 3 ? 1 : std::stoi(argv[2]);
  auto frame_downsampled = sara::Image<float>{frame.sizes() / scale_factor};

  // Halide buffers.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  auto buffer_bgra = halide::as_interleaved_runtime_buffer(frame);
#else
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
#endif
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);

  auto sift_extractor = halide::SIFTExtractor{};

  // Show the local extrema.
  sara::create_window(frame_downsampled.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;
  const auto skip = argc < 4 ? 0 : std::stoi(argv[3]);
  while (true)
  {
    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    const auto has_frame = video_stream.read(device_bgra_buffer);
    sara::toc("Read frame");
    if (!has_frame)
      break;

    sara::tic();
    device_bgra_buffer.to_host(frame);
    sara::toc("Copy to host");
#else
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
#endif
    sara::toc("Video Decoding");

    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    // Use parallelization and vectorization.
    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    shakti_bgra8u_to_gray32f_cpu(buffer_bgra, buffer_gray32f);
#else
    shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray32f);
#endif
    sara::toc("Grayscale");

    if (scale_factor != 1)
    {
      // Use parallelization and vectorization.
      sara::tic();
      halide::scale(frame_gray32f, frame_downsampled);
      sara::toc("Downsample");
    }

    auto& frame_to_process = scale_factor == 1  //
                                 ? frame_gray32f
                                 : frame_downsampled;

    sara::tic();
    sift_extractor(frame_to_process);
    sara::toc("Oriented DoG");

    sara::tic();
    sara::display(frame_to_process);
    draw_extrema(sift_extractor.pipeline.oriented_extrema);
    sara::toc("Display");
  }
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  // test_on_image();
  test_on_video(argc, argv);
  return 0;
}
